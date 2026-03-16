import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import random
import datetime
import albumentations as A

_shadow_aug = A.RandomShadow(
    shadow_roi=(0, 0, 1, 1),
    num_shadows_limit=(1, 2),
    shadow_dimension=4,
    p=1.0
)

# ── Logging ──────────────────────────────────────────────────
LOG_FILE         = 'training_history.log'
BEST_RECORD_FILE = 'best_target_record.txt'

def log_msg(msg):
    """Print to terminal and append to log file simultaneously."""
    print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def save_best_record(epoch, train_loss, val_f):
    """Save best record for Phase 2 blind training reference."""
    record = (
        f"BEST MODEL RECORD\n"
        f"Time      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Best Epoch: {epoch}\n"
        f"Train Loss: {train_loss:.4f}, use this as LOSS_TARGET in Phase 2!\n"
        f"Val F-Score: {val_f:.4f}\n"
    )
    with open(BEST_RECORD_FILE, 'w', encoding='utf-8') as f:
        f.write(record)


# ── Model ────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class DilatedBottleneck(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)

class WiderUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=19):
        super().__init__()
        self.inc      = DoubleConv(n_channels, 32)
        self.down1    = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2    = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3    = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 144))
        self.down4    = nn.Sequential(nn.MaxPool2d(2), DilatedBottleneck(144, dropout=0.1), SEBlock(144))
        self.up1      = nn.ConvTranspose2d(144, 128, 2, stride=2)
        self.conv_up1 = nn.Sequential(DoubleConv(128+144, 128, dropout=0.1), SEBlock(128))
        self.up2      = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = nn.Sequential(DoubleConv(64+128, 64, dropout=0.1), SEBlock(64))
        self.up3      = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = nn.Sequential(DoubleConv(32+64, 32, dropout=0.05), SEBlock(32))
        self.up4      = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.conv_up4 = nn.Sequential(DoubleConv(32+32, 32, dropout=0.05), SEBlock(32))
        self.outc     = nn.Conv2d(32, n_classes, 1)
        self.aux2     = nn.Conv2d(64, n_classes, 1)
        self.aux3     = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x);  x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.conv_up1(torch.cat([self.up1(x5), x4], dim=1))
        x  = self.conv_up2(torch.cat([self.up2(x),  x3], dim=1))
        aux2_out = self.aux2(x)
        x  = self.conv_up3(torch.cat([self.up3(x),  x2], dim=1))
        aux3_out = self.aux3(x)
        x  = self.conv_up4(torch.cat([self.up4(x),  x1], dim=1))
        main_out = self.outc(x)
        if self.training:
            return main_out, aux2_out, aux3_out
        return main_out


# ── Loss ─────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.weights = torch.ones(19)
        self.weights[8]  = 3.0   # l_ear
        self.weights[9]  = 3.0   # r_ear
        self.weights[15] = 4.0   # ear_r
        self.weights[13] = 0.5   # hair
        self.weights[14] = 2.0   # hat

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        total = (inputs + targets_one_hot).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (total + self.smooth)
        return 1 - (dice * self.weights.to(inputs.device).unsqueeze(0)).mean()


# ── EMA ──────────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})


# ── Flip pairs for TTA ───────────────────────────────────────
# 0:bg 1:skin 2:nose 3:eye_g 4:l_eye 5:r_eye 6:l_brow 7:r_brow
# 8:l_ear 9:r_ear 10:mouth 11:u_lip 12:l_lip 13:hair 14:hat
# 15:ear_r 16:neck_l 17:neck 18:cloth
FLIP_PAIRS = {4:5, 5:4, 6:7, 7:6, 8:9, 9:8}


# ── Dataset ──────────────────────────────────────────────────
class CelebAMaskDataset(Dataset):
    def __init__(self, img_paths, mask_paths, is_train=False):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.is_train   = is_train
        self.normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx])
        image = TF.resize(image, (512, 512), interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.resize(mask,  (512, 512), interpolation=TF.InterpolationMode.NEAREST)
        if self.is_train:
            if random.random() > 0.3:
                image = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2)(image)
            if random.random() > 0.7:
                image_np = np.array(image)
                image = Image.fromarray(_shadow_aug(image=image_np)["image"])
            if random.random() > 0.5:
                scale     = random.uniform(0.7, 1.3)
                translate = [int(random.uniform(-10, 10)), int(random.uniform(-10, 10))]
                image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0,
                                  interpolation=TF.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask,  angle=0, translate=translate, scale=scale, shear=0,
                                  interpolation=TF.InterpolationMode.NEAREST)
            image = TF.to_tensor(image)
            mask  = torch.from_numpy(np.array(mask, dtype=np.int64))
            if random.random() > 0.5:
                image = transforms.RandomErasing(p=1.0, scale=(0.02, 0.10),
                                                  ratio=(0.3, 3.3), value=0.5)(image)
                if random.random() > 0.5:
                    image = transforms.RandomErasing(p=1.0, scale=(0.02, 0.05),
                                                      ratio=(0.3, 3.3), value=0)(image)
        else:
            image = TF.to_tensor(image)
            mask  = torch.from_numpy(np.array(mask, dtype=np.int64))
        image = self.normalize(image)
        return image, mask


# ── F-score ──────────────────────────────────────────────────
def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f_scores = []
    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f_score   = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-7)
        f_scores.append(f_score)
    return np.mean(f_scores)


# ── Data loading ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_msg(f"Device: {device}")

train_img_dir  = 'train/images'
train_mask_dir = 'train/masks'

all_imgs  = sorted([os.path.join(train_img_dir,  f) for f in os.listdir(train_img_dir)])
all_masks = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir)])

random.seed(42)
random.shuffle(combined := list(zip(all_imgs, all_masks)))
all_imgs, all_masks = zip(*combined)

train_imgs, train_masks = all_imgs[:900], all_masks[:900]
val_imgs,   val_masks   = all_imgs[900:], all_masks[900:]

log_msg(f"Train split: {len(train_imgs)} images")
log_msg(f"Val   split: {len(val_imgs)} images")

train_loader = DataLoader(CelebAMaskDataset(train_imgs, train_masks, is_train=True),
                          batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(CelebAMaskDataset(val_imgs, val_masks, is_train=False),
                          batch_size=4, shuffle=False, num_workers=2, pin_memory=True)


# ── Training setup ───────────────────────────────────────────
BEST_MODEL  = 'best_model_phase1_baseline.pth'
RESUME_CKPT = 'last_checkpoint_phase1.pth'

model = WiderUNet().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log_msg(f"Params: {total_params:,} / 1,821,085")
assert total_params <= 1_821_085, f"Exceeded param limit: {total_params:,}"

ce_loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.05)
dice_loss_fn = DiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
epochs    = 960
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=1, eta_min=1e-5)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
ema    = EMA(model, decay=0.999)

best_val_f  = 0.0
start_epoch = 0

if os.path.exists(RESUME_CKPT):
    ckpt = torch.load(RESUME_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    start_epoch = ckpt['epoch']
    best_val_f  = ckpt['best_val_f']
    if 'ema_shadow' in ckpt:
        ema.shadow = ckpt['ema_shadow']
    log_msg(f"Resumed from epoch {start_epoch}/{epochs}, best F so far: {best_val_f:.4f}")
else:
    log_msg("Starting Phase 1 (900/100 split) from scratch...")

eval_model = WiderUNet().to(device)


# ── Training loop ────────────────────────────────────────────
log_msg("=" * 60)
log_msg("Phase 1 training start! Best val F-score saved to best_target_record.txt")
log_msg("=" * 60)

for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        # CutMix 50%
        if random.random() < 0.5:
            idx = torch.randperm(images.size(0), device=device)
            lam = np.random.beta(1.0, 1.0)
            _, _, H, W = images.shape
            cut_ratio = np.sqrt(1 - lam)
            cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
            cx, cy = np.random.randint(W), np.random.randint(H)
            x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
            y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
            images_cut = images.clone()
            images_cut[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
            masks_cut  = masks.clone()
            masks_cut[:, y1:y2, x1:x2] = masks[idx, y1:y2, x1:x2]
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                main, aux2, aux3 = model(images_cut)
                masks_128 = F.interpolate(masks_cut.unsqueeze(1).float(), size=(128, 128),
                                          mode='nearest').squeeze(1).long()
                masks_256 = F.interpolate(masks_cut.unsqueeze(1).float(), size=(256, 256),
                                          mode='nearest').squeeze(1).long()
                loss = (ce_loss_fn(main, masks_cut) + dice_loss_fn(main, masks_cut)
                      + 0.4 * ce_loss_fn(aux2, masks_128)
                      + 0.4 * ce_loss_fn(aux3, masks_256))
        else:
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                main, aux2, aux3 = model(images)
                masks_128 = F.interpolate(masks.unsqueeze(1).float(), size=(128, 128),
                                          mode='nearest').squeeze(1).long()
                masks_256 = F.interpolate(masks.unsqueeze(1).float(), size=(256, 256),
                                          mode='nearest').squeeze(1).long()
                loss = (ce_loss_fn(main, masks) + dice_loss_fn(main, masks)
                      + 0.4 * ce_loss_fn(aux2, masks_128)
                      + 0.4 * ce_loss_fn(aux3, masks_256))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Apply EMA weights to eval_model after warm-up
    if epoch < 20:
        eval_model.load_state_dict(model.state_dict())
    else:
        ema.apply(eval_model)
    eval_model.eval()

    val_f_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                prob_orig = F.softmax(eval_model(images), dim=1)
                prob_flip = F.softmax(eval_model(torch.flip(images, dims=[3])), dim=1)
            prob_flip = torch.flip(prob_flip, dims=[3])
            prob_flip_fixed = prob_flip.clone()
            for src, dst in FLIP_PAIRS.items():
                prob_flip_fixed[:, dst] = prob_flip[:, src]
            preds = ((prob_orig + prob_flip_fixed) / 2.0).argmax(dim=1).cpu().numpy()
            gts   = masks.cpu().numpy()
            for p, g in zip(preds, gts):
                val_f_scores.append(compute_multiclass_fscore(g, p))

    epoch_val_f = np.mean(val_f_scores)
    status_msg = (f"Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.1e} | "
                  f"Loss: {epoch_train_loss:.4f} | Val F: {epoch_val_f:.4f}")

    if epoch_val_f > best_val_f:
        best_val_f = epoch_val_f
        torch.save(eval_model.state_dict(), BEST_MODEL)
        save_best_record(epoch + 1, epoch_train_loss, best_val_f)
        status_msg += f"  --> New best! Saved {BEST_MODEL}"

    log_msg(status_msg)

    torch.save({
        'epoch':      epoch + 1,
        'model':      model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer':  optimizer.state_dict(),
        'scheduler':  scheduler.state_dict(),
        'scaler':     scaler.state_dict(),
        'best_val_f': best_val_f,
    }, RESUME_CKPT)

log_msg(f"\nPhase 1 complete! Best val F: {best_val_f:.4f}")
log_msg(f"Check {BEST_RECORD_FILE} for the LOSS_TARGET to use in Phase 2 blind training.")