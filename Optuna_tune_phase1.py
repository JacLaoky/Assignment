import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import optuna
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ─────────────────────────────────────────────────────
MODEL_PATH     = 'best_model_phase1_baseline.pth'
TRAIN_IMG_DIR  = 'train/images'
TRAIN_MASK_DIR = 'train/masks'
CACHE_DIR      = 'prob_cache_phase1_val'
os.makedirs(CACHE_DIR, exist_ok=True)

all_imgs  = sorted([os.path.join(TRAIN_IMG_DIR,  f) for f in os.listdir(TRAIN_IMG_DIR)])
all_masks = sorted([os.path.join(TRAIN_MASK_DIR, f) for f in os.listdir(TRAIN_MASK_DIR)])
random.seed(42)
combined = list(zip(all_imgs, all_masks))
random.shuffle(combined)
all_imgs, all_masks = zip(*combined)
val_imgs  = list(all_imgs[900:])
val_masks = list(all_masks[900:])
print(f"Val split: {len(val_imgs)} images (seed=42, indices 900-999)")

# ── Flip pairs for TTA ────────────────────────────────────────
# 0:bg 1:skin 2:nose 3:eye_g 4:l_eye 5:r_eye 6:l_brow 7:r_brow
# 8:l_ear 9:r_ear 10:mouth 11:u_lip 12:l_lip 13:hair 14:hat
# 15:ear_r 16:neck_l 17:neck 18:cloth
FLIP_PAIRS = {4:5, 5:4, 6:7, 7:6, 8:9, 9:8}

# ── Multi-scale TTA settings ─────────────────────────────────
SCALES = [0.75, 0.875, 1.0, 1.125, 1.25]
_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

def base_transform(pil_img):
    return _normalize(TF.to_tensor(pil_img))


# ── Model ─────────────────────────────────────────────────────
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
        x  = self.conv_up3(torch.cat([self.up3(x),  x2], dim=1))
        x  = self.conv_up4(torch.cat([self.up4(x),  x1], dim=1))
        return self.outc(x)


# ── Build prob cache for the 100 val images ───────────────────
def build_prob_cache():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = WiderUNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model: {MODEL_PATH}  |  device: {device}")
    print(f"Scales: {SCALES}  |  {len(SCALES) * 2} forward passes per image")

    skipped = 0
    with torch.no_grad():
        for img_path in val_imgs:
            base       = os.path.splitext(os.path.basename(img_path))[0]
            cache_path = os.path.join(CACHE_DIR, f"{base}.npy")
            if os.path.exists(cache_path):
                skipped += 1
                continue

            image      = Image.open(img_path).convert("RGB")
            prob_total = torch.zeros((1, 19, 512, 512), device=device)

            for s in SCALES:
                size  = int(round(512 * s / 16.0) * 16)
                img_s = TF.resize(image, (size, size), interpolation=TF.InterpolationMode.BILINEAR)
                x_s   = base_transform(img_s).unsqueeze(0).to(device)

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    prob_orig = F.softmax(model(x_s), dim=1)
                    prob_flip = F.softmax(model(torch.flip(x_s, dims=[3])), dim=1)

                prob_flip = torch.flip(prob_flip, dims=[3])
                prob_flip_fixed = prob_flip.clone()
                for src, dst in FLIP_PAIRS.items():
                    prob_flip_fixed[:, dst] = prob_flip[:, src]

                prob_s = (prob_orig + prob_flip_fixed) / 2.0
                if s != 1.0:
                    prob_s = F.interpolate(prob_s, size=(512, 512),
                                           mode='bilinear', align_corners=False)
                prob_total += prob_s

            prob_avg = (prob_total / len(SCALES)).cpu().numpy()  # (1, 19, 512, 512)
            np.save(cache_path, prob_avg)

    new = len(val_imgs) - skipped
    print(f"Cache ready: {new} new, {skipped} already cached.")


# ── F-score ───────────────────────────────────────────────────
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


# ── Morphological cleanup ─────────────────────────────────────
def remove_small_blobs_dynamic(mask, min_sizes):
    clean = mask.copy()
    for c, min_size in min_sizes.items():
        if min_size == 0:
            continue
        class_map = (clean == c).astype(np.uint8)
        if class_map.sum() == 0:
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_map, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                clean[labels == i] = 0
    return clean


# ── Joint evaluation ──────────────────────────────────────────
def evaluate_joint_params(shift_dict, min_size_dict):
    total, count = 0.0, 0
    for img_path, mask_path in zip(val_imgs, val_masks):
        base       = os.path.splitext(os.path.basename(img_path))[0]
        cache_path = os.path.join(CACHE_DIR, f"{base}.npy")
        if not os.path.exists(cache_path):
            continue

        prob = np.load(cache_path).copy()   # (1, 19, 512, 512)
        gt   = np.array(Image.open(mask_path), dtype=np.uint8)

        for class_id, mult in shift_dict.items():
            prob[0, class_id] *= mult

        pred = prob.argmax(axis=1).squeeze(0).astype(np.uint8)
        pred = remove_small_blobs_dynamic(pred, min_size_dict)

        total += compute_multiclass_fscore(gt, pred)
        count += 1

    return total / max(count, 1)


# ── Optuna objective ──────────────────────────────────────────
def objective(trial):
    shift_dict    = {}
    min_size_dict = {}

    for i in range(19):
        if i == 0:
            shift_dict[i] = trial.suggest_float(f's_{i}', 1.0, 1.5)
        elif i in [1, 2, 13, 14]:
            shift_dict[i] = trial.suggest_float(f's_{i}', 0.7, 1.1)
        elif i == 18:
            shift_dict[i] = trial.suggest_float(f's_{i}', 0.8, 1.3)
        elif i in [8, 9]:
            shift_dict[i] = trial.suggest_float(f's_{i}', 1.0, 2.0)
        elif i in [3, 15, 16]:
            shift_dict[i] = trial.suggest_float(f's_{i}', 1.2, 3.0)
        else:
            shift_dict[i] = trial.suggest_float(f's_{i}', 0.8, 1.3)

        if i in [13, 14, 18]:
            min_size_dict[i] = trial.suggest_int(f'z_{i}', 100, 400, step=50)
        elif i in [3, 15, 16]:
            min_size_dict[i] = trial.suggest_int(f'z_{i}', 0, 15, step=5)
        else:
            min_size_dict[i] = trial.suggest_int(f'z_{i}', 0, 50, step=10)

    return evaluate_joint_params(shift_dict, min_size_dict)

if __name__ == '__main__':
    print("Step 1: Building probability cache for Phase 1 val split...")
    build_prob_cache()

    print("\nStep 2: Running Optuna joint optimisation...")
    study = optuna.create_study(
        study_name='joint_tuning_phase1_val',
        storage='sqlite:///joint_history_phase1_val.db',
        load_if_exists=True,
        direction='maximize',
    )
    study.optimize(objective, n_trials=200)

    print("\n" + "=" * 60)
    print(f"Best mF-score : {study.best_value:.8f}")
    print("Best parameters:")
    print("-" * 60)

    shifts = {int(k.split('_')[1]): v
              for k, v in study.best_params.items() if k.startswith('s_')}
    sizes  = {int(k.split('_')[1]): v
              for k, v in study.best_params.items() if k.startswith('z_')}

    print("# Prob shift ")
    for i in range(19):
        if i in shifts and abs(shifts[i] - 1.0) > 1e-4:
            print(f"prob_final[:, {i:<2}, :, :] *= {shifts[i]:.4f}")

    print("\nBEST_MIN_SIZES = {")
    for i in range(19):
        print(f"    {i}: {sizes.get(i, 0)},")
    print("}")
    print("=" * 60)