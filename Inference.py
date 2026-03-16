import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import zipfile
import shutil
import cv2

# ── Post-processing parameters (from Phase 1 Optuna) ─────────
# Update BEST_MIN_SIZES and prob shift values with the output
BEST_MIN_SIZES = {
    0: 40,
    1: 0,
    2: 50,
    3: 10,
    4: 50,
    5: 40,
    6: 0,
    7: 40,
    8: 30,
    9: 30,
    10: 30,
    11: 40,
    12: 30,
    13: 250,
    14: 400,
    15: 15,
    16: 10,
    17: 0,
    18: 100,
}

def remove_small_blobs(mask):
    clean = mask.copy()
    for c, min_size in BEST_MIN_SIZES.items():
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


# ── Setup ─────────────────────────────────────────────────────
# 0:bg 1:skin 2:nose 3:eye_g 4:l_eye 5:r_eye 6:l_brow 7:r_brow
# 8:l_ear 9:r_ear 10:mouth 11:u_lip 12:l_lip 13:hair 14:hat
# 15:ear_r 16:neck_l 17:neck 18:cloth
FLIP_PAIRS = {4:5, 5:4, 6:7, 7:6, 8:9, 9:8}
SCALES     = [0.75, 0.875, 1.0, 1.125, 1.25]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_MODEL_NAME = 'best_model_phase2_shadow.pth'
model = WiderUNet().to(device)
model.load_state_dict(torch.load(BEST_MODEL_NAME, map_location=device, weights_only=True))
model.eval()
print(f"Loaded weights: {BEST_MODEL_NAME}")

val_image_dir = 'test/images'
output_dir    = 'masks'
zip_name      = 'masks.zip'

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"Scales: {SCALES}  |  {len(SCALES) * 2} forward passes per image")

# ── Inference ─────────────────────────────────────────────────
with torch.no_grad():
    for img_name in sorted(os.listdir(val_image_dir)):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        image      = Image.open(os.path.join(val_image_dir, img_name)).convert("RGB")
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

        prob_final = prob_total / len(SCALES)

        # ── Prob shift (Phase 1 Optuna best params) ───────────
        prob_final[:, 0 , :, :] *= 1.205
        prob_final[:, 1 , :, :] *= 1.060
        prob_final[:, 2 , :, :] *= 0.892
        prob_final[:, 3 , :, :] *= 2.837
        prob_final[:, 4 , :, :] *= 0.801
        prob_final[:, 5 , :, :] *= 1.063
        prob_final[:, 6 , :, :] *= 0.876
        prob_final[:, 7 , :, :] *= 1.155
        prob_final[:, 8 , :, :] *= 1.001
        prob_final[:, 9 , :, :] *= 1.066
        prob_final[:, 10, :, :] *= 1.084
        prob_final[:, 11, :, :] *= 0.933
        prob_final[:, 12, :, :] *= 0.980
        prob_final[:, 13, :, :] *= 1.064
        prob_final[:, 14, :, :] *= 0.769
        prob_final[:, 15, :, :] *= 1.246
        prob_final[:, 16, :, :] *= 1.609
        prob_final[:, 17, :, :] *= 0.807
        prob_final[:, 18, :, :] *= 1.228

        pred_mask = prob_final.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_mask = remove_small_blobs(pred_mask)

        out_name = os.path.splitext(img_name)[0] + '.png'
        Image.fromarray(pred_mask, mode='L').save(
            os.path.join(output_dir, out_name), format='PNG')

# ── Package for submission ────────────────────────────────────
print(f"Packing {zip_name} ...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            zipf.write(os.path.join(output_dir, file), arcname=file)

print(f"Done! Upload {zip_name} to CodaBench.")