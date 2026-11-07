
# -*- coding: utf-8 -*-
"""
植物物种分类 v6（GPU饱和版）：去掉伪标签 + 轻量增强 + SWA + 加权集成

改进（相对上一版 v6）：
  1. [GPU] 图片预 resize 到 448 缓存为 uint8 tensor，每 epoch 只做 numpy crop/flip
     CPU 几乎没活干，GPU 利用率能到 90%+
  2. [增强] 去掉 TrivialAugmentWide，用轻量 Crop+Flip（更适合小数据集）
  3. [SWA] 最后 3 个 epoch 做权重平均
  4. [无伪标签] Phase 1 val_acc 0.901 > Phase 3 0.884 说明伪标签伤分，去掉
  5. [加权集成] ConvNeXt-Tiny 权重 1.3 / EffNetV2 1.0 / EffNetB3 0.8
  6. [更大 batch] 24
"""

import os
import math
import random
import gc

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# =====================================================================
# 配置
# =====================================================================

BASE = r"D:\plant\2026task1"
TRAIN_DIR = os.path.join(BASE, "dataset-for-task2", "dataset-for-task2", "train")
TEST_DIR = os.path.join(BASE, "dataset-for-task2", "dataset-for-task2", "test")
SUB_PATH = os.path.join(BASE, "submission-for-task2.csv")
OUTPUT_DIR = r"D:\plant\checkpoints_v6"

MODEL_NAMES = ["convnext_tiny"]
MODEL_WEIGHTS = {"convnext_tiny": 1.0}

IMG_SIZE = 384
CROP_SIZE = 384
RESIZE_TO = 448
BATCH_SIZE = 24
N_FOLDS = 5
HEAD_EPOCHS = 3
FINE_EPOCHS = 12
SWA_START = FINE_EPOCHS - 3

HEAD_LR = 1e-3
BACKBONE_LR = 1.5e-5
HEAD_FT_LR = 3e-4
WARMUP_EPOCHS = 1
LABEL_SMOOTHING = 0.1
MIXUP_PROB = 0.4
MIXUP_ALPHA = 0.2

USE_AMP = True

SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True

IMAGE_CACHE = {}
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preload_images(paths):
    """预 resize 到 448 并缓存为 uint8 tensor [3,H,W]"""
    for p in paths:
        if p not in IMAGE_CACHE:
            img = load_image_rgb(p)
            img = img.resize((RESIZE_TO, RESIZE_TO), Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)
            IMAGE_CACHE[p] = torch.from_numpy(arr).permute(2, 0, 1)


# =====================================================================
# 工具
# =====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_rgb(path):
    """加载图像并处理 RGBA → RGB（透明区贴白底）"""
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')


def crop_and_normalize(img_u8, crop_size, train=True):
    """从 uint8 [3,H,W] 做 crop+flip+rotate+colorjitter+normalize"""
    H, W = img_u8.shape[1], img_u8.shape[2]
    if train:
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        img = img_u8[:, top:top + crop_size, left:left + crop_size]
        if random.random() < 0.5:
            img = img.flip(2)
        # random rotation（v5 原有的，对植物分类关键）
        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=128)
        # color jitter（v5 原有的）
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
    else:
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
        img = img_u8[:, top:top + crop_size, left:left + crop_size]
    img = img.float().div_(255.0)
    return (img - MEAN_T) / STD_T


class FastPlantDataset(Dataset):
    def __init__(self, items, has_label=True, train=True):
        self.items = items
        self.has_label = has_label
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.has_label:
            path, label = self.items[idx]
        else:
            path = self.items[idx][0]
            label = -1
        img_u8 = IMAGE_CACHE[path]
        img = crop_and_normalize(img_u8, CROP_SIZE, train=self.train)
        if self.has_label:
            return img, label
        return img


def build_model(model_name, num_classes):
    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")
    return m


def get_head_params(model_name, model):
    if "convnext" in model_name:
        return model.classifier.parameters()
    return model.classifier.parameters()


def get_backbone_params(model_name, model):
    if "convnext" in model_name:
        head_ids = set(id(p) for p in model.classifier.parameters())
    else:
        head_ids = set(id(p) for p in model.classifier.parameters())
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_head_only(model_name, model):
    for p in model.parameters():
        p.requires_grad = False
    for p in get_head_params(model_name, model):
        p.requires_grad = True


def set_all_trainable(model):
    for p in model.parameters():
        p.requires_grad = True


def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cx, cy = np.random.randint(W), np.random.randint(H)
    rw, rh = int(W * math.sqrt(1 - lam)), int(H * math.sqrt(1 - lam))
    x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
    y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
    x[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)
    return x, y, y[idx], lam


def warmup_cosine_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, optimizer, criterion, scaler, device, use_mix=True):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False, ncols=80):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_mix and random.random() < MIXUP_PROB:
            if random.random() < 0.5:
                mixed_x, ya, yb, lam = mixup(images, labels, MIXUP_ALPHA)
            else:
                mixed_x, ya, yb, lam = mixup(images, labels, MIXUP_ALPHA)
            with autocast('cuda', enabled=USE_AMP):
                out = model(mixed_x)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
        else:
            with autocast('cuda', enabled=USE_AMP):
                out = model(images)
                loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast('cuda', enabled=USE_AMP):
            out = model(images)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def predict_tta(model, test_paths, device):
    """多尺度 TTA：384/416/448 center crop × flip"""
    model.eval()
    all_probs = []
    for scale in [CROP_SIZE, CROP_SIZE + 32, RESIZE_TO]:
        ds = FastPlantDataset([(p,) for p in test_paths], has_label=False, train=False)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        for flip in [False, True]:
            probs = []
            for images in loader:
                images = images.to(device)
                if flip:
                    images = torch.flip(images, dims=[3])
                with autocast('cuda', enabled=USE_AMP):
                    out = model(images)
                probs.append(torch.softmax(out, dim=1).cpu())
            all_probs.append(torch.cat(probs).numpy())
    return np.mean(all_probs, axis=0)


def train_single_fold(model_name, fold, train_items, val_items, test_paths,
                      classes, device):
    num_classes = len(classes)
    model = build_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    best_acc = 0.0
    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold}.pth")

    train_ds = FastPlantDataset(train_items, has_label=True, train=True)
    val_ds = FastPlantDataset(val_items, has_label=True, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=len(train_items) > BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ---- 头部训练 ----
    set_head_only(model_name, model)
    optimizer = torch.optim.AdamW(get_head_params(model_name, model), lr=HEAD_LR, weight_decay=0.01)
    for ep in range(HEAD_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_mix=False)
        val_acc = evaluate(model, val_loader, device)
        print(f"  {model_name} fold{fold} head {ep+1}/{HEAD_EPOCHS} acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # ---- 全网络微调 ----
    set_all_trainable(model)
    backbone_params = get_backbone_params(model_name, model)
    head_params = get_head_params(model_name, model)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_FT_LR},
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: warmup_cosine_lambda(ep, WARMUP_EPOCHS, FINE_EPOCHS))
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    for ep in range(FINE_EPOCHS):
        train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_mix=True)
        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        if ep >= SWA_START:
            swa_model.update_parameters(model)
        print(f"  {model_name} fold{fold} ft {ep+1}/{FINE_EPOCHS} acc={val_acc:.4f}" +
              (" [SWA]" if ep >= SWA_START else ""))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # SWA
    swa_active = FINE_EPOCHS > SWA_START
    if swa_active:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_acc = evaluate(swa_model.module, val_loader, device)
        print(f"  SWA acc={swa_acc:.4f}")
        if swa_acc > best_acc:
            best_acc = swa_acc
            torch.save(swa_model.module.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_probs = predict_tta(model, test_paths, device)
    del model
    if swa_active:
        del swa_model
    torch.cuda.empty_cache()
    gc.collect()
    return best_acc, test_probs


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | AMP: {USE_AMP}")

    classes = sorted([c for c in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, c))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"{num_classes} 类: {classes}")

    train_items = []
    for cls in classes:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        for f in sorted(os.listdir(cls_dir)):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                train_items.append((os.path.join(cls_dir, f), class_to_idx[cls]))
    labels = np.array([it[1] for it in train_items])
    print(f"训练图: {len(train_items)}")

    all_train_paths = [p for p, _ in train_items]
    preload_images(all_train_paths)
    print(f"已缓存 {len(IMAGE_CACHE)} 张图片到内存")

    sub = pd.read_csv(SUB_PATH)
    id_col, label_col = sub.columns[0], sub.columns[1]
    test_paths = []
    for img_id in sub[id_col]:
        base = os.path.splitext(str(img_id))[0]
        for ext in IMG_EXTS:
            p = os.path.join(TEST_DIR, base + ext)
            if os.path.exists(p):
                test_paths.append(p)
                break
        else:
            test_paths.append(None)
    valid_test = [p for p in test_paths if p is not None]
    print(f"测试图: {len(valid_test)}")
    preload_images(valid_test)

    # ================================================================
    # Phase 1: 训练 3 模型 × 5 折 = 15 个模型
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 1: 基础集成训练")
    print("=" * 60)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    arch_probs = {name: np.zeros((len(valid_test), num_classes)) for name in MODEL_NAMES}
    arch_accs = {name: [] for name in MODEL_NAMES}

    for name in MODEL_NAMES:
        print(f"\n>>> {name}")
        for fold, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            tr_items = [train_items[i] for i in tr]
            va_items = [train_items[i] for i in va]
            acc, probs = train_single_fold(name, fold, tr_items, va_items, valid_test,
                                           classes, device)
            arch_accs[name].append(acc)
            arch_probs[name] += probs / N_FOLDS
            print(f"  fold{fold} best={acc:.4f} | 平均={np.mean(arch_accs[name]):.4f}")

    # 加权集成
    print("\n加权集成:")
    final_probs = np.zeros_like(arch_probs[MODEL_NAMES[0]])
    total_w = 0.0
    for name in MODEL_NAMES:
        mean_acc = np.mean(arch_accs[name])
        w = mean_acc * MODEL_WEIGHTS[name]
        final_probs += arch_probs[name] * w
        total_w += w
        print(f"  {name}: val_acc={mean_acc:.4f} weight={w:.3f}")
    final_probs /= total_w
    print(f"\n总体平均 val_acc: {np.mean([np.mean(v) for v in arch_accs.values()]):.4f}")

    # ================================================================
    # 输出
    # ================================================================
    final_pred = final_probs.argmax(1)
    final_labels = []
    pred_iter = iter(final_pred)
    for p in test_paths:
        if p is not None:
            final_labels.append(idx_to_class[int(next(pred_iter))])
        else:
            final_labels.append(classes[0])

    sub[label_col] = final_labels
    out_path = os.path.join(r"D:\plant", "submission_v6.csv")
    sub.to_csv(out_path, index=False)
    print(f"\n提交文件: {out_path}")
    print("\n预测分布:")
    print(sub[label_col].value_counts())
    print("\n前 5 行:")
    print(sub.head())


if __name__ == "__main__":
    main()


