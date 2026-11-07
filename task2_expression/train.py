"""2026task2 表情识别：六种预测等权融合，生成 submission.csv。

在 VS Code 选择 D:/py jieshiqi/Scripts/python.exe 后运行：
    python train.py --data D:/2026task2/fer_data

组成：ConvNeXtV2、AffectNet EfficientNet、DDAMFN++、FMAE 线性分类器、MLP 及主干微调。
固定 15% 分层验证集，选择模型后用全部训练图片微调；全量模型不报告验证成绩。
只使用训练标签及公开预训练权重，字节完全相同的训练/测试图片复用训练标签。
首次训练会下载公开权重（最大文件约 3.64 GB），缓存放在 Documents/Codex/fer2026task2_cache。
同一数据已有缓存时直接生成提交；换一个 --work-dir 可以重新训练。
--work-dir 可指定持久缓存目录；--reuse-predictions 用同一数据的缓存重新生成 CSV。
依赖：torch、torchvision、timm==1.0.30、numpy、pillow、scikit-learn、safetensors。
预训练来源：https://github.com/sb-ai-lab/EmotiEffLib
https://github.com/SainingZhang/DDAMFN  https://github.com/forever208/FMAE-IAT
"""

import argparse
import csv
import copy
import hashlib
import math
import random
import tempfile
import time
import importlib
import urllib.request
from torch import nn
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


CLASSES = ("Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise")
SUBMISSION_IDS = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class NormalizedFeatureHead(torch.nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.register_buffer('mean', checkpoint['mean'])
        self.register_buffer('std', checkpoint['std'])
        self.layers = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(1024, 256),
                                          torch.nn.GELU(), torch.nn.Dropout(.5), torch.nn.Linear(256, 6))
        self.layers.load_state_dict(checkpoint['model'])

    def forward(self, x):
        return self.layers((x - self.mean) / self.std)


def load_images(data_dir):
    train_files = []
    labels = []
    for label, name in enumerate(CLASSES):
        paths = sorted((data_dir / "train" / name).glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(data_dir / "train" / name)
        train_files += paths
        labels += [label] * len(paths)
    test_files = sorted((data_dir / "test").glob("*.jpg"))
    if not test_files:
        raise FileNotFoundError(data_dir / "test")

    def read(files):
        images = np.empty((len(files), 1, 48, 48), dtype=np.uint8)
        for i, path in enumerate(files):
            with Image.open(path) as im:
                images[i, 0] = np.asarray(im.convert("L").resize((48, 48)))
        return torch.from_numpy(images)

    print(f"Loading {len(train_files)} train and {len(test_files)} test images", flush=True)
    return train_files, torch.tensor(labels), read(train_files), test_files, read(test_files)


def preprocess(x, size, mean, std, training=False, strength=1.0):
    x = x.float().div_(255)
    if training:
        n = len(x)
        angles = (torch.rand(n, device=x.device) * 2 - 1) * math.radians(10) * strength
        scale = 1 + (torch.rand(n, device=x.device) * .12 - .06) * strength
        c, s = torch.cos(angles) * scale, torch.sin(angles) * scale
        shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * 0.075 * strength
        theta = torch.stack((c, -s, shift[:, 0], s, c, shift[:, 1]), 1).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
        mask = torch.rand(n, device=x.device) < 0.5
        x[mask] = x[mask].flip(-1)
        x = (x * (1 + (torch.rand(n, 1, 1, 1, device=x.device) * .16 - .08) * strength)).clamp_(0, 1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.expand(-1, 3, -1, -1).contiguous(memory_format=torch.channels_last)
    return (x - mean) / std


class ConvBlock(nn.Module):
    def __init__(self, inp, out, kernel=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.prelu = nn.PReLU(out)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(self, inp, out, kernel=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        return self.bn(self.conv(x))


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu6(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = HardSigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    def __init__(self, channels, learned=False):
        super().__init__()
        self.learned = learned
        if learned:
            self.Linear_h = LinearBlock(channels, channels, (1, 7), groups=channels)
            self.Linear_w = LinearBlock(channels, channels, (7, 1), groups=channels)
            # Present in the published checkpoint, although unused in its forward pass.
            self.Linear = LinearBlock(channels, channels, (7, 7), groups=channels)
            self.flatten = nn.Flatten()
        else:
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        hidden = max(8, channels // 32)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)
        self.relu = HardSwish()

    def forward(self, x):
        height, width = x.shape[-2:]
        horizontal = self.Linear_h(x) if self.learned else self.pool_h(x)
        vertical = self.Linear_w(x) if self.learned else self.pool_w(x)
        vertical = vertical.permute(0, 1, 3, 2)
        z = self.relu(self.bn1(self.conv1(torch.cat((horizontal, vertical), dim=2))))
        horizontal, vertical = torch.split(z, (height, width), dim=2)
        horizontal = self.conv2(horizontal).sigmoid()
        vertical = self.conv3(vertical.permute(0, 1, 3, 2)).sigmoid()
        return vertical * horizontal if self.learned else x * vertical * horizontal


class MixedDepthwise(nn.Module):
    def __init__(self, kernels, splits, stride):
        super().__init__()
        self.split_channels = splits
        self.mixed_depthwise_conv = nn.ModuleList([
            nn.Conv2d(c, c, k, stride, k // 2, groups=c, bias=False)
            for k, c in zip(kernels, splits)
        ])
        # These parameters are retained to load the authors' weights strictly.
        self.bn = nn.BatchNorm2d(sum(splits))
        self.prelu = nn.PReLU(sum(splits))

    def forward(self, x):
        parts = torch.split(x, self.split_channels, dim=1)
        return torch.cat([conv(part) for conv, part in zip(self.mixed_depthwise_conv, parts)], dim=1)


class MixedBlock(nn.Module):
    def __init__(self, inp, out, groups, kernels, splits, stride=1, residual=False):
        super().__init__()
        self.conv = ConvBlock(inp, groups)
        self.conv_dw = MixedDepthwise(kernels, splits, stride)
        self.CA = CoordinateAttention(groups)
        self.project = LinearBlock(groups, out)
        self.residual = residual

    def forward(self, x):
        z = self.project(self.CA(self.conv_dw(self.conv(x))))
        return x + z if self.residual else z


class MixedResidual(nn.Module):
    def __init__(self, channels, count, groups, kernels, splits):
        super().__init__()
        self.model = nn.Sequential(*[
            MixedBlock(channels, channels, groups, kernels, splits, residual=True)
            for _ in range(count)
        ])

    def forward(self, x):
        return self.model(x)


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.CoordAtt = CoordinateAttention(512, learned=True)

    def forward(self, x):
        return self.CoordAtt(x)


class DDAMNet(nn.Module):
    def __init__(self, num_class=7, num_head=2, pretrained=False):
        super().__init__()
        if pretrained:
            raise ValueError('Load the published state_dict explicitly')
        self.features = nn.Sequential(
            ConvBlock(3, 64, 3, 2, 1),
            ConvBlock(64, 64, 3, 1, 1, groups=64),
            MixedBlock(64, 64, 128, [3, 5, 7], [64, 32, 32], stride=2),
            MixedResidual(64, 9, 128, [3, 5], [96, 32]),
            MixedBlock(64, 128, 256, [3, 5, 7], [128, 64, 64], stride=2),
            MixedResidual(128, 16, 256, [3, 5], [192, 64]),
            MixedBlock(128, 256, 1024, [3, 5, 7, 9], [256] * 4, stride=2),
            MixedResidual(256, 6, 512, [3, 5, 7], [172, 170, 170]),
            ConvBlock(256, 512),
        )
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, f'cat_head{i}', AttentionHead())
        self.Linear = LinearBlock(512, 512, (7, 7), groups=512)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)
        heads = [getattr(self, f'cat_head{i}')(x) for i in range(self.num_head)]
        mask = heads[0]
        for head in heads[1:]:
            mask = torch.maximum(mask, head)
        return self.fc(self.flatten(self.Linear(x * mask))), x, heads


def make_model(device, args):
    if args.model == 'fmae_affect':
        import timm
        from safetensors.torch import load_file
        model = timm.create_model('vit_large_patch16_224', pretrained=False,
                                 num_classes=8, global_pool='avg', drop_rate=0.1,
                                 drop_path_rate=0.05)
        state = load_file(str(args.weights))
        model.load_state_dict(state, strict=True)
        rows = [6, 4, 1, 0, 2, 3]  # AffectNet: neutral, happy, sad, surprise, fear, disgust, anger, contempt.
        weight, bias = model.head.weight.detach()[rows].clone(), model.head.bias.detach()[rows].clone()
        model.reset_classifier(len(CLASSES), global_pool='avg')
        with torch.no_grad():
            model.head.weight.copy_(weight)
            model.head.bias.copy_(bias)
        if getattr(args, 'mlp_head', None):
            model.head = NormalizedFeatureHead(torch.load(args.mlp_head, map_location='cpu', weights_only=True))
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True)['model'])
        if args.train_blocks < 24:
            model.requires_grad_(False)
            model.head.requires_grad_(True)
            model.fc_norm.requires_grad_(True)
            if args.train_blocks:
                for block in model.blocks[-args.train_blocks:]:
                    block.requires_grad_(True)
        if args.checkpoint_grad:
            model.set_grad_checkpointing(True)
        return model.to(device=device, memory_format=torch.channels_last)
    if args.model.startswith('ddam_'):
        model = DDAMNet(num_class=7, num_head=2, pretrained=False)
        state = torch.load(args.weights, map_location='cpu', weights_only=True)['model_state_dict']
        model.load_state_dict(state, strict=True)
        # The authors' class_names are Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry.
        rows = [6, 4, 1, 0, 2, 3]
        weight, bias = model.fc.weight.detach()[rows].clone(), model.fc.bias.detach()[rows].clone()
        model.fc = torch.nn.Linear(512, len(CLASSES))
        model.bn = torch.nn.BatchNorm1d(len(CLASSES))
        with torch.no_grad():
            model.fc.weight.copy_(weight)
            model.fc.bias.copy_(bias)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True)['model'])
        return model.to(device=device, memory_format=torch.channels_last)
    if args.model.startswith('affectnet_b2_'):
        import timm
        from safetensors.torch import load_file
        n = int(args.model.rsplit('_', 1)[1])
        model = timm.create_model('tf_efficientnet_b2', pretrained=False,
                                 num_classes=n, global_pool='max', drop_rate=0.2)
        if not args.weights:
            raise ValueError('Provide AffectNet safetensors using --weights')
        model.load_state_dict(load_file(str(args.weights)))
        rows = [0, 2, 3, 4, 5, 6] if n == 7 else [0, 3, 4, 5, 6, 7]
        weight, bias = model.classifier.weight.detach()[rows].clone(), model.classifier.bias.detach()[rows].clone()
        model.reset_classifier(len(CLASSES), global_pool='max')
        with torch.no_grad():
            model.classifier.weight.copy_(weight)
            model.classifier.bias.copy_(bias)
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location='cpu', weights_only=True)['model'])
        return model.to(device=device, memory_format=torch.channels_last)
    if args.model == "convnextv2_tiny":
        import timm
        name = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
        model = timm.create_model(name, pretrained=not bool(args.weights),
                                  drop_rate=0.2, drop_path_rate=0.15)
        if args.weights:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(str(args.weights)))
        model.reset_classifier(len(CLASSES))
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True)["model"])
        return model.to(device=device, memory_format=torch.channels_last)
    constructors = {
        "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, 1),
    }
    constructor, weights, slot = constructors[args.model]
    # torchvision uses the cached file if present; otherwise it downloads it.
    model = constructor(weights=weights)
    model.classifier[slot] = torch.nn.Linear(model.classifier[slot].in_features, len(CLASSES))
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
    return model.to(device=device, memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, images, indices, batch, size, mean, std, tta=False, ten_crop=False):
    model.eval()
    outputs = []
    feature_average = isinstance(getattr(model, 'head', None), NormalizedFeatureHead)
    def forward(view):
        if feature_average:
            return model.forward_head(model.forward_features(view), pre_logits=True)
        result = model(view)
        return result[0] if isinstance(result, tuple) else result
    for chunk in indices.split(batch):
        x = preprocess(images[chunk], size, mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
            logits = forward(x).float()
            if tta:
                logits += forward(x.flip(-1)).float()
                count = 2
                if ten_crop:
                    shift = max(1, size // 40)
                    padded = F.pad(x, (shift, shift, shift, shift), mode="replicate")
                    for top, left in ((0, 0), (0, 2 * shift), (2 * shift, 0), (2 * shift, 2 * shift)):
                        view = padded[:, :, top:top + size, left:left + size].contiguous(memory_format=torch.channels_last)
                        logits += forward(view).float() + forward(view.flip(-1)).float()
                        count += 2
                logits /= count
            if feature_average:
                logits = model.head(logits).float()
        outputs.append(logits.float().softmax(dim=1).cpu())
    return torch.cat(outputs)


def select_tta(model, images, labels, val_idx, args, mean, std):
    scores = {}
    predictions = {}
    for views in (2, 10):
        p = predict(model, images, val_idx, args.batch, args.size, mean, std, tta=True, ten_crop=views == 10)
        scores[views] = (p.argmax(1) == labels[val_idx].cpu()).float().mean().item()
        predictions[views] = p.numpy()
    selected = max(scores, key=scores.get)
    print(f"Validation TTA comparison: flip={scores[2]:.4f}, 10 views={scores[10]:.4f}; selected={selected}", flush=True)
    np.savez(args.work_dir / "tta_validation.npz", probabilities=predictions[selected],
             targets=labels[val_idx].cpu().numpy(), accuracy=scores[selected], views=selected)
    return str(selected)


def train_one(seed, images, labels, test_images, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    all_idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=args.validation, stratify=labels.cpu().numpy(), random_state=args.split_seed
    )
    if args.full_data:
        train_idx = all_idx
    train_idx = torch.tensor(train_idx, device=device)
    val_idx = torch.tensor(val_idx, device=device)
    test_idx = torch.arange(len(test_images), device=device)
    model = make_model(device, args)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema else None
    ema_updates = 0
    head_module = model.head if args.model in ('convnextv2_tiny', 'fmae_affect') else (model.fc if args.model.startswith('ddam_') else model.classifier)
    head = list(head_module.parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * args.head_lr}],
        weight_decay=args.weight_decay, fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    mean, std = MEAN.to(device), STD.to(device)
    best_accuracy = -1.0
    best_state = None
    stale = 0

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        if args.model == 'fmae_affect' and args.train_blocks < 24:
            model.patch_embed.eval()
            frozen_blocks = model.blocks[:-args.train_blocks] if args.train_blocks else model.blocks
            for block in frozen_blocks:
                block.eval()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        total_loss = 0.0
        for batch_number, idx in enumerate(order.split(args.batch), 1):
            if len(idx) < 2:
                continue  # BatchNorm after the DDAM spatial pool requires at least two photos.
            x = preprocess(images[idx], args.size, mean, std, training=True, strength=args.augment_strength)
            permutation = None
            if args.mixup > 0 and random.random() < 0.5:
                ratio = float(np.random.beta(args.mixup, args.mixup))
                permutation = torch.randperm(len(idx), device=device)
                x = ratio * x + (1 - ratio) * x[permutation]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
                raw = model(x)
                logits = raw[0] if isinstance(raw, tuple) else raw
                loss = F.cross_entropy(logits, labels[idx], label_smoothing=args.smoothing)
                if permutation is not None:
                    loss = ratio * loss + (1 - ratio) * F.cross_entropy(
                        logits, labels[idx][permutation], label_smoothing=args.smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            if ema is not None:
                ema_updates += 1
                decay = min(args.ema, (1 + ema_updates) / (10 + ema_updates))
                with torch.no_grad():
                    ema_params = list(ema.parameters())
                    live_params = list(model.parameters())
                    torch._foreach_mul_(ema_params, decay)
                    torch._foreach_add_(ema_params, live_params, alpha=1 - decay)
                    for dst, src in zip(ema.buffers(), model.buffers()):
                        dst.copy_(src)
            total_loss += loss.item() * len(idx)
            if args.progress and batch_number % 100 == 0:
                print(f'epoch={epoch+1} batch={batch_number}/{math.ceil(len(order)/args.batch)}', flush=True)
        scheduler.step()
        if args.full_data:
            accuracy = None
            probs = None
        else:
            probs = predict(ema if ema is not None else model, images, val_idx, args.batch, args.size, mean, std, tta=True)
            accuracy = (probs.argmax(1) == labels[val_idx].cpu()).float().mean().item()
        validation_message = "full-data fit" if accuracy is None else f"val_acc={accuracy:.4f} best={max(best_accuracy,accuracy):.4f}"
        print(f"seed={seed} epoch={epoch+1:02d} loss={total_loss/len(train_idx):.4f} "
              f"{validation_message} "
              f"seconds={time.time()-start:.0f}", flush=True)
        improved = (epoch + 1 == args.epochs) if args.full_data else accuracy > best_accuracy
        if improved:
            if accuracy is not None:
                best_accuracy = accuracy
            selected_model = ema if ema is not None else model
            best_state = {k: v.detach().cpu().clone() for k, v in selected_model.state_dict().items()}
            torch.save({"model": best_state, "accuracy": accuracy, "epoch": epoch + 1,
                        "classes": CLASSES, "size": args.size, "architecture": args.model}, args.work_dir / f"best_seed{seed}.pth")
            if probs is not None:
                np.savez(args.work_dir / f"validation_seed{seed}.npz", probabilities=probs.numpy(),
                         targets=labels[val_idx].cpu().numpy(), indices=val_idx.cpu().numpy())
            stale = 0
        else:
            stale += 1
        if not args.full_data and stale >= args.patience and epoch + 1 >= args.min_epochs:
            break

    model.load_state_dict(best_state)
    del best_state
    torch.cuda.empty_cache()
    if not args.full_data and args.tta_views == "auto":
        args.tta_views = select_tta(model, images, labels, val_idx, args, mean, std)
    test_probs = predict(model, test_images, test_idx, args.batch, args.size, mean, std,
                         tta=True, ten_crop=args.tta_views == "10")
    np.savez(args.work_dir / f"predictions_seed{seed}.npz", probabilities=test_probs.numpy(),
             validation_accuracy=np.nan if args.full_data else best_accuracy,
             ids=np.array([p.name for p in sorted((args.data / "test").glob("*.jpg"))]))
    if not args.full_data:
        print(f"seed={seed} best validation accuracy={best_accuracy:.4f}", flush=True)
    del model, optimizer
    torch.cuda.empty_cache()
    return test_probs, None if args.full_data else best_accuracy


def apply_exact_matches(train_files, labels, test_files, predictions):
    # Reuse labels only for byte-identical images. Ignore contradictory labels.
    hashes = {}
    for path, label in zip(train_files, labels.tolist()):
        digest = hashlib.sha256(path.read_bytes()).digest()
        hashes.setdefault(digest, set()).add(label)
    matches = 0
    for i, path in enumerate(test_files):
        label_set = hashes.get(hashlib.sha256(path.read_bytes()).digest(), set())
        if len(label_set) == 1:
            predictions[i] = next(iter(label_set))
            matches += 1
    print(f"Exact train/test image matches: {matches}", flush=True)


def single_model_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("D:/2026task2/fer_data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("submission.csv"))
    parser.add_argument("--sample", type=Path, help="Submission template; defaults to data parent/submission.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model", choices=["efficientnet_b2", "convnextv2_tiny", "affectnet_b2_7", "affectnet_b2_8", 'ddam_raf', 'ddam_affect', 'fmae_affect'], default="convnextv2_tiny")
    parser.add_argument('--head-lr', type=float, default=5.0)
    parser.add_argument('--weight-decay', type=float, default=0.015)
    parser.add_argument('--smoothing', type=float, default=0.04)
    parser.add_argument('--ema', type=float, default=0.0)
    parser.add_argument('--train-blocks', type=int, default=4)
    parser.add_argument('--checkpoint-grad', action='store_true')
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--mlp-head', type=Path)
    parser.add_argument('--augment-strength', type=float, default=1.0)
    parser.add_argument("--weights", type=Path, help="Optional local timm ImageNet safetensors weights")
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--full-data", action="store_true", help="Fit every training image for exactly --epochs")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.00008)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--refit-epochs", type=int, default=6, help="Fine-tune the selected checkpoint on all images; 0 disables")
    parser.add_argument("--refit-lr", type=float, default=0.00001)
    parser.add_argument("--refit-seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--tta-views", choices=["auto", "2", "10"], default="auto")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()) / "fer_training")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training configuration")
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    device = torch.device("cuda")
    train_files, labels, images, test_files, test_images = load_images(args.data)
    images = images.to(device)
    labels = labels.to(device)
    test_images = test_images.to(device)
    if args.full_data and args.tta_views == "auto":
        if args.resume and torch.load(args.resume, map_location="cpu", weights_only=True).get("accuracy") is not None:
            _, heldout = train_test_split(np.arange(len(labels)), test_size=args.validation,
                                         stratify=labels.cpu().numpy(), random_state=args.split_seed)
            evaluation_model = make_model(device, args)
            args.tta_views = select_tta(evaluation_model, images, labels, torch.tensor(heldout, device=device),
                                        args, MEAN.to(device), STD.to(device))
            del evaluation_model
            torch.cuda.empty_cache()
        else:
            args.tta_views = "2"
    ensemble = None
    for seed in args.seeds:
        probs, score = train_one(seed, images, labels, test_images, args, device)
        if not args.full_data and args.refit_epochs > 0:
            full_args = argparse.Namespace(**vars(args))
            full_args.full_data = True
            full_args.resume = args.work_dir / f"best_seed{seed}.pth"
            full_args.epochs = args.refit_epochs
            full_args.lr = args.refit_lr
            full_args.work_dir = args.work_dir / f"full_from_seed{seed}"
            full_args.work_dir.mkdir(parents=True, exist_ok=True)
            for full_seed in args.refit_seeds:
                full_probs, _ = train_one(full_seed, images, labels, test_images, full_args, device)
                ensemble = full_probs if ensemble is None else ensemble + full_probs
        else:
            ensemble = probs if ensemble is None else ensemble + probs
    predictions = ensemble.argmax(1).numpy()
    apply_exact_matches(train_files, labels.cpu(), test_files, predictions)
    results = {path.name: SUBMISSION_IDS[CLASSES[int(label)]] for path, label in zip(test_files, predictions)}
    sample = args.sample or args.data.parent / "submission.csv"
    if sample.exists():
        with sample.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["ID", "Emotion"]:
                raise ValueError("Expected template columns ID,Emotion")
            order = [row["ID"] for row in reader]
        if len(order) != len(results) or len(set(order)) != len(order) or set(order) != set(results):
            raise ValueError("Template IDs do not exactly match the test images")
    else:
        order = [path.name for path in test_files]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ID", "Emotion"))
        writer.writerows((name, results[name]) for name in order)
    print(f"Wrote {len(predictions)} rows to {args.output}", flush=True)
    print("Submission class IDs:", SUBMISSION_IDS, flush=True)




"""Successful ensemble recipe; appended to the self-contained deliverable."""


def download_weight(url, destination, expected_sha):
    destination = Path(destination)
    if destination.exists():
        return destination
    temporary = destination.with_suffix('.partial')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(12):
        offset = temporary.stat().st_size if temporary.exists() else 0
        separator = '&' if '?' in url else '?'
        request = urllib.request.Request(url + separator + 'attempt=' + str(attempt),
                                         headers={'Range': f'bytes={offset}-'})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if offset and response.status != 206:
                    temporary.unlink()
                    continue
                total = offset + int(response.headers.get('Content-Length', 0))
                with temporary.open('ab') as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if temporary.stat().st_size == total:
                digest = hashlib.sha256()
                with temporary.open('rb') as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha:
                    raise ValueError(f'Pretrained weight SHA256 mismatch: {destination}')
                temporary.replace(destination)
                return destination
        except (TimeoutError, OSError) as exc:
            print('Retrying pretrained download:', type(exc).__name__, flush=True)
    raise RuntimeError(f'Could not finish pretrained download: {url}')


def ensure_pretrained(cache, model):
    if model == 'convnextv2_tiny':
        return None  # timm downloads and caches the official ImageNet checkpoint.
    folder = cache / 'pretrained'
    folder.mkdir(parents=True, exist_ok=True)
    if model == 'affectnet_b2_7':
        converted = folder / 'enet_b2_7.safetensors'
        if converted.exists():
            return converted
        raw = download_weight(
            'https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/enet_b2_7.pt',
            folder / 'enet_b2_7.pt', 'e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f')
        allowed = []
        for dotted in torch.serialization.get_unsafe_globals_in_checkpoint(raw):
            if not dotted.startswith(('torch.nn.', 'timm.')):
                raise ValueError(f'Unexpected serialized type: {dotted}')
            module, attribute = dotted.rsplit('.', 1)
            allowed.append(getattr(importlib.import_module(module), attribute))
        with torch.serialization.safe_globals(allowed):
            network = torch.load(raw, map_location='cpu', weights_only=True)
        from safetensors.torch import save_file
        save_file({key: value.contiguous() for key, value in network.state_dict().items()}, str(converted))
        return converted
    if model == 'ddam_raf':
        return download_weight(
            'https://raw.githubusercontent.com/SainingZhang/DDAMFN/main/DDAMFN%2B%2B/checkpoints_ver2.0/rafdb_epoch20_acc0.9204_bacc0.8617.pth',
            folder / 'ddam_raf.pth', '61f2e763c4f451728b4d08bd05d99ef30ff8b14c253c32bf04e5e3da2f6038f7')
    if model == 'fmae_affect':
        converted = folder / 'fmae_affectnet8.safetensors'
        if converted.exists():
            return converted
        raw = download_weight(
            'https://huggingface.co/forever208/FMAE-IAT/resolve/main/FMAE_AffectNet8.pth?download=true',
            folder / 'FMAE_AffectNet8.pth', '51ba91d213224ae4d3d72b242a91ffbef034c1109c0036b0e831b4aa26dd8f96')
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(raw, map_location='cpu', weights_only=True, mmap=True)
        from safetensors.torch import save_file
        save_file({key: value.contiguous() for key, value in checkpoint['model'].items()}, str(converted))
        return converted
    raise ValueError(model)


def recipe_args(options, model, **overrides):
    configuration = dict(data=options.data, model=model, weights=ensure_pretrained(options.work_dir, model),
                         resume=None, epochs=30, min_epochs=20, patience=8, batch=64, size=224,
                         lr=.00008, head_lr=2, weight_decay=.0005, smoothing=.02, mixup=.1,
                         full_data=False, validation=.15, split_seed=42, ema=0., train_blocks=4,
                         checkpoint_grad=False, progress=False, mlp_head=None, augment_strength=1.,
                         tta_views='auto', work_dir=options.work_dir / model / 'validation')
    configuration.update(overrides)
    args = argparse.Namespace(**configuration)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    return args


def fit_cnn_recipe(options, model, images, labels, test_images, device):
    defaults = {}
    if model == 'convnextv2_tiny':
        defaults = dict(size=160, head_lr=5, weight_decay=.015, smoothing=.04, mixup=.2)
    if model == 'ddam_raf':
        defaults = dict(size=112, batch=128, epochs=25, min_epochs=15, patience=7, lr=.00006, ema=.995)
    prediction_cache = options.work_dir / (model + '_full_predictions.npz')
    validation_cache = options.work_dir / (model + '_validation.npz')
    if prediction_cache.exists() and validation_cache.exists():
        return np.load(prediction_cache)['probabilities'], np.load(validation_cache)['probabilities']
    args = recipe_args(options, model, **defaults)
    if model == 'ddam_raf' and not (args.work_dir / 'best_seed42.pth').exists():
        warm = argparse.Namespace(**vars(args))
        warm.epochs, warm.batch, warm.lr, warm.tta_views = 1, 32, .00005, '2'
        warm.work_dir = options.work_dir / model / 'warmup'
        warm.work_dir.mkdir(parents=True, exist_ok=True)
        train_one(42, images, labels, test_images, warm, device)
        args.resume = warm.work_dir / 'best_seed42.pth'
    if not (args.work_dir / 'tta_validation.npz').exists():
        train_one(42, images, labels, test_images, args, device)
    validation = np.load(args.work_dir / 'tta_validation.npz')
    np.savez(validation_cache, probabilities=validation['probabilities'], targets=validation['targets'])
    full = argparse.Namespace(**vars(args))
    full.full_data = True
    full.resume = args.work_dir / 'best_seed42.pth'
    full.epochs = 4 if model == 'ddam_raf' else 6
    full.lr = .00001
    full.tta_views = str(int(validation['views']))
    full.work_dir = options.work_dir / model / 'full'
    full.work_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for seed in ((42, 2026) if model == 'convnextv2_tiny' else (42,)):
        p, _ = train_one(seed, images, labels, test_images, full, device)
        predictions.append(p.numpy())
    prediction = np.mean(predictions, axis=0)
    np.savez(prediction_cache, probabilities=prediction)
    return prediction, validation['probabilities']


def extract_face_features(options, images, test_images):
    path = options.work_dir / 'fmae_features.npz'
    if path.exists():
        data = np.load(path)
        return data['train'], data['test']
    args = recipe_args(options, 'fmae_affect', train_blocks=0)
    model = make_model(images.device, args).eval()
    model.reset_classifier(0, global_pool='avg')
    results = []
    with torch.inference_mode():
        for pool in (images, test_images):
            output = []
            for index, chunk in enumerate(torch.arange(len(pool), device=pool.device).split(16)):
                x = preprocess(pool[chunk], 224, MEAN.to(pool.device), STD.to(pool.device))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    feature = (model(x).float() + model(x.flip(-1)).float()) / 2
                output.append(feature.cpu().numpy())
                if index % 128 == 0:
                    print('Face feature extraction:', index * 16, '/', len(pool), flush=True)
            results.append(np.concatenate(output))
    np.savez(path, train=results[0], test=results[1])
    del model
    torch.cuda.empty_cache()
    return results


def mlp_classifier():
    return torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(1024, 256), torch.nn.GELU(),
                               torch.nn.Dropout(.5), torch.nn.Linear(256, 6))


def fit_feature_recipe(options, features, test_features, labels, train_indices, val_indices):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    cache = options.work_dir
    names = ('fmae_linear', 'fmae_mlp')
    if all((cache / (name + '_full_predictions.npz')).exists() and
           (cache / (name + '_validation.npz')).exists() for name in names):
        return [(np.load(cache / (name + '_full_predictions.npz'))['probabilities'],
                 np.load(cache / (name + '_validation.npz'))['probabilities']) for name in names]
    y = labels.cpu().numpy()
    scaler = StandardScaler().fit(features[train_indices])
    linear = LogisticRegression(C=.01, max_iter=800)
    linear.fit(scaler.transform(features[train_indices]), y[train_indices])
    linear_validation = linear.predict_proba(scaler.transform(features[val_indices])).astype('float32')
    scaler = StandardScaler().fit(features)
    linear.fit(scaler.transform(features), y)
    linear_prediction = linear.predict_proba(scaler.transform(test_features)).astype('float32')
    np.savez(cache / 'fmae_linear_model.npz', mean=scaler.mean_, scale=scaler.scale_,
             coefficients=linear.coef_, intercept=linear.intercept_)
    device = labels.device
    x = torch.tensor(features, device=device)
    tr, va = torch.tensor(train_indices, device=device), torch.tensor(val_indices, device=device)
    mean, std = x[tr].mean(0), x[tr].std(0).clamp_min(.01)
    x = (x - mean) / std
    tx = (torch.tensor(test_features, device=device) - mean) / std
    torch.manual_seed(42)
    model = mlp_classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=.00001)
    best, stale = -1., 0
    best_state, mlp_validation = None, None
    for epoch in range(100):
        model.train()
        for idx in tr[torch.randperm(len(tr), device=device)].split(512):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x[idx]), labels[idx], label_smoothing=.03)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(x[va]).float().softmax(1)
            accuracy = (p.argmax(1) == labels[va]).float().mean().item()
        print(f'FMAE head epoch={epoch+1} val_acc={accuracy:.4f}', flush=True)
        if accuracy > best:
            best, stale = accuracy, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mlp_validation = p.cpu().numpy()
        else:
            stale += 1
        if epoch > 35 and stale >= 20:
            break
    predictions = []
    torch.save(dict(model=best_state, mean=mean, std=std, accuracy=best),
               cache / 'fmae_mlp_validation.pth')
    for seed in (42, 2026):
        torch.manual_seed(seed)
        model = mlp_classifier().to(device)
        model.load_state_dict(best_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.000075, weight_decay=.01, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=.000005)
        for epoch in range(20):
            model.train()
            for idx in torch.randperm(len(x), device=device).split(512):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x[idx]), labels[idx], label_smoothing=.03)
                loss.backward()
                optimizer.step()
            scheduler.step()
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = torch.cat([model(chunk).float().softmax(1).cpu() for chunk in tx.split(512)]).numpy()
        predictions.append(prediction)
        torch.save(dict(model=model.state_dict(), mean=mean, std=std, full_data=True),
                   cache / f'fmae_mlp_full_seed{seed}.pth')
    mlp_prediction = np.mean(predictions, axis=0)
    results = [(linear_prediction, linear_validation), (mlp_prediction, mlp_validation)]
    for name, (prediction, validation) in zip(names, results):
        np.savez(cache / (name + '_full_predictions.npz'), probabilities=prediction)
        np.savez(cache / (name + '_validation.npz'), probabilities=validation, targets=y[val_indices])
    del model, optimizer, x, tx
    torch.cuda.empty_cache()
    return results


def fit_backbone_recipe(options, images, labels, test_images, device):
    model = 'fmae_finetune'
    cache = options.work_dir
    prediction_cache = cache / (model + '_full_predictions.npz')
    validation_cache = cache / (model + '_validation.npz')
    if prediction_cache.exists() and validation_cache.exists():
        return np.load(prediction_cache)['probabilities'], np.load(validation_cache)['probabilities']
    args = recipe_args(options, 'fmae_affect', mlp_head=cache / 'fmae_mlp_validation.pth',
                       epochs=8, min_epochs=5, patience=3, size=224, batch=16, lr=.0000075,
                       head_lr=2, weight_decay=.005, smoothing=.03, mixup=.1, augment_strength=.5,
                       work_dir=cache / model / 'validation')
    if not (args.work_dir / 'tta_validation.npz').exists():
        train_one(42, images, labels, test_images, args, device)
    validation = np.load(args.work_dir / 'tta_validation.npz')
    np.savez(validation_cache, probabilities=validation['probabilities'], targets=validation['targets'])
    full = argparse.Namespace(**vars(args))
    full.full_data, full.resume = True, args.work_dir / 'best_seed42.pth'
    full.epochs, full.lr = 2, .0000015
    full.tta_views = str(int(validation['views']))
    full.work_dir = cache / model / 'full'
    full.work_dir.mkdir(parents=True, exist_ok=True)
    prediction, _ = train_one(42, images, labels, test_images, full, device)
    np.savez(prediction_cache, probabilities=prediction.numpy())
    return prediction.numpy(), validation['probabilities']


def write_submission(options, train_files, labels, test_files, probabilities):
    if probabilities.shape != (len(test_files), 6) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid prediction matrix')
    predictions = probabilities.argmax(1)
    apply_exact_matches(train_files, labels.cpu(), test_files, predictions)
    results = {p.name: SUBMISSION_IDS[CLASSES[int(label)]] for p, label in zip(test_files, predictions)}
    sample = options.sample or options.data.parent / 'submission.csv'
    with sample.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['ID', 'Emotion']:
            raise ValueError('Expected submission columns ID,Emotion')
        order = [row['ID'] for row in reader]
    if len(set(order)) != len(order) or len(order) != len(results) or set(order) != set(results):
        raise ValueError('Submission template IDs differ from the test images')
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'Emotion'])
        writer.writerows((name, results[name]) for name in order)
    print(f'Wrote {len(order)} predictions: {options.output}', flush=True)


def ensemble_main():
    parser = argparse.ArgumentParser(description='2026task2: validated six-model ensemble')
    parser.add_argument('--data', type=Path, default=Path('D:/2026task2/fer_data'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('submission.csv'))
    parser.add_argument('--sample', type=Path)
    parser.add_argument('--work-dir', type=Path,
                        default=Path.home() / 'Documents' / 'Codex' / 'fer2026task2_cache')
    parser.add_argument('--reuse-predictions', action='store_true', help='Use saved inference caches for these same images')
    options = parser.parse_args()
    options.work_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    files, labels, images, tests, test_images = load_images(options.data)
    # A cache is valid only for this exact data and label ordering.
    fingerprint = hashlib.sha256()
    for path in files + tests:
        fingerprint.update(path.name.encode())
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    fingerprint.update(labels.numpy().tobytes())
    digest = fingerprint.hexdigest()
    marker = options.work_dir / 'data_sha256.txt'
    if options.reuse_predictions and not marker.exists():
        raise ValueError('Missing data fingerprint for prediction caches')
    if marker.exists() and marker.read_text() != digest:
        raise ValueError('Data changed; choose a fresh --work-dir')
    marker.write_text(digest)
    keys = ('convnextv2_tiny', 'affectnet_b2_7', 'ddam_raf', 'fmae_linear', 'fmae_mlp', 'fmae_finetune')
    if options.reuse_predictions:
        prediction = np.mean([np.load(options.work_dir / (key + '_full_predictions.npz'))['probabilities']
                              for key in keys], axis=0)
        write_submission(options, files, labels, tests, prediction)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is needed for training')
    device = torch.device('cuda')
    images, test_images, labels = images.to(device), test_images.to(device), labels.to(device)
    train, val = train_test_split(np.arange(len(labels)), test_size=.15,
                                 stratify=labels.cpu().numpy(), random_state=42)
    results = [fit_cnn_recipe(options, key, images, labels, test_images, device) for key in keys[:3]]
    head_caches = [options.work_dir / (key + suffix) for key in keys[3:5]
                   for suffix in ('_full_predictions.npz', '_validation.npz')]
    if all(path.exists() for path in head_caches):
        features, test_features = None, None
    else:
        features, test_features = extract_face_features(options, images, test_images)
    results.extend(fit_feature_recipe(options, features, test_features, labels, train, val))
    results.append(fit_backbone_recipe(options, images, labels, test_images, device))
    validation = np.mean([result[1] for result in results], axis=0)
    score = float((validation.argmax(1) == labels[val].cpu().numpy()).mean())
    print(f'Ensemble heldout accuracy: {score:.4%}; full-data models have no holdout score', flush=True)
    np.savez(options.work_dir / 'ensemble_validation.npz', probabilities=validation,
             targets=labels[val].cpu().numpy(), indices=val, accuracy=score)
    prediction = np.mean([result[0] for result in results], axis=0)
    write_submission(options, files, labels, tests, prediction)


if __name__ == '__main__':
    import sys
    if '--single-model' in sys.argv:
        sys.argv.remove('--single-model')
        single_model_main()
    else:
        ensemble_main()
