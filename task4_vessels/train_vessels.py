"""Train/refine retinal vessel U-Nets and predict. Run main.py for all stages.

CUDA training is not bitwise deterministic. Retain final weights and the selected
validation threshold if the competition requires exact prediction reproduction.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F

VIEWS = 8


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(co), nn.SiLU(inplace=True),
                                  nn.Conv2d(co, co, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(co), nn.SiLU(inplace=True))
    def forward(self, x):
        return self.body(x)


class UNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.enc = nn.ModuleList([Block(4, width), Block(width, width*2),
                                 Block(width*2, width*4), Block(width*4, width*8),
                                 Block(width*8, width*16)])
        self.dec = nn.ModuleList([Block(width*24, width*8), Block(width*12, width*4),
                                 Block(width*6, width*2), Block(width*3, width)])
        self.head = nn.Conv2d(width, 1, 1)
    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


def preprocess(path, resolution=0):
    image = Image.open(path).convert('RGB')
    if resolution:
        image = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.
    fov = rgb.max(axis=2) > .08
    green = rgb[:, :, 1]
    smooth = gaussian_filter(green*fov, 10) / np.maximum(gaussian_filter(fov.astype(np.float32), 10), .001)
    local = np.clip((green-smooth)*8, -1.5, 1.5) * fov
    return np.concatenate(((rgb-.5)/.25, local[:, :, None]), axis=2).transpose(2, 0, 1), fov


def read_data(root, split, labels=False, resolution=0):
    files = sorted((root/split/'image').glob('*.jpg'), key=lambda p: int(p.stem))
    data, masks, fovs = [], [], []
    for path in files:
        x, fov = preprocess(path, resolution)
        data.append(x)
        fovs.append(fov)
        if labels:
            label = Image.open(root/split/'label'/path.name).convert('L')
            if resolution:
                label = label.resize((resolution, resolution), Image.Resampling.NEAREST)
            masks.append(np.asarray(label) > 127)
    return files, np.stack(data), np.stack(masks) if labels else None, np.stack(fovs)


@torch.inference_mode()
def predict(model, x, tta=True):
    model.eval()
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (-w)%16, 0, (-h)%16), mode='reflect')
    pred = 0
    flips = [(), (2,), (3,), (2, 3)] if tta else [()]
    for transpose in ([False, True] if tta and VIEWS == 8 else [False]):
        source = x.transpose(2, 3) if transpose else x
        for dims in flips:
            a = source.flip(dims) if dims else source
            with torch.autocast('cuda', dtype=torch.float16):
                a = model(a).sigmoid().float()
            a = a.flip(dims) if dims else a
            pred = pred + (a.transpose(2, 3) if transpose else a)
    return (pred/(len(flips)*(2 if tta and VIEWS == 8 else 1)))[..., :h, :w]


def score(probs, labels):
    thresholds = np.arange(.20, .701, .025)
    records = []
    for threshold in thresholds:
        p = probs > threshold
        inter = (p & labels).sum(axis=(1, 2))
        dice = 2*inter/(p.sum(axis=(1, 2))+labels.sum(axis=(1, 2)))
        records.append((float(dice.mean()), float(threshold), dice.tolist()))
    return max(records)


def train(args):
    global VIEWS
    VIEWS = getattr(args, 'views', 8)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files, images, labels, fovs = read_data(root, 'train', True, getattr(args, 'resolution', 0))
    val_ids = [3, 8, 13, 18]
    train_indices = [i for i, p in enumerate(files) if args.full or int(p.stem) not in val_ids]
    val_indices = [i for i, p in enumerate(files) if int(p.stem) in val_ids]
    xall = torch.tensor(images[train_indices], device='cuda')
    yall = torch.tensor(labels[train_indices, None].astype(np.float32), device='cuda')
    positives = [np.argwhere(labels[i]) for i in train_indices]
    fields = [np.argwhere(fovs[i]) for i in train_indices]
    xv = torch.tensor(images[val_indices], device='cuda')
    yv = labels[val_indices]
    model = UNet().cuda()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location='cuda', weights_only=False)['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    h, w = images.shape[-2:]
    best, history = -1., []
    start = time.time()
    batch = args.batch
    size = args.patch
    for step in range(1, args.steps+1):
        model.train()
        indices = np.random.randint(len(train_indices), size=batch)
        centers = []
        for j in indices:
            candidates = positives[j] if random.random()<.5 else fields[j]
            centers.append(candidates[random.randrange(len(candidates))])
        centers = np.array(centers)
        angles = np.random.uniform(-math.pi, math.pi, batch)
        scales = np.random.uniform(.7, 1.35, batch)
        flip = np.random.choice([-1., 1.], batch)
        theta = np.zeros((batch, 2, 3), dtype=np.float32)
        theta[:, 0, 0] = np.cos(angles)*size/w*scales*flip
        theta[:, 0, 1] = -np.sin(angles)*size/w*scales
        theta[:, 1, 0] = np.sin(angles)*size/h*scales*flip
        theta[:, 1, 1] = np.cos(angles)*size/h*scales
        theta[:, 0, 2] = (centers[:, 1]+.5)*2/w-1
        theta[:, 1, 2] = (centers[:, 0]+.5)*2/h-1
        grid = F.affine_grid(torch.tensor(theta, device='cuda'), (batch, 4, size, size), align_corners=False)
        idx = torch.tensor(indices, device='cuda')
        x = F.grid_sample(xall[idx], grid, align_corners=False, padding_mode='reflection')
        y = F.grid_sample(yall[idx], grid, align_corners=False, mode='nearest')
        gain = torch.empty((batch, 4, 1, 1), device='cuda').uniform_(.85, 1.15)
        shift = torch.empty((batch, 4, 1, 1), device='cuda').uniform_(-.10, .10)
        shift[:, 3] = 0
        x = x*gain + shift + torch.randn_like(x)*.025
        lr = args.lr*(.05+.95*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            p = logits.sigmoid().float()
            dice_loss = 1-(2*(p*y).sum((1, 2, 3))+1)/(p.sum((1, 2, 3))+y.sum((1, 2, 3))+1)
            loss = bce + .5*dice_loss.mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()
        if step%50 == 0 or step==1:
            print(json.dumps({'step':step, 'loss':round(loss.item(),5), 'seconds':round(time.time()-start,1)}), flush=True)
        if (step%250==0 or step==args.steps) and not args.full:
            probs = torch.cat([predict(model, xv[i:i+1]) for i in range(len(xv))]).cpu().numpy()[:, 0]
            result = score(probs, yv)
            record = {'step':step, 'dice':result[0], 'threshold':result[1], 'per_image':result[2]}
            history.append(record)
            print('VALIDATION '+json.dumps(record), flush=True)
            if result[0] > best:
                best = result[0]
                torch.save({'model':model.state_dict(), 'step':step, 'seed':seed, 'validation':record, 'full':False}, out/'best.pt')
                np.savez_compressed(out/'validation.npz', probabilities=probs, labels=yv, ids=np.array(val_ids))
            (out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if args.full:
        torch.save({'model':model.state_dict(), 'step':args.steps, 'seed':seed, 'full':True}, out/'best.pt')
    print('FINISHED '+str(out/'best.pt'), flush=True)


def infer(args):
    global VIEWS
    VIEWS = getattr(args, 'views', 8)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files, images, _, fovs = read_data(Path(args.data), 'test', resolution=getattr(args, 'resolution', 0))
    ensemble = []
    for ckpt in args.checkpoints:
        model = UNet().cuda()
        state = torch.load(ckpt, map_location='cuda', weights_only=False)
        model.load_state_dict(state['model'])
        predictions = []
        for i, path in enumerate(files):
            x = torch.tensor(images[i:i+1], device='cuda')
            prob = predict(model, x).cpu().numpy()[0, 0] * fovs[i]
            predictions.append(prob)
            print(f'{Path(ckpt).parent.name}: {path.name}', flush=True)
        ensemble.append(np.stack(predictions))
        del model
    np.savez_compressed(out/'test_probabilities.npz', probabilities=np.mean(ensemble, axis=0), ids=np.array([p.stem for p in files]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'infer'])
    parser.add_argument('--data', default='D:/2026task-4/segmentation')
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patch', type=int, default=144)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--checkpoints', nargs='+')
    parser.add_argument('--init', default=None)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--resolution', type=int, default=0)
    parser.add_argument('--views', type=int, choices=[4, 8], default=8)
    args = parser.parse_args()
    (train if args.mode=='train' else infer)(args)
