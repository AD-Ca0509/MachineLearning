"""黄斑区域定位 + 中心凹热力图回归。训练、交叉验证、集成推理及 CSV 校验。

python train.py --mode train --data D:/2026task3/detection
python train.py --mode predict --data D:/2026task3/detection
不上传图片，不使用测试标签。原图坐标原点在左上，X 向右，Y 向下。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.model_selection import KFold
from torchvision.models import resnet50, ResNet50_Weights


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


class FoveaNet(nn.Module):
    """ResNet50 + FPN，共享主干，输出中心凹热图、黄斑区域及可见性。"""
    def __init__(self, pretrained=False):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2 = base.layer1, base.layer2
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in [256, 512, 1024, 2048]])
        self.smooth = nn.ModuleList([ConvBlock(64, 64) for _ in range(3)])
        self.head = nn.Sequential(ConvBlock(64, 64), nn.Conv2d(64, 2, 1))
        self.presence = nn.Sequential(nn.Linear(2048, 128), nn.SiLU(), nn.Linear(128, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p = self.lateral[3](c4)
        for k in [2, 1, 0]:
            p = self.smooth[k](F.interpolate(p, size=[c1, c2, c3][k].shape[-2:],
                                             mode='bilinear', align_corners=False)
                               + self.lateral[k]([c1, c2, c3][k]))
        maps = self.head(p)
        return maps[:, 0], maps[:, 1], self.presence(c4.mean((2, 3))).squeeze(1)

    def freeze_batchnorm(self):
        # 小批量训练保持 ImageNet 的 BN 均值/方差，仿射参数仍可训练。
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def load_rgb(path, side):
    with Image.open(path) as original:
        im = original.convert('RGB')
        width, height = im.size
        small = np.asarray(im.resize((256, 256)))
        mask = (small[:, :, 0] > 20) & (small[:, :, 1] > 8)
        xs = np.where(mask.mean(0) > .06)[0]
        ys = np.where(mask.mean(1) > .06)[0]
        if len(xs) > 20 and len(ys) > 20:
            x0 = max(0, int((xs[0] / 256 - .015) * width))
            y0 = max(0, int((ys[0] / 256 - .015) * height))
            x1 = min(width, int(((xs[-1] + 1) / 256 + .015) * width))
            y1 = min(height, int(((ys[-1] + 1) / 256 + .015) * height))
        else:
            x0, y0, x1, y1 = 0, 0, width, height
        crop = (x0, y0, x1, y1)
        arr = np.asarray(im.crop(crop).resize((side, side), Image.Resampling.BILINEAR)).copy()
    return arr, crop, (width, height)


def read_data(root, side):
    root = Path(root)
    labels = list(csv.DictReader((root / 'fovea_localization_train_GT.csv').open(encoding='utf-8-sig')))
    gt = {int(float(r['data'])): np.array([float(r['Fovea_X']), float(r['Fovea_Y'])]) for r in labels}
    if len(gt) != len(labels):
        raise ValueError('训练标注中存在重复图片 ID')
    data = []
    for split in ['train', 'test']:
        for path in sorted((root / split).glob('*.jpg')):
            image_id = int(path.stem)
            arr, crop, size = load_rgb(path, side)
            offset, extent = np.array(crop[:2]), np.array(crop[2:]) - crop[:2]
            original_xy = gt.get(image_id) if split == 'train' else None
            visible = original_xy is not None and not np.all(original_xy == 0)
            point = (original_xy - offset) / extent if visible else np.array([.5, .5])
            box = np.array([*point - .05, *point + .05])
            xp = root / 'train_location' / (path.stem + '.xml')
            if split == 'train' and xp.exists():
                tree = ET.parse(xp)
                xml_size = np.array([float(tree.findtext('size/width')), float(tree.findtext('size/height'))])
                # XML 的图片名可能是 png，按数字 stem 配对，检查坐标尺度。
                candidates = tree.findall('.//object/bndbox')
                if visible and candidates:
                    raw = np.array([[float(b.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                                    for b in candidates])
                    raw = raw / np.tile(xml_size, 2) * np.tile(size, 2)
                    choice = np.argmin(np.linalg.norm((raw[:, :2] + raw[:, 2:]) / 2 - original_xy, axis=1))
                    box = (raw[choice] - np.tile(offset, 2)) / np.tile(extent, 2)
            if visible and not ((point >= 0).all() and (point <= 1).all()):
                raise ValueError(f'{path.name}: 中心凹不在预处理后的视野内')
            data.append(dict(id=image_id, path=str(path), split=split, image=arr, crop=crop,
                             size=size, point=point.tolist(), box=box.tolist(), visible=visible,
                             gt=original_xy.tolist() if original_xy is not None else None))
    train = [r for r in data if r['split'] == 'train']
    test = [r for r in data if r['split'] == 'test']
    if set(gt) != {r['id'] for r in train}:
        raise ValueError('图片与训练坐标标注未一一匹配')
    if not test:
        raise ValueError('测试图片为空')
    return train, test


def tensors(records, device):
    images = torch.from_numpy(np.stack([r['image'] for r in records])).permute(0, 3, 1, 2)
    images = images.to(device=device, dtype=torch.float32) / 255
    points = torch.tensor([r['point'] for r in records], device=device, dtype=torch.float32)
    boxes = torch.tensor([r['box'] for r in records], device=device, dtype=torch.float32)
    visible = torch.tensor([r['visible'] for r in records], device=device, dtype=torch.float32)
    return images, points, boxes, visible


def augment(images, points, boxes, visible, negative_rate=.12):
    """批量仿射变换；坐标与 XML 框共用逆矩阵，避免增强后的标签错位。"""
    n, _, h, w = images.shape
    device = images.device
    angle = (torch.rand(n, device=device) * 2 - 1) * math.radians(18)
    scale = .85 + torch.rand(n, device=device) * .3
    flip_x = torch.where(torch.rand(n, device=device) < .5, -1., 1.)
    flip_y = torch.where(torch.rand(n, device=device) < .15, -1., 1.)
    matrix = torch.zeros(n, 2, 3, device=device)
    matrix[:, 0, 0] = angle.cos() / scale * flip_x
    matrix[:, 0, 1] = -angle.sin() / scale * flip_y
    matrix[:, 1, 0] = angle.sin() / scale * flip_x
    matrix[:, 1, 1] = angle.cos() / scale * flip_y
    matrix[:, :, 2] = (torch.rand(n, 2, device=device) * 2 - 1) * .10
    # 合成负例：从不含中心凹的图像一侧裁剪，绝不更改原始标签文件。
    synthetic = torch.rand(n, device=device) < negative_rate
    for i in torch.where(synthetic)[0].tolist():
        px, py = points[i].tolist()
        options = []
        if px > .32: options.append((0, 0, px - .14, 1))
        if px < .68: options.append((px + .14, 0, 1, 1))
        if py > .32: options.append((0, 0, 1, py - .14))
        if py < .68: options.append((0, py + .14, 1, 1))
        if not options:
            synthetic[i] = False
            continue
        x0, y0, x1, y1 = random.choice(options)
        matrix[i] = torch.tensor([[x1-x0, 0, x0+x1-1], [0, y1-y0, y0+y1-1]], device=device)
    inv = torch.linalg.inv(matrix[:, :, :2])
    transformed_points = torch.bmm(inv, (points * 2 - 1 - matrix[:, :, 2]).unsqueeze(-1)).squeeze(-1)
    transformed_points = (transformed_points + 1) / 2
    corners = torch.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], 1)
    corners = torch.bmm(corners * 2 - 1 - matrix[:, None, :, 2], inv.transpose(1, 2)) / 2 + .5
    transformed_boxes = torch.cat([corners.min(1).values, corners.max(1).values], 1).clamp(0, 1)
    valid = visible * (~synthetic).float() * ((transformed_points > 0) & (transformed_points < 1)).all(1)
    grid = F.affine_grid(matrix, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, align_corners=False, padding_mode='zeros')
    gain = .75 + torch.rand(n, 1, 1, 1, device=device) * .5
    gamma = .8 + torch.rand(n, 1, 1, 1, device=device) * .4
    colors = .9 + torch.rand(n, 3, 1, 1, device=device) * .2
    images = (images.clamp(0, 1).pow(gamma) * gain * colors).clamp(0, 1)
    return images, transformed_points, transformed_boxes, valid


def make_grid(h, w, device):
    yy, xx = torch.meshgrid((torch.arange(h, device=device) + .5) / h,
                            (torch.arange(w, device=device) + .5) / w, indexing='ij')
    return xx, yy


def loss_fn(outputs, points, boxes, visible):
    heat, region, presence = [o.float() for o in outputs]
    h, w = heat.shape[-2:]
    xx, yy = make_grid(h, w, heat.device)
    d2 = ((xx - points[:, 0, None, None]) * w) ** 2 + ((yy - points[:, 1, None, None]) * h) ** 2
    target = torch.exp(-d2 / (2 * 1.6 ** 2))
    target /= target.sum((1, 2), keepdim=True).clamp_min(1e-8)
    ce = -(target * heat.flatten(1).log_softmax(1).view_as(heat)).sum((1, 2))
    heat_loss = (ce * visible).sum() / visible.sum().clamp_min(1)
    box_target = ((xx >= boxes[:, 0, None, None]) & (xx <= boxes[:, 2, None, None]) &
                  (yy >= boxes[:, 1, None, None]) & (yy <= boxes[:, 3, None, None])).float() * visible[:, None, None]
    weights = 1 + 8 * box_target
    region_loss = (F.binary_cross_entropy_with_logits(region, box_target, reduction='none') * weights).mean()
    region_prob = region.sigmoid()
    dice = 1 - (2 * (region_prob * box_target).sum((1, 2)) + 1) / (region_prob.sum((1, 2)) + box_target.sum((1, 2)) + 1)
    presence_loss = F.binary_cross_entropy_with_logits(presence, visible)
    prob = heat.flatten(1).softmax(1).view_as(heat)
    xy = torch.stack([(prob * xx).sum((1, 2)), (prob * yy).sum((1, 2))], 1)
    coord_loss = (((xy - points) ** 2).sum(1) * visible).sum() / visible.sum().clamp_min(1)
    return heat_loss + region_loss + .25 * dice.mean() + .5 * presence_loss + 10 * coord_loss


@torch.no_grad()
def predict_records(model, records, device, batch=8, tta=True):
    """先用黄斑分割限定候选区域，再局部 soft-argmax 取中心凹坐标。"""
    model.eval()
    xy_all, vis_all, boxes_all = [], [], []
    for start in range(0, len(records), batch):
        subset = records[start:start+batch]
        images = tensors(subset, device)[0]
        maps, regions, visible = [], [], []
        for flip in ([False, True] if tta else [False]):
            inp = images.flip(-1) if flip else images
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                heat, region, presence = model(inp)
            if flip:
                heat, region = heat.flip(-1), region.flip(-1)
            maps.append(heat.float().flatten(1).softmax(1).view_as(heat))
            regions.append(region.float().sigmoid())
            visible.append(presence.float().sigmoid())
        prob = torch.stack(maps).mean(0)
        region = torch.stack(regions).mean(0)
        visibility = torch.stack(visible).mean(0)
        h, w = prob.shape[-2:]
        xx, yy = make_grid(h, w, device)
        # 区域概率作为软门控，防止检测略偏时彻底剪掉正确位置。
        gated = prob * (.20 + .80 * region)
        peak = gated.flatten(1).argmax(1)
        cx, cy = (peak % w + .5) / w, (peak // w + .5) / h
        local = ((xx - cx[:, None, None]).abs() <= 6 / w) & ((yy - cy[:, None, None]).abs() <= 6 / h)
        weights = gated * local
        weights /= weights.sum((1, 2), keepdim=True).clamp_min(1e-12)
        xy = torch.stack([(weights * xx).sum((1, 2)), (weights * yy).sum((1, 2))], 1).cpu().numpy()
        region_np = region.cpu().numpy()
        for j, r in enumerate(subset):
            offset = np.array(r['crop'][:2])
            extent = np.array(r['crop'][2:]) - offset
            point = xy[j] * extent + offset
            xy_all.append(np.clip(point, [0, 0], np.array(r['size']) - 1))
            ys, xs = np.where((region_np[j] > .4) & ((xx.cpu().numpy()-xy[j,0])**2 +
                                                   (yy.cpu().numpy()-xy[j,1])**2 < .15**2))
            if len(xs):
                box = np.array([xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h])
            else:
                box = np.array([*xy[j]-.05, *xy[j]+.05]).clip(0,1)
            boxes_all.append(box * np.tile(extent, 2) + np.tile(offset, 2))
        vis_all.extend(visibility.cpu().numpy())
    return np.array(xy_all), np.array(vis_all), np.array(boxes_all)


def score(records, xy, visibility, threshold):
    pred = xy.copy()
    pred[visibility < threshold] = 0
    gt = np.array([r['gt'] for r in records])
    return float(np.sqrt(np.mean((pred-gt)**2)))


def manifest(root, records, side):
    return dict(data_root=str(Path(root).resolve()), side=side,
                images=[dict(id=r['id'], split=r['split'], size=list(r['size']), crop=list(r['crop']),
                             sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),
                             xy=r['gt'], box=r['box']) for r in records])


def grouped_splits(records, folds, seed):
    """先按原文件 SHA256 分组，再划分，重复图片始终留在同一折。"""
    hashes=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in records]
    groups=list(dict.fromkeys(hashes))
    if len(groups)<folds:
        raise ValueError('去重后图片组数小于折数')
    splitter=KFold(folds,shuffle=True,random_state=seed)
    for train_groups,val_groups in splitter.split(groups):
        selected={groups[i] for i in val_groups}
        val_idx=np.array([i for i,h in enumerate(hashes) if h in selected])
        train_idx=np.array([i for i,h in enumerate(hashes) if h not in selected])
        assert not ({hashes[i] for i in train_idx}&{hashes[i] for i in val_idx})
        yield train_idx,val_idx


def train_fold(args, train, train_idx, val_idx, fold, out, device):
    seed_everything(args.seed + fold)
    model = FoveaNet(pretrained=True).to(device)
    decoder_ids = {id(p) for m in [model.lateral, model.smooth, model.head, model.presence] for p in m.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in decoder_ids], 'lr': args.lr * .2},
        {'params': [p for p in model.parameters() if id(p) in decoder_ids], 'lr': args.lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, history = float('inf'), []
    train_records = [train[i] for i in train_idx]
    val_records = [train[i] for i in val_idx]
    cache = tensors(train_records, device)
    checkpoint = out / f'fold_{fold}.pt'
    since = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.freeze_batchnorm()
        permutation = torch.randperm(len(train_records), device=device)
        losses = []
        for ids in permutation.split(args.batch):
            images, points, boxes, visible = augment(*[c[ids] for c in cache], negative_rate=args.negative_rate)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, points, boxes, visible)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            xy, vis, boxes = predict_records(model, val_records, device, args.batch)
            rmse = score(val_records, xy, vis, args.visibility_threshold)
            item = dict(epoch=epoch, loss=float(np.mean(losses)), val_rmse=rmse,
                        min_visibility=float(vis.min()), seconds=round(time.time()-since,1))
            history.append(item)
            print(json.dumps(dict(fold=fold, **item)), flush=True)
            if rmse < best:
                best = rmse
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                epoch=epoch, val_rmse=best, fold=fold, side=args.side,
                                train_ids=[r['id'] for r in train_records], val_ids=[r['id'] for r in val_records],
                                split_strategy='sha256_grouped_kfold_v1',
                                train_sha256=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in train_records],
                                val_sha256=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in val_records],
                                args=vars(args)), checkpoint)
                np.savez(out/f'fold_{fold}_val.npz', ids=np.array([r['id'] for r in val_records]),
                         xy=xy, visibility=vis, boxes=boxes)
            (out/f'fold_{fold}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    del model, optimizer, cache
    if device.type == 'cuda': torch.cuda.empty_cache()
    return best


def write_submission(root, out, records, xy, visibility, boxes, threshold, invisible_ids=()):
    xy = xy.copy()
    rejected = visibility < threshold
    for i,r in enumerate(records):
        rejected[i] |= r['id'] in invisible_ids
    xy[rejected] = 0
    lookup = {r['id']: (xy[i], r['size']) for i,r in enumerate(records)}
    with (Path(root)/'sample_submission.csv').open(encoding='utf-8-sig', newline='') as f:
        template = list(csv.DictReader(f))
    seen, rows = set(), []
    for row in template:
        name, axis = row['ImageID'].rsplit('_Fovea_',1)
        image_id = int(Path(name).stem)
        if image_id not in lookup or axis not in ['X','Y']:
            raise ValueError(f'提交模板中含未知项目：{row}')
        key = (image_id, axis)
        if key in seen: raise ValueError(f'提交模板存在重复：{key}')
        seen.add(key)
        value = float(lookup[image_id][0][0 if axis=='X' else 1])
        if not np.isfinite(value): raise ValueError('预测中出现 NaN/Inf')
        rows.append(dict(ImageID=row['ImageID'], value=f'{value:.6f}'))
    if seen != {(r['id'],axis) for r in records for axis in ['X','Y']}:
        raise ValueError('提交模板未覆盖全部测试图的 X、Y')
    with (out/'submission.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ImageID','value']); writer.writeheader(); writer.writerows(rows)
    details=[]
    for i,r in enumerate(records):
        details.append(dict(image=r['id'],Fovea_X=float(xy[i,0]),Fovea_Y=float(xy[i,1]),
                            visibility=float(visibility[i]),rejected=bool(rejected[i]),
                            macula_box=[round(float(v),3) for v in boxes[i]]))
    (out/'predictions.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    render_predictions(records,xy,boxes,out/'prediction_preview.jpg',rejected)
    print(f'Submission verified: {len(rows)} rows, {len(records)} images: {out / "submission.csv"}',flush=True)


def render_predictions(records, xy, boxes, target, rejected=None):
    width, height = 320, 260
    canvas=Image.new('RGB',(5*width,math.ceil(len(records)/5)*height),'white')
    drawer=ImageDraw.Draw(canvas)
    for i,r in enumerate(records):
        im=Image.open(r['path']).convert('RGB')
        im.thumbnail((width-8,height-30))
        xoff,yoff=(i%5)*width,(i//5)*height+24
        canvas.paste(im,(xoff,yoff))
        sx,sy=im.width/r['size'][0],im.height/r['size'][1]
        if rejected is None or not rejected[i]:
            px,py=xy[i]*[sx,sy]+[xoff,yoff]
            b=boxes[i]*[sx,sy,sx,sy]+[xoff,yoff,xoff,yoff]
            drawer.rectangle(tuple(b),outline='cyan',width=2)
            drawer.line((px-5,py,px+5,py),fill='lime',width=2)
            drawer.line((px,py-5,px,py+5),fill='lime',width=2)
        drawer.text((xoff+4,yoff-21),f'{r["id"]}: ({xy[i,0]:.1f}, {xy[i,1]:.1f})',fill='black')
    canvas.save(target,quality=95)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',choices=['train','predict','audit'],default='train')
    parser.add_argument('--data',default='D:/2026task3/detection')
    parser.add_argument('--output',default=str(Path(__file__).resolve().parent/'models'/'resnet_cv'))
    parser.add_argument('--side',type=int,default=384)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--only-folds',type=int,nargs='+',default=None)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--lr',type=float,default=5e-4)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--negative-rate',type=float,default=.12)
    parser.add_argument('--visibility-threshold',type=float,default=.10,
                        help='合成负例训练的判定阈值；缺少真实负例，无法验证真实不可见图的召回率')
    parser.add_argument('--invisible-ids',type=int,nargs='*',default=[],help='已确认不可见的图片 ID，强制输出 (0,0)')
    parser.add_argument('--resume',action='store_true',help='跳过已完成的折；不恢复中途的优化器状态')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.side % 32: parser.error('--side 必须为32的倍数')
    if not 2 <= args.folds <= 80: parser.error('--folds 范围为2至80')
    if not 0 <= args.visibility_threshold <= 1: parser.error('阈值必须在0至1之间')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    if device.type=='cuda': torch.set_float32_matmul_precision('high')
    train,test=read_data(args.data,args.side)
    current_manifest=manifest(args.data,train+test,args.side)
    manifest_path=out/'data_manifest.json'
    if args.mode=='train':
        if args.resume and manifest_path.exists():
            previous=json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous['images']!=current_manifest['images'] or previous['side']!=args.side:
                raise ValueError('数据或分辨率已改变，请选择新的 --output 目录重新训练')
        manifest_path.write_text(json.dumps(current_manifest,indent=2),encoding='utf-8')
    print(f'Device={device}; train={len(train)}; test={len(test)}; real invisible training samples={sum(not r["visible"] for r in train)}',flush=True)
    if args.mode=='audit':
        (out/'data_audit.json').write_text(json.dumps(current_manifest,indent=2),encoding='utf-8')
        return
    if args.mode=='train':
        for fold,(train_idx,val_idx) in enumerate(grouped_splits(train,args.folds,args.seed)):
            if args.only_folds is not None and fold not in args.only_folds: continue
            history_path=out/f'fold_{fold}_history.json'
            finished=(history_path.exists() and json.loads(history_path.read_text())[-1]['epoch']==args.epochs)
            if args.resume and finished and (out/f'fold_{fold}.pt').exists() and (out/f'fold_{fold}_val.npz').exists():
                old=torch.load(out/f'fold_{fold}.pt',map_location='cpu',weights_only=True)
                if old.get('split_strategy')=='sha256_grouped_kfold_v1' and old['val_ids']==[train[i]['id'] for i in val_idx]:
                    continue
            train_fold(args,train,train_idx,val_idx,fold,out,device)
    checkpoints=sorted(out.glob('fold_*.pt'))
    if not checkpoints: raise FileNotFoundError('没有模型权重，请先 --mode train')
    all_xy,all_vis,all_boxes=[],[],[]
    oof={}
    metrics=[]
    for path in checkpoints:
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['side']!=args.side: raise ValueError('推理分辨率必须与训练一致')
        if len(ckpt['train_ids'])!=len(set(ckpt['train_ids'])) or set(ckpt['train_ids']) & set(ckpt['val_ids']):
            raise ValueError('训练/验证划分存在泄漏')
        if ckpt.get('split_strategy')!='sha256_grouped_kfold_v1' or set(ckpt['train_sha256'])&set(ckpt['val_sha256']):
            raise ValueError('检测到旧版划分或相同图片跨越训练/验证，请重新训练')
        model=FoveaNet().to(device); model.load_state_dict(ckpt['model'])
        xy,vis,boxes=predict_records(model,test,device,args.batch)
        all_xy.append(xy); all_vis.append(vis); all_boxes.append(boxes)
        fold_metrics=dict(fold=ckpt['fold'],epoch=ckpt['epoch'],val_rmse=ckpt['val_rmse'])
        metrics.append(fold_metrics)
        vp=out/f'fold_{ckpt["fold"]}_val.npz'
        if vp.exists():
            with np.load(vp) as data:
                for j,image_id in enumerate(data['ids']):
                    if int(image_id) in oof: raise ValueError('同一训练图存在重复折外预测')
                    oof[int(image_id)]=(data['xy'][j],float(data['visibility'][j]),data['boxes'][j])
        print(f'Predicted using {path.name}',flush=True)
        del model
    if oof:
        with (out/'oof_predictions.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f); writer.writerow(['data','GT_X','GT_Y','Pred_X','Pred_Y','visibility'])
            errors=[]
            box_ious=[]
            for r in train:
                if r['id'] not in oof: continue
                point,vis,pred_box=oof[r['id']]
                if vis < args.visibility_threshold: point=np.zeros(2)
                errors.extend((point-np.array(r['gt']))**2)
                writer.writerow([r['id'],*r['gt'],*point,vis])
                if r['visible']:
                    offset=np.array(r['crop'][:2]); extent=np.array(r['crop'][2:])-offset
                    gt_box=np.array(r['box'])*np.tile(extent,2)+np.tile(offset,2)
                    intersection=np.maximum(0,np.minimum(pred_box[2:],gt_box[2:])-np.maximum(pred_box[:2],gt_box[:2])).prod()
                    area_pred=np.maximum(0,pred_box[2:]-pred_box[:2]).prod()
                    area_gt=np.maximum(0,gt_box[2:]-gt_box[:2]).prod()
                    box_ious.append(float(intersection/max(area_pred+area_gt-intersection,1e-8)))
        oof_rmse=float(np.sqrt(np.mean(errors)))
        oof_box_iou=float(np.mean(box_ious)) if box_ious else None
    else: oof_rmse,oof_box_iou=None,None
    report=dict(folds=metrics,oof_images=len(oof),oof_rmse=oof_rmse,oof_macula_box_mean_iou=oof_box_iou,
                metric='sqrt(mean of all 2*N squared coordinate errors in original-image pixels)',
                validation_note='每折最优 epoch 由该折验证集选择；此分数用于开发，不是独立测试成绩。',
                split_strategy='sha256_grouped_kfold_v1',
                visibility_note='训练中无真实不可见图时，只有合成裁剪负例监督，真实不可见判定性能未知。',
                visibility_threshold=args.visibility_threshold,real_invisible_training_samples=sum(not r['visible'] for r in train),
                ensemble_size=len(checkpoints),side=args.side)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    write_submission(args.data,out,test,np.mean(all_xy,axis=0),np.mean(all_vis,axis=0),
                     np.mean(all_boxes,axis=0),args.visibility_threshold,args.invisible_ids)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
