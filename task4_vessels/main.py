"""在 VS Code 中打开本文件，点击“运行 Python 文件”即可重新训练并生成 submission.csv。

本机可用的 Python 解释器：D:\\py jieshiqi\\Scripts\\python.exe
依赖：torch、torchvision、numpy、pandas、Pillow、scipy；需要 NVIDIA CUDA GPU。
数据默认位于 D:/2026task-4/segmentation，可通过 --data 更改。
模型、预测图等中间文件放在系统临时目录，运行结束后自动清理。
只检查环境与数据：python main.py --check

复现此次提交所用环境（不要仅凭随机种子承诺跨环境逐像素一致）：
Python 3.14.0，torch 2.14.0+cu130，torchvision 0.29.0+cu130，
numpy 2.5.0，pandas 3.0.6，Pillow 12.2.0，scipy 1.18.1，CUDA 13.0。
GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8GB。
验证：种子 42，基础训练 2500 步，再精调 2000 步，留出图 3、8、13、18；
最终模型：种子 43、2026，各基础训练 3500 步，再精调 2500 步。
精调使用 192×192 图块、batch=12、初始学习率 0.0003 和八方向预测。
精调图像尺寸由 --resolution 指定，默认 0（原尺寸）；提交转换为 512×512。
导出必须先在原尺寸二值化，再交给原转换脚本执行双线性缩放及 >0 判断。
验证标签使用同样的转换规则；该验证口径根据提供脚本推断，线上分数仍需实际评测。
默认参数对应现有 submission.csv 的训练流程；修改步数属于新的实验。
CUDA 训练包含非确定性运算，所以从头训练可能产生略有差别的预测。
若竞赛要求逐像素复现，应一并提供该次最终模型权重及该次选出的阈值。
完整重跑且保留现有提交的命令：
python main.py --output reproduced_submission.csv
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

# 避免在代码目录中额外生成 __pycache__ 文件夹。
sys.dont_write_bytecode = True


def check_environment(data):
    packages = {'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
                'pandas': 'pandas', 'PIL': 'Pillow', 'scipy': 'scipy'}
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError('缺少依赖：' + ', '.join(missing) +
                           '\n请先在 VS Code 选择 D:/py jieshiqi/Scripts/python.exe 解释器。')
    for relative in ['train/image', 'train/label', 'test/image']:
        directory = data / relative
        if not directory.is_dir() or not any(directory.glob('*.jpg')):
            raise FileNotFoundError(f'数据目录不存在或没有 JPG 图片：{directory}')
    for relative in ['segmentation_to_csv.py', 'sample_submission.csv']:
        if not (data / relative).is_file():
            raise FileNotFoundError(f'缺少数据配套文件：{data / relative}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('没有检测到 CUDA GPU。请选择已安装 CUDA 版 PyTorch 的 Python 解释器。')
    print(f'Python：{sys.executable}', flush=True)
    print(f'GPU：{torch.cuda.get_device_name(0)}', flush=True)
    print(f'数据目录：{data}', flush=True)


def export_submission(data, probabilities, validation, output, temporary):
    import numpy as np
    import pandas as pd
    from PIL import Image

    def converter_mask(binary):
        # Exactly match segmentation_to_csv.py: uint8 mask -> bilinear -> >0.
        return np.asarray(Image.fromarray((binary*255).astype(np.uint8)).resize(
            (512, 512), Image.Resampling.BILINEAR)) > 0

    # 仅使用留出验证图的真值选择阈值，不使用测试集标签。
    with np.load(validation) as val:
        val_prob = val['probabilities'].copy()
        val_labels = np.stack([np.asarray(Image.open(
            data/'train'/'label'/f'{ident}.jpg').convert('L').resize(
                (512, 512), Image.Resampling.BILINEAR)) > 0 for ident in val['ids']])
    scores = []
    for threshold in np.arange(.25, .651, .005):
        binary = np.stack([converter_mask(p > threshold) for p in val_prob])
        dice = 2*(binary & val_labels).sum((1, 2)) / (
            binary.sum((1, 2)) + val_labels.sum((1, 2)))
        scores.append((float(dice.mean()), float(threshold)))
    dice, threshold = max(scores)
    print(f'按原转换脚本口径的留出验证 Dice：{dice:.6f}，阈值：{threshold:.3f}；非线上分数', flush=True)

    conversion_dir = temporary / 'conversion'
    image_dir = conversion_dir / 'image'
    image_dir.mkdir(parents=True)
    with np.load(probabilities) as payload:
        ids = [str(v) for v in payload['ids']]
        native_masks = {ident: p > threshold
                 for ident, p in zip(ids, payload['probabilities'])}
        masks = {ident: converter_mask(p) for ident, p in native_masks.items()}
    sample = pd.read_csv(data / 'sample_submission.csv', dtype={'Id': str})
    if len(set(ids)) != len(ids) or set(ids) != set(sample['Id']):
        raise ValueError('预测编号与 sample_submission.csv 不一致。')
    for ident, vessel in native_masks.items():
        # 原脚本编码白色像素，临时转换输入采用白色血管。
        # 若需要黑色血管图，其像素数组为 np.where(vessel, 0, 255)。
        Image.fromarray((vessel * 255).astype(np.uint8)).save(image_dir / f'{ident}.png')
    subprocess.run([sys.executable, str(data / 'segmentation_to_csv.py')],
                   cwd=conversion_dir, check=True)
    frame = pd.read_csv(conversion_dir / 'submission.csv',
                        dtype={'Id': str, 'Predicted': str}, keep_default_na=False)
    if len(frame) != len(sample) or not frame['Id'].is_unique:
        raise ValueError('CSV 编号缺失或重复。')
    frame = frame.set_index('Id').loc[sample['Id']].reset_index()
    for row in frame.itertuples(index=False):
        values = np.array([int(v) for v in row.Predicted.split()], dtype=np.int64)
        if len(values) % 2 or np.any(values <= 0):
            raise ValueError(f'{row.Id}：RLE 起点与长度必须成对且为正整数。')
        starts, lengths = values[::2], values[1::2]
        ends = starts + lengths - 1
        if np.any(ends > 512*512) or np.any(starts[1:] <= ends[:-1]):
            raise ValueError(f'{row.Id}：RLE 越界、未排序或存在重叠。')
        decoded = np.zeros(512*512, dtype=bool)
        for start, length in zip(starts, lengths):
            decoded[start-1:start-1+length] = True
        if not np.array_equal(decoded.reshape(512, 512), masks[row.Id]):
            raise ValueError(f'{row.Id}：RLE 解码与血管掩膜不一致。')
    checked = temporary / 'checked_submission.csv'
    frame.to_csv(checked, index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 保留原提交，直到新文件生成并通过校验后才替换。
    staged = output.with_suffix('.csv.tmp')
    shutil.copyfile(checked, staged)
    os.replace(staged, output)
    print(f'完成：{output}（{len(frame)} 张图，RLE 校验通过）', flush=True)


def main():
    parser = argparse.ArgumentParser(description='训练眼底血管分割模型并生成 submission.csv')
    parser.add_argument('--data', type=Path, default=Path('D:/2026task-4/segmentation'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'submission.csv')
    parser.add_argument('--steps', type=int, default=3500)
    parser.add_argument('--validation-steps', type=int, default=2500)
    parser.add_argument('--refine-steps', type=int, default=2500)
    parser.add_argument('--validation-refine-steps', type=int, default=2000)
    parser.add_argument('--resolution', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='只检查运行环境与数据')
    args = parser.parse_args()
    data = args.data.resolve()
    check_environment(data)
    if args.check:
        print('环境与数据检查通过。直接运行 main.py 即可重新生成提交文件。', flush=True)
        return
    if args.steps < 1 or args.validation_steps < 1:
        parser.error('训练步数必须为正整数。')
    import torch
    from train_vessels import train, infer
    with tempfile.TemporaryDirectory(prefix='retinal-vessels-') as temp:
        temporary = Path(temp)
        common = dict(data=str(data), batch=16, patch=144, init=None, lr=.001,
                      resolution=0, views=4)
        validation_dir = temporary/'validation'
        print('第一步：训练留出验证模型。', flush=True)
        train(SimpleNamespace(**common, out=str(validation_dir), seed=42,
                              steps=args.validation_steps, full=False))
        torch.cuda.empty_cache()
        refined_validation_dir = temporary/'validation_refined'
        refinement = dict(common, batch=12, patch=192, lr=.0003,
                          resolution=args.resolution, views=8)
        train(SimpleNamespace(**dict(refinement, init=str(validation_dir/'best.pt')),
                              out=str(refined_validation_dir), seed=42,
                              steps=args.validation_refine_steps, full=False))
        torch.cuda.empty_cache()
        checkpoints = []
        for seed in [43, 2026]:
            model_dir = temporary/f'seed{seed}'
            print(f'第二步：使用全部训练图训练最终模型，随机种子 {seed}。', flush=True)
            train(SimpleNamespace(**common, out=str(model_dir), seed=seed,
                                  steps=args.steps, full=True))
            torch.cuda.empty_cache()
            refined_dir = temporary/f'refined{seed}'
            train(SimpleNamespace(**dict(refinement, init=str(model_dir/'best.pt')),
                                  out=str(refined_dir), seed=seed,
                                  steps=args.refine_steps, full=True))
            checkpoints.append(str(refined_dir/'best.pt'))
            torch.cuda.empty_cache()
        inference_dir = temporary/'inference'
        print('第三步：预测测试图。', flush=True)
        infer(SimpleNamespace(data=str(data), out=str(inference_dir), checkpoints=checkpoints,
                              resolution=args.resolution, views=8))
        print('第四步：生成并检查 CSV。', flush=True)
        export_submission(data, inference_dir/'test_probabilities.npz',
                          refined_validation_dir/'validation.npz', args.output.resolve(), temporary)


if __name__ == '__main__':
    main()
