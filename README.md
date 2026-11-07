# 机器学习课程设计

作者：范迅豪 · 学号：20245620 · 班级：人工智能2401班

本仓库提供四个课程任务的实际代码、预测 CSV 和复查说明。源码从各任务的本地正式工程复制，未为上传而重新训练，也没有把历史设想改写成已完成实验。

| 目录 | 任务 | 主要入口 | 本人反馈的最终线上成绩 |
|---|---|---|---|
| [task1_plant](task1_plant) | 植物物种五分类 | plant_classifier_v6.py | Accuracy 0.9656 |
| [task2_expression](task2_expression) | 人脸表情六分类 | train.py | Accuracy 0.769 |
| [task3_fovea](task3_fovea) | 黄斑中央凹定位 | train.py / predict.py | RMSE 19.66617 |
| [task4_vessels](task4_vessels) | 眼底血管分割 | main.py | Dice 0.86179 |

分数由本人于2026年11月7日提供。本仓库中的 CSV 为当日核对的本地文件；未取得完整平台提交记录，不能仅凭本地文件就证明它与每项线上成绩一一对应。特别是课设1，当前单架构配置与早期多架构注释不一致，详见任务说明。

## 安装与运行

每个任务目录都有独立 README 和 requirements.txt。请先安装与设备匹配的 PyTorch / torchvision，再安装所需包。数据来自课程，由使用者自行准备；仓库不包含训练/测试图片、真实测试标签、个人缓存、Python 运行环境及大体积模型权重。

任务2至4的历史环境为 Python 3.14、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130，GPU 为 RTX 5060 Laptop 8 GB。此处记录实际环境，不保证所有机器能安装相同构建。requirements.txt 是依赖清单，不是跨设备逐位复现的环境锁文件。

源码保留原来的本机默认数据路径；任务2至4可以使用 --data 更换路径。任务1需编辑脚本顶部的 BASE、OUTPUT_DIR 和输出 CSV 路径。不要不加检查就运行训练入口：训练耗时较长，默认输出可能覆盖任务目录中的 submission。

## 实验与复现边界

- 植物：分层五折、迁移学习、MixUp、SWA；当前 v6 有效配置仅 ConvNeXt-Tiny。已披露多尺度循环未生效、CutMix 未调用，不能据注释声称使用这些方法。
- 表情：六路等权融合。本地验证为3232/4242（76.190476%），不同于线上0.769；使用公开预训练权重，最终对511张精确重复测试图复用无冲突训练标签。
- 定位：ResNet/ConvNeXt FPN，热图、区域与可见性多任务。历史最好线上 RMSE 12.238；最终反馈19.66617，未证明增加全量分支有效。
- 分割：从头训练四通道 U-Net，修正原尺寸二值化与提供转换脚本的衔接。本地4图验证 Dice 0.8582091，线上反馈0.86179；小样本验证与反复阈值选择存在局限。

源码、配置和现有预测可用于代码复查。由于未上传大模型权重，任务3的 predict.py 不能在刚下载本仓库后直接推理，应先训练获得 models/ 中相应权重，或另外配置已保留权重。任务2的 --reuse-predictions 仅适用于已有缓存的原机器；冷启动需要下载公开权重并训练。任务4从头训练含 CUDA 非确定性，不保证逐像素或逐字节重建现有 CSV。

## 公开模型来源

- EfficientNet 表情预训练：[EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)
- DDAM 公开实现与权重：[DDAMFN](https://github.com/SainingZhang/DDAMFN)
- FMAE 公开实现与权重：[FMAE-IAT](https://github.com/forever208/FMAE-IAT)
- ConvNeXt / ResNet 等基础模型通过 torchvision 或 timm 创建。公开架构与权重不作为本人原创结构。

`SOURCE_MANIFEST.json` 记录本次复制的代码和提交文件 SHA256。`verify_sources.py` 可离线检查这些文件及 Python 语法，不运行训练、不下载权重、不读取用户凭据。

```powershell
python verify_sources.py
```

报告通过课程平台提交；代码通过本仓库供复查。原数据与公开预训练模型的使用应遵守课程规定及相应来源许可。
