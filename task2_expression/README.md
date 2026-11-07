# 课设2 人脸表情六分类

训练28275张灰度48×48图像，测试7065张。原数据路径 D:/2026task2/fer_data，包含 train/Angry、Fear、Happy、Neutral、Sad、Surprise 及 test/。

```powershell
pip install -r requirements.txt
python train.py --data D:/2026task2/fer_data --work-dir D:/fer_cache --output reproduced_submission.csv
```

首次运行需要 CUDA、下载公开预训练权重，并训练六路模型；最大源 checkpoint 约3.64 GB。以上新缓存路径会触发训练，耗时较长。只在同一数据已具备完整概率缓存时，才可用：

```powershell
python train.py --data D:/2026task2/fer_data --work-dir D:/existing_cache --reuse-predictions --output reproduced_submission.csv
```

组成：ConvNeXtV2-Tiny、EfficientNet-B2、DDAM、冻结FMAE线性头、FMAE MLP、FMAE最后4块微调。使用公开 ImageNet、人脸与表情预训练权重，来源见根目录 README；并非从零训练或完全没有外部数据知识。

固定15%分层验证 seed 42，共4242张，六路等权融合正确3232张（76.190476%）。确定方案后全量微调，不把全量模型对原验证集的成绩视为独立验证。

内部类别顺序 Angry,Fear,Happy,Neutral,Sad,Surprise，提交编号分别为0,1,2,5,3,4。最终对511张字节相同、训练标签无冲突的测试图复用训练标签；该处理未计入本地76.19%，未建立去掉此处理的独立效果对照。未使用测试集真实标签。

仓库不含公开大权重和本机预测缓存。已有缓存的再生成验证不能等同于已重新验证冷启动全部训练流程。

## 模型结构

三路CNN为ConvNeXtV2-Tiny、EfficientNet-B2、双注意力头DDAM；三路FMAE为冻结1024维特征的线性分类器、1024→256→6的MLP、最后4个Transformer块的微调模型。各路输出六类概率，先分支内平均种子与视角，再对六路等权平均。

## 实验方案与训练过程

统一使用seed 42的15%分层验证，按相同样本索引保存概率。训练采用AdamW、余弦学习率、标签平滑和部分批次MixUp。先用验证结果选定分支与TTA，再全量微调或重新拟合分类头；全量分支没有独立验证分数。DDAM采用EMA，FMAE只微调部分层以控制显存和过拟合。

CNN基准batch 64、学习率0.00008、分类头2倍学习率，最多30轮、最少20轮、早停耐心8轮。ConvNeXt分类头改为5倍学习率，权重衰减0.015、平滑0.04、MixUp系数0.2。DDAM使用batch 128、学习率0.00006、EMA 0.995，最多25轮、最少15轮、耐心7轮，正式训练前另有1轮预热。FMAE MLP使用batch 512、学习率0.001、权重衰减0.01，最多100轮并按验证结果停止。

## 分析过程

在4242张验证图上比较各分支，六路融合相较最佳单分支净增91张正确，约2.1452个百分点。逐类别召回率与混淆矩阵显示恐惧、悲伤、中性较易混淆，不能仅用总准确率概括六类效果。固定验证集被重复用于选择配置，未按人物分组，仍需承认选择偏差与样本相关性。
