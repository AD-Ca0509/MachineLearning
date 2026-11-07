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

固定15%分层验证 seed 42，共4242张，六路等权融合正确3232张（76.190476%）。确定方案后全量微调，不把全量模型对原验证集的成绩视为独立验证。最终线上0.769为本人反馈。

内部类别顺序 Angry,Fear,Happy,Neutral,Sad,Surprise，提交编号分别为0,1,2,5,3,4。最终对511张字节相同、训练标签无冲突的测试图复用训练标签；该处理未计入本地76.19%，线上没有去掉此处理的独立对照。未使用测试集真实标签。

仓库不含公开大权重和本机预测缓存。已有缓存的再生成验证不能等同于已重新验证冷启动全部训练流程。
