# 课设1 植物物种分类

训练500张，五类各100张；测试378张。主要脚本为 `plant_classifier_v6.py`，`plant_classifier.py` 是历史方案，不是其必需依赖。

## 运行

```powershell
pip install -r requirements.txt
python plant_classifier_v6.py
```

先编辑脚本的 BASE（数据根目录）、OUTPUT_DIR（检查点目录）与底部输出路径。原机数据结构为 BASE/dataset-for-task2/dataset-for-task2/train/<类别>、同级 test，以及 BASE/submission-for-task2.csv。脚本可能下载 ImageNet 预训练权重，并重新训练五折，原默认输出 D:/plant/submission_v6.csv。请在试跑前改成单独输出位置。

## 实际配置

ConvNeXt-Tiny，384输入，batch 24，分层五折 seed 42；分类头3轮，全网络12轮；标签平滑0.1，MixUp概率0.4，最后3轮SWA。输出列 ID,Category。

## 已知局限

当前 MODEL_NAMES 只有 ConvNeXt-Tiny，不能按开头旧注释认定训练了三架构。cutmix 函数未被训练路径调用；TTA 的 scale 变量未传给数据集，实际重复使用384中心裁剪及水平翻转。此次上传保留源代码，没有静默修正这些行为，以便报告复查。

线上0.9656来自本人反馈；缺少完整提交映射及折外概率，不能唯一确定该成绩对应当前配置，不将注释中的历史数值作为已核验消融结果。模型权重不在此仓库。
