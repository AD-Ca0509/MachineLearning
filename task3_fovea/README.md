# 课设3 黄斑中央凹定位

80张训练图、20张测试图；同内容重复训练图按哈希分组验证，全量阶段平均重复标注后79张独立图。数据根目录需含 train/、test/、train_location/、fovea_localization_train_GT.csv 与 sample_submission.csv。

## 从头训练

```powershell
pip install -r requirements.txt
python train.py --data D:/2026task3/detection
```

默认顺序训练 resnet、convnext、refit、fresh。可以用 --stage 选择其中一个；refit依赖已训练的ConvNeXt五折检查点。训练更新 models/，耗时较长。

## 推理

```powershell
python predict.py --data D:/2026task3/detection --output reproduced_submission.csv
```

**先准备模型权重**：本仓库仅含代码、配置和提交 CSV，不含17个大体积权重，也不含原机 runtime。新机器需自行安装依赖，再训练生成 model_config.json 指定的 models/。仅下载代码不能直接执行上述推理。原机最终权重仍保存在原工程，不因本次上传而删除。

模型为 ResNet50/ConvNeXt-Tiny + 64通道FPN，联合热图、区域与可见性，水平翻转TTA及局部soft-argmax。融合组权重为30%、30%、20%、20%，共5+5+5+2个模型。

本地 ResNet 五折 OOF RMSE 约15.7962，ConvNeXt 约15.8007，两类骨干各50%融合约15.408445。上述为开发阶段估计，方案选择仍可能产生偏差。新增全量分支没有可靠的独立整体OOF结果，训练拟合误差不代表泛化效果。

输出为 ImageID,value，共40行坐标。CUDA重新运行可能产生浮点差异，训练随机性也不保证精确重建原CSV。

## 模型结构

ResNet50或ConvNeXt-Tiny输出四级特征，1×1卷积统一为64通道，自顶向下上采样相加。空间头输出中央凹热图和黄斑区域图，最深层全局特征另接128维可见性头。热图经区域概率软门控，在峰值邻域内作soft-argmax，逆变换到原图坐标。

## 实验方案与训练过程

相同内容图像按SHA256分组五折，避免重复图跨训练与验证。ResNet训练100轮；ConvNeXt训练120轮，骨干/预测头分组学习率，保存原始与EMA中验证较好的参数。再进行全量refit和fresh分支训练。损失包括热图交叉熵、区域BCE与Dice、可见性BCE和归一化坐标平方误差。

## 分析过程

按原图X/Y坐标计算OOF RMSE，比较单骨干与双骨干融合；检查裁剪偏移、缩放、翻转逆变换及局部解码。全量训练只能说明拟合程度，不能与OOF直接比较。小幅固定偏移只有局部交叉拟合证据，因此不作为已确认有效的泛化改进；无真实不可见验证样本，也不声称已验证该分支的实际可靠性。
