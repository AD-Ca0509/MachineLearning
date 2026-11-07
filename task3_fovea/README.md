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

本地历史v2 OOF RMSE约15.408445，历史最好线上12.238；最终线上由本人反馈为19.66617。新增全量分支没有可靠的独立整体OOF结果，训练拟合误差不代表测试精度。不声称达到原先约11的目标。

输出为 ImageID,value，共40行坐标。CUDA重新运行可能产生浮点差异，训练随机性也不保证精确重建原CSV。
