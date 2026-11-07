# 课设4 眼底血管分割

训练/测试各20张，原图565×584。四通道U-Net，从头训练；输入归一化RGB与绿色局部对比度。验证留出3、8、13、18，最终种子43及2026的两个精调模型，8方向TTA。

数据根目录包含 train/image、train/label、test/image、sample_submission.csv 和课程提供的 segmentation_to_csv.py。provided/ 中保存原转换脚本供代码复查；main.py 实际调用的是**数据根目录内**的脚本。请使用课程原始数据及配套脚本。

```powershell
pip install -r requirements.txt
python main.py --data D:/2026task-4/segmentation --check
python main.py --data D:/2026task-4/segmentation --output reproduced_submission.csv
```

完整入口会基础训练、精调、推理，再调用原脚本导出；中间权重保存在临时目录并清理，使用单独输出位置可保留仓库中的原submission。依赖CUDA GPU，默认训练耗时较长。

## 黑白与RLE

实际训练标签白色血管，与题目文字黑色血管约定相反。RLE编码血管位置：先原尺寸阈值化，再以白色血管PNG交给原脚本；原脚本双线性缩放512×512，非零置1，按行展开，从1开始编码。当前选择阈值0.385。

最新本地4图验证 Dice 0.8582091（原脚本口径），最终线上0.86179来自本人反馈，两者不可混用。精调与导出修正的联合改善不是单独网络结构的纯消融。

## 复现限制

不含最终权重；原机保留备份。完整冷启动重新训练可能因CUDA非确定性产生不同预测，不能承诺逐像素/字节相同。最新统一入口未从空缓存整体复跑验证一次，不能把分阶段执行误写成已经完成该项验证。精确重建既有CSV需原权重、固定阈值、环境与数据。
