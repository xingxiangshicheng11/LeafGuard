# 分阶段实施计划

## Phase 0：环境检查
两台电脑分别运行 `python scripts/check_environment.py`，确认 Python、torch、torchvision、CUDA。

## Phase 1：数据与接口
B 扫描 PlantVillage、统计类别、检查损坏/重复图像、生成固定分层 70/15/15 split 和类别映射。A 审核标签和 split。

## Phase 2：最小训练
B 提供 Dataset/DataLoader/transforms。A 实现 CNN、ResNet18，先用小样本跑 1–2 epoch smoke test。

## Phase 3：核心实验
A 运行 CNN、冻结 ResNet18、微调 ResNet18；保存日志、checkpoint、指标。

## Phase 4：泛化分析
A 做数据增强对比、混淆矩阵和误分类分析。B 审计 PlantDoc 标签映射并准备外部评估与单图预测。

## Phase 5：课程交付
共同完成 3–5 页论文、8–10 页 PPT、约 10 分钟汇报和可运行演示。
每阶段结束需记录实际测试结果；未运行的实验不能标记完成。
