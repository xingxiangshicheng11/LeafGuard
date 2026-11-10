# 双人协作流程

## 成员 A — Model & Experiment
CNN、ResNet18、冻结/微调、训练评估、指标和混淆矩阵、模型实验记录。

## 成员 B — Data & Application
PlantVillage 数据审计、划分、Dataset/DataLoader/transforms、PlantDoc 标签映射、单图预测。

## 每周安排
- 第 1 周：A 调研模型；B 配置环境、下载并审计数据；共同确认接口。
- 第 2 周：A 实现模型并 smoke test；B 完成划分和 Dataset；共同打通接口。
- 第 3 周：A 完成核心训练；B 完成预测接口和 PlantDoc 映射。
- 第 4 周：A 做增强消融与错误分析；B 做外部评估和演示。
- 第 5 周：A 整理实验章节；B 整理数据/系统章节；共同完成论文、PPT、彩排。

## Git
main 为稳定分支；A 使用 `feature/model`；B 使用 `feature/data-demo`。
开发前 `git pull`，提交只暂存自己负责的文件。通过 Pull Request 合并。
不提交数据、模型权重、虚拟环境。每周至少一次集成测试。
