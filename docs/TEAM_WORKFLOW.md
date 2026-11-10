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
分支模型：

- `main`：稳定/交付分支，只从 `develop` 合并，不直接接收 feature 分支。
- `develop`：集成分支，A、B 的 feature 分支 PR 到这里，集成测试通过后才进 `main`。
- `feature/model`（A）、`feature/data-demo`（B）：各自开发分支。

流程：

1. 各自在 feature 分支开发，开发前 `git pull`。
2. feature → PR → `develop`，由另一人 review 后合并。
3. 双方代码都进 develop 后，在 develop 上跑集成测试（train + predict 全链路）。
4. 集成测试通过 → develop → PR → `main`。

约束：提交只暂存自己负责的文件；不提交数据、模型权重、虚拟环境；`main`/`develop` 不直接 push，只走 PR 合并。

## Pull Request 规范
所有 PR 必须按统一模板填写。标题格式：

```
[模块] 类型: 简述
```

- 模块：`A`（模型/训练）｜`B`（数据/应用）｜`公共`
- 类型：`feat`｜`fix`｜`docs`｜`test`｜`refactor`｜`chore`
- 简述：中文，动词开头，≤ 50 字

示例：

```
[A] feat: 实现 ResNet18 冻结/微调训练
[B] feat: 生成 70/15/15 分层划分与类别映射
[公共] docs: 更新 PR 提交模板
```

正文模板已放在 `.github/PULL_REQUEST_TEMPLATE.md`，GitHub 会自动填充到 PR 描述框；按模块归属、变更文件、测试结果、未解决问题和检查清单逐项填写，不能留空（“无”也要写出来）。
