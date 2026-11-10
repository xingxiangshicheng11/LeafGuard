# LeafGuard OpenCode Rules

## 双人职责
- A：`src/models/`、`src/training/`、模型测试、`docs/model_notes.md`。
- B：`src/data/`、`scripts/predict.py`、环境检查、`docs/data_audit.md`。
- 公共文件 `README.md`、`configs/`、`docs/INTERFACE_CONTRACT.md` 修改前先沟通。
- 未经允许不要修改对方模块。

## 规则
1. 先检查现有文件，不得无提示覆盖。
2. 一次只完成一个阶段，不擅自下载数据、训练或安装系统软件。
3. 不编造指标、图表、训练结果或参考文献。
4. 路径使用相对路径/pathlib，不写个人电脑绝对路径。
5. 固定随机种子，保存配置、日志、checkpoint 和指标。
6. 测试集不得用于调参；PlantDoc 默认只作外部评估。
7. 不提交 data、results 大文件、`.venv`、模型权重。
8. 训练前打印设备、数据集类别数、样本数和配置。
9. 训练增强只作用于 train；val/test 使用确定性变换。
10. 每项工作完成后说明文件变更、运行命令、测试结果和未解决问题。

## 默认实验
输入 RGB 224x224；seed=42；按类别分层 70/15/15；
模型为 SmallCNN 和 ImageNet 预训练 ResNet18；
指标 Accuracy、Macro-F1、Macro Recall、混淆矩阵、训练时间；
AdamW，batch size 初始 32，按显存调整。
