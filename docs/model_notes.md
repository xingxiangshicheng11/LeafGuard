# 成员 A：模型实验记录

只记录真实运行结果；未运行的实验不得填写数值。

## 模型
- SmallCNN：4 段 Conv-BN-ReLU-Pool + AdaptiveAvgPool + Dropout + Linear，从零训练。
- ResNet18：torchvision ImageNet 权重，替换 `fc`；支持冻结 backbone 与解冻 layer4 微调。

## 实验计划
采用「现代化复现」：以 Mohanty 2016（PlantVillage 高准确率）为锚点，用 ResNet18 复现迁移学习结论；微调对比参考 Too 2019。参考文献与 DOI 见 `docs/RELATED_WORK.md` 第 11 节。

| 实验 | 模型/策略 | 用途 | 参考 |
|---|---|---|---|
| E1 | SmallCNN 从零训练 | 基线 | Too 2019（从零 vs 预训练对比） |
| E2 | ResNet18 冻结 backbone | 迁移学习基线 | Mohanty 2016 / Too 2019 |
| E3 | ResNet18 微调 layer4 | 评估微调价值 | Too 2019（微调对比） |
| E4 | ResNet18 + 轻量增强 | 评估增强影响 | Ferentinos 2018 / 常规增强做法 |
| E5（候选） | PlantDoc 共同类别外部评估 | 域泛化 | 候选贡献点，实验做完再评判 |

## 实验记录（真实运行结果）
| 实验 | 模型 | 增强 | Seed | Epoch | Val Macro-F1 | Test Macro-F1 | 训练时长 | 备注 |
|---|---|---|---|---|---|---|---|---|
| E1 | SmallCNN | 否 | 42 | 21（早停，计划25） | 0.9798 | 0.9780 | 3775s（≈63min） | test 用 scripts/evaluate_test.py 另评 |

## 训练命令（B 的 splits 已就绪）
`--lr` 缺省时按模型自动选：CNN 用 `cnn_learning_rate`，ResNet18 微调用 `finetune_learning_rate`，冻结用 `classifier_learning_rate`。增强默认取配置，可用 `--augment/--no-augment` 覆盖。

```powershell
# E1 SmallCNN 基线（无增强）
python scripts/train.py --model cnn --no-augment --output-dir results/e1

# E2 ResNet18 冻结 backbone（无增强）
python scripts/train.py --model resnet18 --freeze --pretrained --no-augment --output-dir results/e2

# E3 ResNet18 微调 layer4（无增强）
python scripts/train.py --model resnet18 --finetune-layer4 --pretrained --no-augment --output-dir results/e3

# E4 ResNet18 微调 layer4 + 轻量增强
python scripts/train.py --model resnet18 --finetune-layer4 --pretrained --augment --output-dir results/e4
```

说明：真实数据模式下会读取 `splits/train.csv`、`splits/val.csv` 和 `splits/class_to_idx.json`；`--pretrained` 首次联网下载 ImageNet 权重。

## 测试集评估与结果分析
```powershell
# 在测试集上评估某个 checkpoint（结果存 results/<exp>/metrics/<model>_test_metrics.json）
python scripts/evaluate_test.py --checkpoint results/e1/checkpoints/cnn_best.pt --device cuda

# 汇总各实验指标 + 训练曲线 + 混淆矩阵 + 最易混淆类别对（产物在 results/figures/）
python scripts/analyze_results.py --experiments e1 e2 e3 e4
```

## 指标
Accuracy、Macro-F1、Macro Recall、混淆矩阵、训练时长。

## 进度
- [x] 实现 SmallCNN
- [x] 实现 ResNet18 分类头替换
- [x] 冻结 backbone / 解冻 layer4
- [x] 模型接口测试（logits / 冻结 / 微调）通过
- [x] 训练管线 + 单入口脚本 + smoke test 通过
- [x] 真实数据 E1（SmallCNN）：Val Macro-F1 0.9798 / Test Macro-F1 0.9780
- [ ] 真实数据 E2–E4（ResNet18 冻结/微调/增强）运行中
- [ ] 数据增强消融（E4 对比 E2/E3）
- [ ] 混淆矩阵与误分类分析（工具已就绪）
- [ ] 外部评估（PlantDoc，数据待确认）

## 环境
- Python 3.13，torch 2.14.1+cu130，torchvision 0.29.1+cu130
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8GB）
