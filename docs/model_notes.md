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

## 实验记录（真实结果，leaf-aware 划分，3 个 seed 均值±标准差）
划分：leaf-aware 70/15/15（train 37,978 / val 8,155 / test 8,151）。每个实验用 seed=42/43/44 各跑一次；Test 用 `scripts/evaluate_test.py` 在 best checkpoint 上评估（测试集仅用于最终评估）。下表为 3 个 seed 的均值±样本标准差。

| 实验 | 模型 | 增强 | Seeds | Val Macro-F1 | Test Macro-F1 | Test Acc |
|---|---|---|---|---|---|---|
| E1 | SmallCNN | 否 | 42/43/44 | 0.9864 ± 0.0017 | 0.9865 ± 0.0012 | 0.9893 |
| E2 | ResNet18 冻结 | 否 | 42/43/44 | 0.9541 ± 0.0013 | 0.9536 ± 0.0012 | 0.9611 |
| E3 | ResNet18 微调 layer4 | 否 | 42/43/44 | 0.9952 ± 0.0005 | 0.9950 ± 0.0006 | 0.9961 |
| E4 | ResNet18 微调 layer4 | 是 | 42/43/44 | 0.9949 ± 0.0016 | 0.9931 ± 0.0011 | 0.9951 |

结果观察：
- E3 显著优于 E2：Test Macro-F1 0.9536 → 0.9950（解冻 layer4 才发挥预训练优势）。
- E1（从零 SmallCNN 0.9865）高于 E2（冻结 0.9536）：仅训分类头在 38 类上欠拟合，未发挥预训练优势。
- E4 增强≈/略低于 E3（0.9931 vs 0.9950），本数据/预算下增强收益不明显。
- 三个 seed 标准差均 ≤ 0.0017，结论稳健、可复现。
- 混淆矩阵与曲线见 `results/figures/`（由 `scripts/analyze_results.py` 对指定 seed 结果生成）。

## 训练协议与公平性说明
- 早停：以验证集 Macro-F1 为准则，`patience=5`，上限 25 epoch；取验证 Macro-F1 最佳的 checkpoint。
- 学习率按方法设定：SmallCNN 1e-3；ResNet18 冻结分类头 1e-3；微调 layer4 1e-4（见 `configs/experiment.yaml`）。
- 数据划分 leaf-aware 70/15/15，seed=42/43/44；**测试集仅用于最终评估**，不参与训练、早停与模型选择。
- 公平性：E1–E4 使用同一早停规则与同一划分；以**独立测试集** Macro-F1 为主、验证集为辅（验证集因参与选模略偏乐观）。
- Windows 下当 `num_workers>0` 时启用 `persistent_workers=True`，避免每轮重建 DataLoader worker 引起的偶发死锁。
- 早停均发生在验证指标平台期后，并非无理截断（以下为 seed42 示例）：

| 实验 | 轮数 | 最佳 val@轮 | 最佳后 val 走势 |
|---|---|---|---|
| E1 | 25（跑满） | 0.9884@23 | 仍在 ~0.98 波动，未触发早停 |
| E2 | 12（早停） | 0.9526@7 | 之后 ~0.950±0.001，平台 |
| E3 | 12（早停） | 0.9956@7 | 峰值后回落，近上限 |
| E4 | 21（早停） | 0.9958@16 | ~0.995 平台 |

- 稳健性：每个实验用 seed=42/43/44 三次，报告均值±标准差（均 ≤0.0017）；如需更严格可关早停统一固定轮数做对照。

## 训练命令（splits 已就绪）
`--lr` 缺省时按模型自动选：CNN 用 `cnn_learning_rate`，ResNet18 微调用 `finetune_learning_rate`，冻结用 `classifier_learning_rate`。增强默认取配置，可用 `--augment/--no-augment` 覆盖。

```powershell
# 多 seed：对每个 seed ∈ {42,43,44} 各跑一遍，结果放 results/seed{seed}/eN
# E1 SmallCNN 基线（无增强）
python scripts/train.py --model cnn --no-augment --seed 42 --output-dir results/seed42/e1

# E2 ResNet18 冻结 backbone（无增强）
python scripts/train.py --model resnet18 --freeze --pretrained --no-augment --seed 42 --output-dir results/seed42/e2

# E3 ResNet18 微调 layer4（无增强）
python scripts/train.py --model resnet18 --finetune-layer4 --pretrained --no-augment --seed 42 --output-dir results/seed42/e3

# E4 ResNet18 微调 layer4 + 轻量增强
python scripts/train.py --model resnet18 --finetune-layer4 --pretrained --augment --seed 42 --output-dir results/seed42/e4
```

说明：真实数据模式下会读取 `splits/train.csv`、`splits/val.csv` 和 `splits/class_to_idx.json`；`--pretrained` 首次联网下载 ImageNet 权重。

## 测试集评估与结果分析
```powershell
# 对某个 seed 的某实验 best checkpoint 做测试集评估（示例 seed42 E1）
python scripts/evaluate_test.py --checkpoint results/seed42/e1/checkpoints/cnn_best.pt --device cuda

# 对某个 seed 目录做汇总/画混淆矩阵与曲线（results-root 指向具体 seed）
python scripts/analyze_results.py --results-root results/seed42 --experiments e1 e2 e3 e4
```

## 指标
Accuracy、Macro-F1、Macro Recall、混淆矩阵、训练时长。

## 进度
- [x] 实现 SmallCNN
- [x] 实现 ResNet18 分类头替换
- [x] 冻结 backbone / 解冻 layer4
- [x] 模型接口测试（logits / 冻结 / 微调）通过
- [x] 训练管线 + 单入口脚本 + smoke test 通过
- [x] 真实数据训练 E1–E4（leaf-aware 划分）
- [x] 多 seed（42/43/44）均值±标准差
- [x] 数据增强消融（E4 vs E3：增强无明显提升）
- [x] 混淆矩阵与误分类分析（`results/figures/`）
- [ ] 外部评估（PlantDoc，数据待确认）

## 环境
- Python 3.13，torch 2.14.1+cu130，torchvision 0.29.1+cu130
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（8GB）

## 方案 B：132 类（PlantVillage + PlantWild）+ PlantDoc 外部评估
> 训练/验证用 `splits_combined/`（合并 132 类，**无内部 test**）；测试用 **PlantDoc 外部集**（2,561 张 / 28 类，映射到 132 类标签空间 `splits_combined/plantdoc_test.csv`）。3 个 seed（42/43/44），结果为均值±样本标准差。

| 实验 | 模型 | 增强 | val Acc | val Macro-F1 | 外部(PlantDoc) Acc | 外部 Macro-F1 |
|---|---|---|---|---|---|---|
| E1 | SmallCNN | 否 | 0.8888 ± 0.0035 | 0.5028 ± 0.0074 | 0.1804 ± 0.0071 | 0.1962 ± 0.0064 |
| E2 | ResNet18 冻结 | 否 | 0.8696 ± 0.0042 | 0.5434 ± 0.0049 | 0.1775 ± 0.0059 | 0.1978 ± 0.0035 |
| E3 | ResNet18 微调 layer4 | 否 | 0.9228 ± 0.0024 | 0.6370 ± 0.0019 | 0.3223 ± 0.0115 | 0.3291 ± 0.0035 |
| E4 | ResNet18 微调 layer4 | 是 | 0.9275 ± 0.0012 | 0.6603 ± 0.0092 | 0.3057 ± 0.0131 | 0.3084 ± 0.0133 |

结论：
- 微调（E3/E4）在内部与**外部真实场景**都显著优于从零/冻结（E1/E2）：外部 Acc 0.18 → 0.31~0.32，外部 Macro-F1 0.20 → 0.31~0.33。
- 增强（E4）内部略高，外部与 E3 相当，无显著泛化增益。
- 域差异巨大：内部 val Acc ~89–93% → PlantDoc 外部 Acc 仅 18–32%（132 类模型预测到 PlantWild 专有类也计为错误，是主要原因之一）。
- 3 seed 标准差 ≤ 0.013，结论稳健。

产物：`results/combined132/seed{42,43,44}/e{1..4}/`（checkpoints + 内部 `*_metrics.json` + `external_metrics.json` + `external_predictions.csv`）。

复现（幂等、可断点续跑）：
```powershell
python train_132_orchestrator.py     # 或双击 run_training_132.bat
```

数据：PlantVillage（`data/plantvillage`）+ PlantWild v2（`data/plantwild`，CC BY-NC-ND 4.0）；PlantDoc 外部（`data/plantdoc`，CC BY 4.0）。
