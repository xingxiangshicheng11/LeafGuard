# 域泛化分析（PlantDoc 外部评估）

本文件记录 132 类模型（PlantVillage + PlantWild 合并训练）在 **PlantDoc** 真实场景外部集上的表现与错误分析。所有数值来自 `results/combined132/` 的真实运行，3 个 seed（42/43/44）。

## 1. 评估设置
- 训练/验证：`splits_combined/`（132 类，train 55,625 / val 9,878；无内部 test）。
- 外部测试：PlantDoc Cropped（`data/plantdoc`），映射到 PlantVillage 28 类，共 **2,561 张**（`splits_combined/plantdoc_test.csv`，132 类标签空间）。
- 指标在 PlantDoc 实际覆盖的 28 类上计算 Macro；预测到其它类计为 Accuracy 错误。

## 2. 外部指标（mean ± std，3 seed）
| 实验 | 外部 Accuracy | 外部 Macro-F1 | 外部 Macro-Recall |
|---|---|---|---|
| E1 SmallCNN | 0.180 ± 0.007 | 0.196 ± 0.006 | 0.167 |
| E2 ResNet18 冻结 | 0.178 ± 0.006 | 0.198 ± 0.004 | 0.163 |
| E3 ResNet18 微调 | **0.322 ± 0.012** | **0.329 ± 0.004** | 0.291 |
| E4 微调+增强 | 0.306 ± 0.013 | 0.308 ± 0.013 | 0.276 |

结论：**微调（E3/E4）在真实场景上显著优于从零/冻结（E1/E2）**，外部 Acc 约翻倍（0.18 → 0.31–0.32）；增强（E4）相对 E3 无提升甚至略降。

## 3. 错误分析（seed42 示例）
### 3.1 最易混类别对（top confusions）
- E1（SmallCNN）：玉米北方叶枯 → 玉米灰斑（65）；番茄斑枯 → 番茄细菌性斑点（25）；马铃薯早疫 ↔ 番茄早疫（19）；马铃薯晚疫 ↔ 番茄晚疫（18）。
- E3（微调）：玉米北方叶枯 → 玉米灰斑（86）；番茄细菌性斑点 → 番茄斑枯（28）；马铃薯早疫 ↔ 晚疫（23）；番茄花叶病毒 → 番茄黄化曲叶病毒（23）。

共性：**同类作物内的相似病害**（玉米两种叶斑、番茄/马铃薯多种疫病与斑点病）最易混，跨作物混淆较少。

### 3.2 最弱类别（F1 最低，support ≥ 5）
- 多个 **healthy（健康）类** 外部 F1 接近 0：Tomato/Grape/Peach/Raspberry/Soybean/Cherry healthy 等。
- 原因：PlantDoc 的"健康叶"来自真实网图（背景杂乱、光照/角度不同），与 PlantVillage 纯色背景的健康叶分布差异极大，模型倾向判成某种病害。

## 4. 原因与局限
- **域差异**：训练图为受控/合成背景，PlantDoc 为真实网图，背景、光照、拍摄条件差别大，是掉点主因。
- **闭集限制**：132 类模型会把 PlantDoc 样本预测到 PlantWild 专有类，这些在 28 类评估中计为错误，进一步拉低 Accuracy。
- **healthy 类尤其脆弱**：真实场景下"健康"更依赖整体外观/背景，模型学到的是受控背景下的特征。
- **标签噪声/质量**：PlantDoc 含教学图、网页截图、水印图，存在少量标签噪声（见 `docs/plantdoc_spotcheck.md`）。

## 5. 复现
```powershell
python scripts/evaluate_external.py --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --manifest splits_combined/plantdoc_test.csv --device cuda --num-workers 0 --metrics-out results/combined132/seed42/e3/metrics/external_metrics.json
python scripts/analyze_external_results.py
```
图：`results/figures/combined132/plantdoc/plantdoc_e{1..4}_seed42_confusion.png`（PlantDoc 28 类混淆矩阵）。

## 6. 启示
- 真实部署应做**域适应/去背景**，或补充真实场景训练数据；
- 对"健康/病害"与低置信度结果应做 **OOD/不确定性提示**，避免误用（Demo 已加 OOD 拒识）。
