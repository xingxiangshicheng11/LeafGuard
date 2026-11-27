# 数据一致性说明（源审计 vs 本地可用）

本文件对齐"数据集来源审计值"和"A 本机实际可用的数据"，避免文档数字互相矛盾。所有数值均为实际核查结果。

## 汇总
| 数据集 | 来源/审计值 | A 本机可用 | 说明 |
|---|---|---|---|
| PlantVillage | 38 类 / 54,305 | 38 类 / 54,305 | 一致 |
| PlantWild v2 | 115 类 / 11,488 | 115 类 / 11,488 | 一致 |
| PlantDoc | 28 类 / **2,572**（test 236 + train 2,336） | 28 类 / **2,564** | A 本机少 8 张（见下） |
| 38 类划分 `splits/` | — | train 37,978 / val 8,155 / test 8,151 | 一致 |
| 132 类划分 `splits_combined/` | — | train 55,625 / val 9,878，132 类 | 一致 |
| PlantDoc 测试清单 `splits_combined/plantdoc_test.csv` | — | **2,561 张 / 28 类**（去重后） | 用于 132 类外部评估 |

## PlantDoc 差 8 张的原因
- 源仓库审计：2,572 张（`data_audit.md` / `plantdoc_mapping.md` 记录）。
- A 本机重新下载解压时受 **Windows 长路径（MAX_PATH）/非法字符** 影响，实际落盘 **2,564 张**；构建 132 类测试清单时再去掉 3 个完全重复副本 → **2,561 张**。
- PlantDoc **仅用于外部评估、不参与训练**，故该差异**不影响训练集与已报告的内部指标**；仅使外部评估样本从约 2,572 降为 2,561（约 -0.4%）。

## 如何恢复到源 2,572
1. 提供原始 `PlantDoc-Dataset` zip（或让 B 提供其已审计版本，B 已处理过 87 个非法/超长文件名）。
2. 解压到 `data/plantdoc/<类>/`，用**短文件名**（如 `split_md5.ext`）规避长路径。
3. 重新生成清单：`python scripts/build_plantdoc_eval.py --plantdoc-root data/plantdoc --class-to-idx splits_combined/class_to_idx.json --out splits_combined/plantdoc_test.csv`
4. 重新评估并更新本文件与 `domain_gap_analysis.md`。

> 结论：**训练/内部结果数据完全一致；PlantDoc 外部集 A 本地为 2,561（源 2,572），差异已说明，不影响结论方向。**
