# 成员 B：数据审计记录

- 数据集来源：PlantVillage-Dataset（`spMohanty/PlantVillage-Dataset`），使用 `raw/color` 彩色原图
- 下载日期/版本：2026-11-11，`master` @ `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`
- 本地相对路径：`data/plantvillage`
- 图片总数：54305
- 类别总数：38
- 损坏图片数：0
- 重复文件数：21（类内完全重复副本；跨类重复 0 组）
- 类别映射文件：`splits/class_to_idx.json`
- train/val/test 比例与样本数：70/15/15 → 37996 / 8144 / 8144（去除 21 个重复副本后，可用图片 54284）
- 跨 split 重复检查：train/val、train/test、val/test 交集均为 0
- 排除样本及原因：21 个类内完全重复副本（SHA-256 相同，保留每类字典序第一个）；无损坏图片
- 引用：Mohanty, Hughes, Salathé (2016), *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science, DOI 10.3389/fpls.2016.01419
- 许可说明：上游 GitHub 仓库未声明 License（API 返回 `license: null`），使用与再分发前需与项目要求核对

## 各类别样本量（扫描有效图片数）

| 类别 | 图片数 |
|---|---|
| Apple___Apple_scab | 630 |
| Apple___Black_rot | 621 |
| Apple___Cedar_apple_rust | 275 |
| Apple___healthy | 1645 |
| Blueberry___healthy | 1502 |
| Cherry_(including_sour)___healthy | 854 |
| Cherry_(including_sour)___Powdery_mildew | 1052 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Corn_(maize)___Common_rust_ | 1192 |
| Corn_(maize)___healthy | 1162 |
| Corn_(maize)___Northern_Leaf_Blight | 985 |
| Grape___Black_rot | 1180 |
| Grape___Esca_(Black_Measles) | 1383 |
| Grape___healthy | 423 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1076 |
| Orange___Haunglongbing_(Citrus_greening) | 5507 |
| Peach___Bacterial_spot | 2297 |
| Peach___healthy | 360 |
| Pepper,_bell___Bacterial_spot | 997 |
| Pepper,_bell___healthy | 1478 |
| Potato___Early_blight | 1000 |
| Potato___healthy | 152 |
| Potato___Late_blight | 1000 |
| Raspberry___healthy | 371 |
| Soybean___healthy | 5090 |
| Squash___Powdery_mildew | 1835 |
| Strawberry___healthy | 456 |
| Strawberry___Leaf_scorch | 1109 |
| Tomato___Bacterial_spot | 2127 |
| Tomato___Early_blight | 1000 |
| Tomato___healthy | 1591 |
| Tomato___Late_blight | 1909 |
| Tomato___Leaf_Mold | 952 |
| Tomato___Septoria_leaf_spot | 1771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1676 |
| Tomato___Target_Spot | 1404 |
| Tomato___Tomato_mosaic_virus | 373 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5357 |

## 复现命令

```powershell
python scripts/scan_dataset.py --hash-duplicates
python scripts/build_splits.py --seed 42
```

- 扫描报告：`results/data_scan.json`（Git 忽略）
- 划分结果：`splits/train.csv`、`splits/val.csv`、`splits/test.csv`、`splits/class_to_idx.json`（已提交）

## PlantDoc 外部评估集审计

- 来源版本：`pratikkayal/PlantDoc-Dataset` `master` @ `5467f6012d78d1c446145d5f582da6096f852ae8`
- 获取日期：2026-11-20
- 使用划分：仅 `test`
- 类别数：27
- 图片数：236
- 损坏图片：0
- 重复图片：0
- 文件名处理：5 个含 `?` 的文件已重命名（`?` → `_`），详见 `docs/plantdoc_mapping.md`
- 类别映射：`splits/plantdoc_class_map.json`（27/27 类可映射到 PlantVillage 38 类）
- 外部评估清单：`splits/plantdoc_eval.csv`
- 用途限制：仅外部评估，不参与训练或调参

复现：

```powershell
python scripts/scan_dataset.py --data-root data/plantdoc
python scripts/build_plantdoc_eval.py
```
