# PlantDoc 外部评估与类别映射

## 数据来源

- 数据集：PlantDoc Cropped-PlantDoc（分类任务版本）
- 来源：https://github.com/pratikkayal/PlantDoc-Dataset
- 版本：`master` @ `5467f6012d78d1c446145d5f582da6096f852ae8`（上游提交日期 2021-05-02）
- 获取日期：2026-11-20（test）、2026-11-24（train）
- 使用范围：`train` + `test` 两个划分，作为 PlantVillage 训练模型的**外部评估集**
- 本地相对路径：
  - `data/plantdoc/<PlantDoc 类别名>/<图片>`（test）
  - `data/plantdoc_train/<PlantDoc 类别名>/<图片>`（train）
- 许可：CC BY 4.0
- 引用：Singh, Jain, Jain, Kayal, Kumawat, Batra (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020, DOI 10.1145/3371158.3371196
- 外部评估集**不参与训练或调参**。

## 审计结果

- 类别数：28
- 图片数：2,572（test 236 + train 2,336）
- 健康叶：10 类 / 847 张
- 病害叶：18 类 / 1,725 张
- 损坏图片：0
- 类内重复：0；跨类完全相同图片 1 组（Potato early blight / late blight 各 1 张，保留并记录）
- 文件名处理：
  - 82 个 train + 5 个 test 文件名包含 `?`，已用 `_` 替换
  - 7 个过长 train 文件名已缩短，避免 Windows 路径长度限制
- 记录：`results/plantdoc_eval_report.json`、`results/plantdoc_train_audit.json`（Git 忽略）

## 类别映射

映射文件：`splits/plantdoc_class_map.json`  
外部评估清单：`splits/plantdoc_eval.csv`（列：`path,label,class_name`，label 使用 PlantVillage 的 `class_to_idx.json`）

| PlantDoc 类别 | 数量 | PlantVillage 目标类别 |
|---|---:|---|
| Apple Scab Leaf | 93 | `Apple___Apple_scab` |
| Apple leaf | 91 | `Apple___healthy` |
| Apple rust leaf | 88 | `Apple___Cedar_apple_rust` |
| Bell_pepper leaf | 61 | `Pepper,_bell___healthy` |
| Bell_pepper leaf spot | 71 | `Pepper,_bell___Bacterial_spot` |
| Blueberry leaf | 115 | `Blueberry___healthy` |
| Cherry leaf | 57 | `Cherry_(including_sour)___healthy` |
| Corn Gray leaf spot | 68 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| Corn leaf blight | 191 | `Corn_(maize)___Northern_Leaf_Blight` |
| Corn rust leaf | 116 | `Corn_(maize)___Common_rust_` |
| Peach leaf | 111 | `Peach___healthy` |
| Potato leaf early blight | 116 | `Potato___Early_blight` |
| Potato leaf late blight | 105 | `Potato___Late_blight` |
| Raspberry leaf | 119 | `Raspberry___healthy` |
| Soyabean leaf | 65 | `Soybean___healthy` |
| Squash Powdery mildew leaf | 130 | `Squash___Powdery_mildew` |
| Strawberry leaf | 96 | `Strawberry___healthy` |
| Tomato Early blight leaf | 88 | `Tomato___Early_blight` |
| Tomato Septoria leaf spot | 151 | `Tomato___Septoria_leaf_spot` |
| Tomato leaf | 63 | `Tomato___healthy` |
| Tomato leaf bacterial spot | 110 | `Tomato___Bacterial_spot` |
| Tomato leaf late blight | 111 | `Tomato___Late_blight` |
| Tomato leaf mosaic virus | 54 | `Tomato___Tomato_mosaic_virus` |
| Tomato leaf yellow virus | 76 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| Tomato mold leaf | 91 | `Tomato___Leaf_Mold` |
| Tomato two spotted spider mites leaf | 2 | `Tomato___Spider_mites Two-spotted_spider_mite` |
| grape leaf | 69 | `Grape___healthy` |
| grape leaf black rot | 64 | `Grape___Black_rot` |

## 映射说明与限制

- 映射基于类别名称，并对 Apple/Peach/Cherry/grape/Tomato/Bell pepper 的“healthy leaf”样例做了人工抽查。
- healthy 类按名称与图片中无明显病斑推断，**未逐图重新标注**。
- PlantDoc 图片来自网络/图库，背景、构图与图像质量与 PlantVillage 差异较大；外部评估结果应结合这些差异解释。
- 28 个 PlantDoc 类别全部能映射到 PlantVillage 的 38 类标签空间；无未映射类别。
- 跨类完全相同图片 1 组仅记录、不删除，避免人为改变类别分布。
- 逐类样例抽查记录（含标签噪声与图像质量问题）：`docs/plantdoc_spotcheck.md`。

## 复现命令

```powershell
python scripts/build_plantdoc_eval.py
```

默认扫描 `data/plantdoc` 与 `data/plantdoc_train` 两个根目录。

输出：

- `splits/plantdoc_eval.csv`
- `results/plantdoc_eval_report.json`（Git 忽略）

## 外部评估运行方式（需要 A 的 checkpoint）

```powershell
python scripts/evaluate_external.py --checkpoint results/checkpoints/<model>.pth
```

输出：

- `results/metrics/external_metrics.json`：Accuracy、Macro-F1、Macro Recall、混淆矩阵、各类别指标
- `results/metrics/external_predictions.csv`：逐图真实/预测类别、置信度、Top-3

说明：

- Macro-F1 / Macro Recall 只在 PlantDoc 实际覆盖的 28 个类别上计算；预测到其余 10 类会作为错误计入 Accuracy。
- 置信度为 softmax 分数，不是校准后的诊断概率。
- 外部数据不参与训练或调参；指标必须由真实 checkpoint 运行得到，不能提前填写。
