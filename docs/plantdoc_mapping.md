# PlantDoc 外部评估与类别映射

## 数据来源

- 数据集：PlantDoc Cropped-PlantDoc（分类任务版本）
- 来源：https://github.com/pratikkayal/PlantDoc-Dataset
- 版本：`master` @ `5467f6012d78d1c446145d5f582da6096f852ae8`（上游提交日期 2021-05-02）
- 获取日期：2026-11-20
- 使用范围：仅 `test` 划分，作为 PlantVillage 训练模型的**外部评估集**
- 本地相对路径：`data/plantdoc/<PlantDoc 类别名>/<图片>`
- 许可：CC BY 4.0
- 引用：Singh, Jain, Jain, Kayal, Kumawat, Batra (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020, DOI 10.1145/3371158.3371196
- 外部评估集**不参与训练或调参**。

## 审计结果

- 类别数：27
- 图片数：236
- 损坏图片：0
- 重复图片：0
- 文件名处理：5 个原文件名包含 `?`（Windows 非法字符），已用 `_` 替换并记录在 `results/plantdoc_eval_report.json`

## 类别映射

映射文件：`splits/plantdoc_class_map.json`  
外部评估清单：`splits/plantdoc_eval.csv`（列：`path,label,class_name`，label 使用 PlantVillage 的 `class_to_idx.json`）

| PlantDoc 类别 | 数量 | PlantVillage 目标类别 |
|---|---:|---|
| Apple Scab Leaf | 10 | `Apple___Apple_scab` |
| Apple leaf | 9 | `Apple___healthy` |
| Apple rust leaf | 10 | `Apple___Cedar_apple_rust` |
| Bell_pepper leaf | 8 | `Pepper,_bell___healthy` |
| Bell_pepper leaf spot | 9 | `Pepper,_bell___Bacterial_spot` |
| Blueberry leaf | 11 | `Blueberry___healthy` |
| Cherry leaf | 10 | `Cherry_(including_sour)___healthy` |
| Corn Gray leaf spot | 4 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| Corn leaf blight | 12 | `Corn_(maize)___Northern_Leaf_Blight` |
| Corn rust leaf | 10 | `Corn_(maize)___Common_rust_` |
| Peach leaf | 9 | `Peach___healthy` |
| Potato leaf early blight | 8 | `Potato___Early_blight` |
| Potato leaf late blight | 8 | `Potato___Late_blight` |
| Raspberry leaf | 7 | `Raspberry___healthy` |
| Soyabean leaf | 8 | `Soybean___healthy` |
| Squash Powdery mildew leaf | 6 | `Squash___Powdery_mildew` |
| Strawberry leaf | 8 | `Strawberry___healthy` |
| Tomato Early blight leaf | 9 | `Tomato___Early_blight` |
| Tomato Septoria leaf spot | 11 | `Tomato___Septoria_leaf_spot` |
| Tomato leaf | 8 | `Tomato___healthy` |
| Tomato leaf bacterial spot | 9 | `Tomato___Bacterial_spot` |
| Tomato leaf late blight | 10 | `Tomato___Late_blight` |
| Tomato leaf mosaic virus | 10 | `Tomato___Tomato_mosaic_virus` |
| Tomato leaf yellow virus | 6 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| Tomato mold leaf | 6 | `Tomato___Leaf_Mold` |
| grape leaf | 12 | `Grape___healthy` |
| grape leaf black rot | 8 | `Grape___Black_rot` |

## 映射说明与限制

- 映射基于类别名称，并对 Apple/Peach/Cherry/grape/Tomato/Bell pepper 的“healthy leaf”样例做了人工抽查。
- healthy 类按名称与图片中无明显病斑推断，**未逐图重新标注**。
- PlantDoc 图片来自网络/图库，背景、构图与图像质量与 PlantVillage 差异较大；外部评估结果应结合这些差异解释。
- 27 个 PlantDoc 类别全部能映射到 PlantVillage 的 38 类标签空间；无未映射类别。
- 若后续需要更严格评估，可只保留映射置信度高的类别，或人工复核全部图片。

## 复现命令

```powershell
python scripts/build_plantdoc_eval.py
```

输出：

- `splits/plantdoc_eval.csv`
- `results/plantdoc_eval_report.json`（Git 忽略）
