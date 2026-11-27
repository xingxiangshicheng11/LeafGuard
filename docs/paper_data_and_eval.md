# 数据与评估设计（论文 / PPT 草稿）

> 状态：数据采集、审计、划分和外部评估准备已完成；**实验结果指标尚未运行，本文不填写任何编造数值**，待 A 产出 checkpoint 后补充。

## 一、论文正文草稿

### 3.1 数据集

实验主数据集为 PlantVillage 彩色原图子集（`raw/color`），来源于公开仓库 `spMohanty/PlantVillage-Dataset`，使用版本为 `master` @ `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`。本地保留 38 个类别、54,305 张图片，覆盖苹果、蓝莓、樱桃、玉米、葡萄、柑橘、桃、辣椒、马铃薯、树莓、大豆、南瓜、草莓和番茄等作物及其健康/病害类别。该数据集为实验室受控背景图像，类别间样本量不均衡（最小类 152 张，最大类 5,507 张）。

引用：Mohanty, Hughes, Salathé (2016), *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science, DOI 10.3389/fpls.2016.01419。

表 1 数据集概况

| 项目 | 数值 |
|---|---|
| 类别数 | 38 |
| 原始图片数 | 54,305 |
| 损坏图片 | 0 |
| 完全重复副本 | 21 |
| 近重复组 / 图片 | 226 组 / 481 张 |
| 划分后可用图片 | 54,284 |

### 3.2 数据审计与清洗

所有图片均通过 PIL 完整解码校验，未发现损坏文件。首先使用 SHA-256 检出类内完全重复副本 21 张，保留每类字典序第一张，其余从划分中剔除。随后对剩余 54,284 张图片计算 64-bit difference hash（dHash），以 Hamming 距离 ≤ 4 判定近重复，共得到 226 个近重复组、涉及 481 张图片，最大组 6 张。另有 33 个近重复组跨越不同类别；这些组仅记录、不在划分中强制绑定，以避免把不同病害错误合并，同时提示可能存在标签噪声或 dHash 误报。

近重复处理的目标不是删除所有相似图片，而是保证**同一叶片的多张近重复图片不会同时出现在 train、val 和 test 中**，从而降低评估偏乐观的风险。

### 3.3 分层分组划分

按类别进行 70/15/15 分层划分，随机种子固定为 42。与普通图片级随机划分不同，本文以“近重复组”为最小分配单位：同一组内的图片整体进入同一 split。组分配过程是确定性的，优先满足类别内图片比例，再检查类别覆盖。最终划分如下：

表 2 数据划分

| Split | 图片数 | 比例 | 类别数 |
|---|---:|---:|---:|
| Train | 37,978 | 70.0% | 38 |
| Val | 8,155 | 15.0% | 38 |
| Test | 8,151 | 15.0% | 38 |
| 合计 | 54,284 | 100% | 38 |

对比显示：普通图片级划分会使 109 组 / 232 张近重复图片跨越不同 split；分组划分后该数量降为 **0**。类别比例最大偏差约 0.3%，38 个类别在三个 split 中均有覆盖。类别编号按类别名称排序生成 `class_to_idx.json`，训练、评估和预测共用同一映射。

### 3.4 数据接口与预处理

`src/data/dataset.py` 读取 `path,label,class_name` 三列清单，其中 `path` 相对项目根目录。Dataset 返回 `(image, label)`：image 为 RGB、`[3,224,224]` 的归一化 float Tensor，label 为整数类别编号。训练集使用随机缩放裁剪、水平翻转和颜色抖动；验证集和测试集仅使用确定性 resize + 归一化，避免评估阶段引入随机性。DataLoader 与变换构建逻辑位于 `src/data/dataloaders.py` 与 `src/data/transforms.py`。

### 3.5 外部评估设计

外部评估使用 PlantDoc Cropped-PlantDoc 的 `train` + `test` 划分，来源版本为 `master` @ `5467f6012d78d1c446145d5f582da6096f852ae8`，共 28 类、2,572 张图片，其中健康叶 10 类 / 847 张。PlantDoc 图像来自网络与图库，背景、拍摄条件和图像质量与 PlantVillage 差异明显，适合检验跨数据集泛化能力。

由于 PlantDoc 类别体系与 PlantVillage 不完全一致，本文先建立类别映射，仅评估共同类别。28 个 PlantDoc 类别均可映射到 PlantVillage 的 38 类标签空间；映射依据类别名称，并对健康叶片类别做了样例抽查。映射文件为 `splits/plantdoc_class_map.json`，外部评估清单为 `splits/plantdoc_eval.csv`（2,572 行）。外部数据**只用于评估，不参与训练或超参数选择**。

样例抽查还发现 PlantDoc 包含教学幻灯片、网页截图和水印图库图片，且个别类内存在标签噪声（例如 Potato early blight 的样例图内标注为 *Phytophthora infestans*，Soyabean healthy 样例带有褐色斑点）。因此外部指标应结合图像质量与标签噪声解释，详见 `docs/plantdoc_spotcheck.md`。

PlantDoc 引用：Singh, Jain, Jain, Kayal, Kumawat, Batra (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020, DOI 10.1145/3371158.3371196；许可 CC BY 4.0。

### 3.6 评价指标与实验设计

内部评估指标包括 Accuracy、Macro-F1、Macro Recall、混淆矩阵和训练时间。外部评估在 PlantDoc 实际覆盖的 28 个类别上计算 Accuracy、Macro-F1、Macro Recall、混淆矩阵和各类别指标；预测到其余 10 类的样本仍计入 Accuracy 错误。所有预测分数为 softmax 值，不作为校准后的诊断概率。

表 3 实验设计

| 编号 | 模型 | 训练策略 | 增强 | 评估 |
|---|---|---|---|---|
| E1 | SmallCNN | 从零训练 | 基础增强 | PlantVillage test |
| E2 | ResNet18（ImageNet 预训练） | 冻结 backbone，仅训练分类头 | 基础增强 | PlantVillage test |
| E3 | ResNet18 | 解冻 layer4 微调 | 基础增强 | PlantVillage test |
| E4 | ResNet18 | 与 E3 相同 | 轻量增强消融 | PlantVillage test |
| E5（可选） | 最佳模型 | 不再训练 | 验证集确定性变换 | PlantDoc 28 类外部评估 |

### 3.7 局限与威胁

1. PlantVillage 为实验室受控背景，模型在真实田间场景的表现可能明显下降；PlantDoc 外部评估用于量化这一差距。
2. 近重复检测基于 dHash 与固定阈值，只能覆盖感知上高度相似的图片；阈值越低越严格，但可能漏检。
3. 33 个跨类近重复组仅记录、未强制绑定，可能包含误报或标签噪声。
4. PlantDoc 类别映射基于名称和样例抽查，健康类未逐图重新标注；映射误差会影响外部指标解释。
5. PlantDoc 含教学幻灯片、网页截图和水印图库图片，且存在类内标签噪声（如 Potato early blight、Soyabean healthy 的样例），外部评估存在额外域偏移。
6. 外部评估只覆盖 28 个共同类别，不能代表全部 38 类。
7. softmax 分数不是校准概率，不能直接作为农业诊断置信度。

### 3.8 复现性

随机种子固定为 42；数据审计、近重复分析、划分和外部评估均可通过脚本复现：

```powershell
python scripts/scan_dataset.py --hash-duplicates
python scripts/analyze_near_duplicates.py --max-distance 4
python scripts/build_splits.py --seed 42
python scripts/build_plantdoc_eval.py
python -m pytest -q
```

数据本体不进入 Git；划分清单、类别映射和 PlantDoc 评估清单提交到 `splits/`。

## 二、PPT 提纲（数据与评估部分，3–4 页）

### Slide 1：数据来源与规模
- PlantVillage `raw/color`，38 类、54,305 张
- 类别不均衡：152–5,507 张/类
- 数据审计：0 损坏、21 个完全重复副本
- 一句话：实验室受控背景，适合作为主训练集

### Slide 2：近重复与分组划分
- 64-bit dHash，Hamming ≤ 4：226 组 / 481 张近重复
- 普通划分：109 组 / 232 张跨 split
- 分组划分：**0 组跨 split**
- train/val/test = 37,978 / 8,155 / 8,151，38 类全覆盖

### Slide 3：外部评估设计
- PlantDoc `train` + `test`：28 类、2,572 张（健康叶 10 类 / 847 张），CC BY 4.0
- 28/28 类可映射到 PlantVillage 标签空间
- 仅评估、不调参；28 类上计算 Accuracy / Macro-F1 / Macro Recall
- 预期讨论：实验室 → 真实场景的泛化差距

### Slide 4：实验与指标
- E1 SmallCNN；E2 冻结 ResNet18；E3 微调 layer4；E4 增强消融；E5 外部评估
- 指标：Accuracy、Macro-F1、Macro Recall、混淆矩阵、训练时间
- 结果表待 A 的 checkpoint 运行后填写，不提前写数值

## 三、待 A 运行后填写的结果表（占位）

| 实验 | 模型/策略 | Accuracy | Macro-F1 | Macro Recall | 训练时间 | 备注 |
|---|---|---:|---:|---:|---:|---|
| E1 | SmallCNN | 待运行 | 待运行 | 待运行 | 待运行 | |
| E2 | 冻结 ResNet18 | 待运行 | 待运行 | 待运行 | 待运行 | |
| E3 | 微调 ResNet18 layer4 | 待运行 | 待运行 | 待运行 | 待运行 | |
| E4 | ResNet18 + 增强消融 | 待运行 | 待运行 | 待运行 | 待运行 | |
| E5 | 最佳模型 → PlantDoc | 待运行 | 待运行 | 待运行 | — | 外部评估 |

## 四、可引用文件

- `docs/data_audit.md`：完整审计数字
- `docs/plantdoc_mapping.md`：PlantDoc 映射与限制
- `docs/INTERFACE_CONTRACT.md`：数据与模型接口
- `scripts/scan_dataset.py`、`scripts/analyze_near_duplicates.py`、`scripts/build_splits.py`
- `scripts/build_plantdoc_eval.py`、`scripts/evaluate_external.py`
