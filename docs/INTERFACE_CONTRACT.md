# 模块接口合同

接口变更前双方沟通并同步修改此文件。

## 类别映射
`splits/class_to_idx.json` 保存类别名到整数编号的映射。由 B 生成；训练、评估、预测必须共用。

## 划分 CSV
`train.csv`、`val.csv`、`test.csv` 必须含 `path,label,class_name` 三列。path 相对项目根目录；同一图片不能跨 split。

## Dataset（B 提供）
```python
image, label = dataset[index]
```
image 为 RGB float Tensor `[3,224,224]`，已归一化；label 为整数类别编号。

## Model（A 提供）
```python
logits = model(images)
```
输入 `[B,3,224,224]`，输出 `[B,num_classes]` 原始 logits，不在模型内 softmax。

## Checkpoint
至少保存 `model_state_dict`、`num_classes`、`class_to_idx`、`model_name`、`input_size`、`config`、`best_val_macro_f1`。

## Prediction（B 实现）
读取 checkpoint 和类别映射，输出预测类别、Top-3 类别与 softmax 分数。分数不是校准后的诊断概率。

## Metrics
JSON 至少含 accuracy、macro_f1、macro_recall、training_seconds、class_names。没有的指标不得填造。
