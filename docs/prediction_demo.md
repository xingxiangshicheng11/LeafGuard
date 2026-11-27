# 预测与演示使用说明

预测依赖 A 产出的 checkpoint（字段见 `docs/INTERFACE_CONTRACT.md`）。  
所有输出中的分数都是 **softmax 值，不是校准后的诊断概率**。

## 0. 本地网页 Demo（零依赖）
不装任何 Web 框架，启动一个本地网页，浏览器里上传/拖拽叶子图片即时看预测：

```powershell
python scripts/serve_demo.py --checkpoint results/seed42/e3/checkpoints/resnet18_best.pt --port 8011
```

然后浏览器打开 `http://127.0.0.1:8011/`，把叶子图片拖进去，即可看到预测类别与 Top-5 置信度条形图。

- `--device auto|cpu|cuda`：推理设备；`--port`：端口
- 类别默认以中文显示（中英对照见 `splits/class_to_zh.json`，可用 `--zh-map` 指定其它映射）
- 分数为 softmax 值，不是校准后的诊断概率；仅用于演示
- **OOD 拒识**：当最大置信度偏低、或 energy 分数偏高（疑似不属于支持的 38 类）时，显示“无法可靠识别”，不给具体类别。阈值需先在验证集标定：
  ```powershell
  python scripts/calibrate_ood.py --checkpoint results/seed42/e3/checkpoints/resnet18_best.pt
  ```
  生成 `<ckpt 上级>/metrics/ood_thresholds.json`，`serve_demo.py` 会自动读取（也可用 `--ood-thresholds` 指定）。
- **防治建议**：识别（非拒识）后出现 **「查看防治建议」按钮**，点开显示该病害的参考防治措施/预防与来源（`assets/disease_advice_zh.json`）；健康类显示养护建议；被拒识时不显示建议。建议仅供学习参考，具体以当地农技/植保与登记药剂标签为准。
- 命令行 `scripts/predict.py` 的 JSON 输出也包含 `advice` 字段。

## 0.1 132 类（方案B）Demo
训练集为 PlantVillage + PlantWild 合并的 132 类。用对应 checkpoint 与 132 类 OOD 阈值：

```powershell
python scripts/serve_demo.py --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --port 8011
```

或直接双击项目根目录的 **`run_demo_132.bat`**（自动打开浏览器）。

- 132 类 OOD 阈值需用合并验证集标定：
  ```powershell
  python scripts/calibrate_ood.py --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --split splits_combined/val.csv
  ```
- 支持约 34 种作物（含小麦、柑橘、香蕉、咖啡、水稻、烟草、黄瓜、茄子等），比 38 类版覆盖更广。

## 1. 单图预测

```powershell
python scripts/predict.py --image data/plantvillage/Tomato___healthy/xxxx.jpg --checkpoint results/checkpoints/<model>.pth
```

输出 JSON，包含：

- 预测类别
- Top-3 类别与 softmax 分数
- 使用的 checkpoint / 类别映射来源

## 2. 批量预测（演示用）

```powershell
python scripts/batch_predict.py --input data/plantdoc/Tomato\ leaf --checkpoint results/checkpoints/<model>.pth
```

输出：

- `results/predictions/batch_predictions.csv`
  - `path,pred_class,pred_label,confidence,top_k_classes,top_k_scores,error`
- `results/predictions/batch_summary.json`
  - 图片总数、成功/失败数、预测类别计数、失败文件列表

常用参数：

- `--input`：图片文件或目录
- `--recursive` / `--no-recursive`：是否递归子目录（默认递归）
- `--limit N`：只预测前 N 张，方便演示
- `--top-k`：默认 3
- `--device auto|cpu|cuda`
- `--batch-size`、`--num-workers`

示例（只跑 8 张）：

```powershell
python scripts/batch_predict.py --input data/plantdoc --limit 8 --checkpoint results/checkpoints/<model>.pth
```

## 3. 演示建议

1. A 训练完成后，选 5–10 张不同病害图片放在一个文件夹
2. 运行 `batch_predict.py`，打开 `batch_predictions.csv` 展示预测类别和置信度
3. 需要讲解时，用 `predict.py` 对单张图输出 Top-3

## 4. 注意事项

- 没有 checkpoint 时不要编造预测结果；可用临时 checkpoint 只验证流程，数值无意义
- 外部数据（PlantDoc）只用于评估/演示，不参与训练或调参
- 预测结果不能作为真实农业诊断结论
