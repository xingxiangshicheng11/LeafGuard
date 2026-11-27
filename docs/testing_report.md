# 端到端验收测试报告

仅记录真实运行结果（未运行的项标注"未测"）。测试对象：方案B 132 类模型与相关工具。
用到的 checkpoint：`results/combined132/seed42/e3/checkpoints/resnet18_best.pt`（ResNet18 微调，132 类）。

## 1. 环境自检（`scripts/check_environment.py`）
- Python 3.13.14，PyTorch 2.14.1+cu130，torchvision 0.29.1+cu130
- CUDA available: **True**，GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- data/plantvillage、data/plantdoc 均存在 ✅

## 2. 单元/回归测试（`pytest -q`）
- **39 passed** ✅

## 3. 单图预测 CLI（`scripts/predict.py`）
命令：`predict.py --image 落木/柑橘病害.jpg --checkpoint <e3> --device cuda`
- prediction: `PlantWild::citrus canker`（柑橘-溃疡病），score≈0.980
- 输出含 `top_k` 与 `advice`（摘要/防治措施/预防/来源）✅

## 4. 批量预测 CLI（`scripts/batch_predict.py`）
命令：`batch_predict.py --input 落木 --checkpoint <e3> --num-workers 0`
- images: 4，ok: 4，failed: 0
- 预测分布：citrus canker / soybean frog eye leaf spot / wheat powdery mildew / Tomato early blight 各 1
- 产出：`results/predictions/batch_predictions.csv`、`batch_summary.json` ✅

## 5. 外部评估 CLI（`scripts/evaluate_external.py`）
命令：`evaluate_external.py --checkpoint <e3> --manifest splits_combined/plantdoc_test.csv --num-workers 0`
- samples: 2561，evaluated classes: 28
- **Accuracy 0.3335，Macro-F1 0.3264，Macro-Recall 0.3022**
- 与 `docs/model_notes.md` 中 seed42 E3 外部指标一致 ✅（口径可复现）
- 产出：`results/acceptance/external_metrics.json`、`external_predictions.csv`

## 6. 外部分析 CLI（`scripts/analyze_external_results.py`）
- 正常输出 top 易混类别与最弱类别，并生成 28 类混淆图到 `results/figures/combined132/plantdoc/` ✅
- 示例（E1 seed42）：玉米北方叶枯→玉米灰斑(65)、番茄斑枯→番茄细菌性斑点(25) 等

## 7. 网页 Demo HTTP 接口
以进程内启动 `ThreadingHTTPServer` 验证（等价于浏览器所用接口）：
- `GET /health` → `ok` ✅
- `GET /` → 200，页面含「查看防治建议」按钮与"共 132 类" ✅
- `POST /api/predict` → `rejected=False`，预测=番茄-早疫病，`advice` 存在且含"来源" ✅

> 说明：真实浏览器拖拽上传未由自动化执行；建议人工双击 `run_demo_132.bat` 再拖图确认（UI 所依赖的接口已逐项验证通过）。

## 8. 结论
- 核心链路（环境 / 单元测试 / 单图 / 批量 / 外部评估 / 外部分析 / Demo 接口）**全部通过**。
- 已知待办（不在本次验收范围）：
  - `scripts/analyze_results.py` 在 132 布局下会把 `external_metrics.json` 误当作一个模型条目（若直接对 `results/combined132/seedN` 运行需先加过滤）；外部结果请用 `analyze_external_results.py`。
  - 自建"随手拍"带标签集量化评估未做（可选，需人工标注图片）。
