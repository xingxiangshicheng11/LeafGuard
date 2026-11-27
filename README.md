# LeafGuard
**Plant Leaf Disease Recognition and Generalization Using Transfer Learning**

双人课程项目：植物叶片病害识别、迁移学习与跨数据集（域）泛化研究。

## 分工
- 成员 A（模型与实验）：`src/models/`、`src/training/`、模型测试、`docs/model_notes.md`。
- 成员 B（数据与应用）：`src/data/`、`scripts/predict.py`、环境检查、`docs/data_audit.md`。
- 共同：接口合同、集成测试、论文、PPT 与演示。
详见 `docs/TEAM_WORKFLOW.md` 与 `docs/INTERFACE_CONTRACT.md`。

## 数据（不提交 Git）
- PlantVillage `raw/color`：38 类 / 54,305 张（干净背景）。
- PlantWild v2：115 类 / 11,488 张（真实场景，CC BY-NC-ND 4.0）。
- 合并方案 B：**132 类**（PlantVillage 38 + PlantWild 新增 94，21 个重叠合并），划分在 `splits_combined/`。
- 外部评估：PlantDoc（28 类；源 2,572、本地可用 2,564、132 类测试清单 2,561）。
- 一致性说明见 `docs/data_consistency.md`。

## 结果概览（3 seed：42/43/44，mean±std）
- 38 类（PlantVillage，内部测试集）：见 `docs/model_notes.md`。
- 132 类（方案 B）：
  - 内部 val Macro-F1：E1 0.503 / E2 0.543 / E3 0.637 / E4 0.660
  - PlantDoc 外部：E1/E2 Acc≈0.18，E3≈0.32、E4≈0.31（**微调明显改善真实场景泛化**）
- 详细指标与结论：`docs/model_notes.md`、`docs/domain_gap_analysis.md`。

## 目录
```
src/data, src/models, src/training, src/utils   核心代码
scripts/                                         训练/评估/预测/分析/演示脚本
configs/experiment.yaml                          默认配置
splits/  (38类)   splits_combined/  (132类)       固定划分 + 类别映射
docs/                                            设计、接口、审计、结果、总结等
tests/                                           单元/集成测试
```

## 环境（Windows）
```powershell
py -3.11 -m venv .venv        # 或 3.12
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# torch/torchvision 按 https://pytorch.org/get-started/locally/ 生成命令安装
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

## 常用命令
```powershell
# 132 类训练（幂等，可断点续跑；也可双击 run_training_132.bat）
python train_132_orchestrator.py

# 单图预测（含防治建议 advice）
python scripts/predict.py --image 图.jpg --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --device cuda

# 批量预测
python scripts/batch_predict.py --input 某目录 --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --num-workers 0

# 外部评估（132 类模型 + 132 类 PlantDoc 清单）
python scripts/evaluate_external.py --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --manifest splits_combined/plantdoc_test.csv --metrics-out results/metrics/external_metrics.json --num-workers 0

# 本地网页 Demo（约 34 种作物，中文病害名 + Top-5 + OOD 拒识 + 防治建议）
python scripts/serve_demo.py --checkpoint results/combined132/seed42/e3/checkpoints/resnet18_best.pt --port 8011
```

## 文档索引
- 研究设计 `docs/RESEARCH_DESIGN.md`；接口 `docs/INTERFACE_CONTRACT.md`；协作 `docs/TEAM_WORKFLOW.md`
- 数据审计 `docs/data_audit.md`；数据一致性 `docs/data_consistency.md`；PlantDoc 映射 `docs/plantdoc_mapping.md`
- 模型结果 `docs/model_notes.md`；域泛化 `docs/domain_gap_analysis.md`；项目总结（含思政）`docs/project_summary.md`
- 相关/参考工作 `docs/RELATED_WORK.md`；验收测试 `docs/testing_report.md`；预测与演示 `docs/prediction_demo.md`

> 数据、虚拟环境、模型权重、results/ 不进 Git。不得编造实验结果。
