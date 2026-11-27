# 相关开源项目调研与借鉴说明（RELATED WORK）

本文档记录 LeafGuard 在起步阶段对同类开源项目的调研、评估与借鉴方式，供成员 A、B 共同参考。原则：**参考借鉴，不整仓 fork**。

## 1. 目的
- 降低从零起步的返工风险，避免走弯路。
- 复用成熟的数据来源与工程结构，但**实验、结果与分析必须由本项目自己完成**。
- 明确合规边界（许可证、引用、学术诚信）。

## 2. 核心决策
| 选项 | 是否采用 | 原因 |
|---|---|---|
| 整仓 fork 替换现有仓库 | 否 | 许可不明、结构冲突、学术诚信风险、已搭好的 CI/分工会被推翻 |
| 参考数据 + 代码结构 + demo 思路 | 是 | 低风险、保留自主实验、符合课程要求 |

## 3. 调研方法（可复现）
- 日期：2026-11-16
- 方式：GitHub Search API（`/search/repositories`，按 stars 排序）
- 检索词：`plant disease classification`、`plantvillage`、`plant disease detection deep learning`
- 复核：对候选仓库读取顶层目录结构（`/repos/{owner}/{repo}/contents`）

## 4. 候选项目清单（实测）
| 项目 | Stars | 语言 | 许可 | 用途 |
|---|---|---|---|---|
| spMohanty/PlantVillage-Dataset | 1037 | Python | 无 LICENSE | 数据集本体（数据来源） |
| MarkoArsenovic/DeepLearning_PlantDiseases | 246 | Python | 自定义(待核实) | SOTA CNN 评测脚本、对比表 |
| mehra-deepak/Plant-Disease-Detection | 210 | Notebook | 无 LICENSE | 经典 ML 基线 |
| soumyajit4419/Plant_AI | 197 | Jupyter | MIT | 含 Flask Web demo |
| abdullahselek/plant-disease-classification-tf | 61 | Python | MIT | TF 版分类（可选） |
| AhmadJeddi/Plant-Disease-Classification | 51 | Python | MIT | ResNet + 可复现训练/评估，结构最贴合 |

> 顶层结构观察：AhmadJeddi = `src/ experiments/ data/ docs/ requirements.txt LICENSE`（最接近本项目布局）。

## 5. 评估维度
- 许可证是否明确、可借鉴（优先 MIT）
- 框架是否与本项目一致（PyTorch + torchvision）
- 是否基于 PlantVillage、类别数是否吻合
- 是否含可复现的训练/评估/混淆矩阵
- 是否含可直接对接的推理/demo
- 代码质量与活跃度

## 6. 采用策略（分层复用）
| 层 | 采用内容 | 来源 | 注意 |
|---|---|---|---|
| 数据 | PlantVillage 38 类（+ 可选 PlantDoc） | spMohanty 官方仓 / Kaggle 镜像 | 记录来源与引用；数据不进 Git |
| 代码参考 | 数据加载、评估、混淆矩阵、可视化写法 | AhmadJeddi / MarkoArsenovic | 保留版权、注明来源 |
| 演示 | 单图预测 Web demo 思路 | soumyajit4419/Plant_AI | 与 B 的 scripts/predict.py 对接 |
| 结构与实验 | **保持本项目自有**的 A/B 分工、接口合同、E1–E4 | 本仓库 | 不照搬他人结果 |

## 7. 借鉴流程（SOP）
1. 明确需求：缺的是数据、代码、还是 demo。
2. 用 GitHub API 检索候选，按 stars 与许可证筛选。
3. 读取候选的 README、LICENSE、requirements、目录结构。
4. 判定许可证是否允许借鉴（MIT/Apache 可；无 LICENSE 只读不抄）。
5. 抽取可借鉴的最小片段/思路，落到本项目对应模块。
6. 在提交说明与本文件中注明来源。
7. 用本项目数据与脚本重新实验，结果以真实运行为准。

## 8. 合规、许可与引用
- 数据集：记录来源、版本、许可、引用到 `docs/DATASET_NOTES.md`；使用前核实官方引用。
- 代码：MIT/Apache 可借鉴，需保留版权声明并注明出处；无 LICENSE 的仓库只作阅读参考。
- 结果：**不得**把他人论文/仓库的指标当作本项目结果。

## 9. 落地映射（本项目）
| 本项目位置 | 借鉴对象 | 说明 |
|---|---|---|
| src/data/ | AhmadJeddi 的数据组织 | 仅借鉴思路，代码自写 |
| src/training/ | AhmadJeddi / MarkoArsenovic 的评估 | 指标口径对齐 INTERFACE_CONTRACT |
| scripts/predict.py | Plant_AI 的推理/demo | B 负责实现 |
| docs/model_notes.md | 评估维度 | 记录真实结果 |

## 10. 学术诚信红线
- 所有指标、混淆矩阵、训练时长来自本项目真实运行。
- 引用真实存在的论文/仓库，不伪造 DOI、作者、链接。
- 借鉴代码需在提交中说明来源。

## 11. 参考文献与复现方案
本项目采用「现代化复现」：以高被引论文为锚点复现其结论，再用 ResNet18 做现代实现，并加入自己的对比与域外评估。所有数字以本项目真实运行为准。

### 11.1 参考文献（Crossref 核实，含 DOI）
| 论文 | 年 | 出处 | DOI | 作用 |
|---|---|---|---|---|
| Mohanty, Hughes, Salathé. Using Deep Learning for Image-Based Plant Disease Detection | 2016 | Frontiers in Plant Science | 10.3389/fpls.2016.01419 | 复现锚点（PlantVillage 高准确率） |
| Ferentinos. Deep learning models for plant disease detection and diagnosis | 2018 | Computers and Electronics in Agriculture | 10.1016/j.compag.2018.01.009 | CNN 模型对比参考 |
| Too, Yujian, Njuki. A comparative study of fine-tuning deep learning models for plant disease identification | 2019 | Computers and Electronics in Agriculture | 10.1016/j.compag.2018.03.032 | 微调对比（对应 E1–E4） |
| He et al. Deep Residual Learning for Image Recognition | 2016 | CVPR | 10.1109/cvpr.2016.90 | ResNet18 backbone 出处 |

> 作者名、卷期页以论文原文为准；使用前再次核实。

### 11.2 复现策略
- 现代实现：用 ResNet18（torchvision 预训练）替代论文中的 AlexNet/GoogLeNet，复现「ImageNet 预训练 + 微调在 PlantVillage 上显著优于从零训练」的结论。
- E1–E4 对齐 Too 2019 的微调对比思路（详见 `docs/model_notes.md`）。
- 域外评估（PlantDoc）：作为候选扩展点，实验完成后再评判是否作为主要贡献。
- 数字口径：与 Mohanty 2016 对比时，说明划分/预处理/网络差异，避免直接对标数值。

### 11.3 合规
- 明确标注「复现自 Mohanty 2016（DOI）」，参考文献必须真实存在，不伪造。
- 不把论文报告的数字当作本项目结果。
- 复现与自研部分在报告中分开描述。

## 12. 后续维护
- 新增参考项目时更新第 4、6 节；新增论文时更新第 11.1 节。
- 调研日期、检索词一并记录，保证可复现。

## 13. 国内相关工作（举例，Crossref 已核实）
| 论文 | 年 | 出处 | DOI | 备注 |
|---|---|---|---|---|
| Lu, Yi, Zeng, Liu. Identification of rice diseases using deep convolutional neural networks | 2017 | Neurocomputing | 10.1016/j.neucom.2017.06.023 | 国内团队，水稻病害 CNN |
| Ma, Du, Zheng, et al. A recognition method for cucumber diseases using leaf symptom images based on deep convolutional neural network | 2018 | Computers and Electronics in Agriculture | 10.1016/j.compag.2018.08.048 | 国内团队，黄瓜病害 |
| Liu, Zhang, He, et al. Identification of Apple Leaf Diseases Based on Deep Convolutional Neural Networks | 2018 | Symmetry | 10.3390/sym10010011 | 国内团队，苹果叶病 |

> 用于体现"国内相关工作"；作者名、卷期页以论文原文为准，使用前核实。

## 14. 选型理由：为什么选择 ResNet18 + 迁移学习
- **任务特性**：132 类细粒度图像分类，同作物内病害形态相近、易混，需要较强表征；从零训练的小型 CNN 在有限数据/算力下表征不足。
- **数据特性**：PlantVillage 干净背景 + PlantWild 真实场景并存、类别不均衡；ImageNet 预训练提供通用纹理/形状特征，利于小样本与跨域。
- **工程可行性**：ResNet18 轻量（~11M 参数），8GB 显存笔记本即可训练/微调；torchvision 直接可用、复现性好。
- **可对照性**：以 E1 从零 CNN 为基线、E2 冻结、E3 微调、E4 增强，系统回答"是否用预训练、冻结还是微调、增强是否有用"。
- **文献依据**：Mohanty 2016（预训练强基线）、Too 2019（微调对比）、He 2016（ResNet），以及国内 Lu 2017 / Ma 2018 / Liu 2018 证实在作物病害识别上的有效性。
- **实验对应**：结果显示微调（E3/E4）在内部与真实场景外部均最佳，支持该选型。
