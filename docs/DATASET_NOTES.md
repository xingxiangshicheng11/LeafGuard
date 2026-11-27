# 数据集来源

## PlantVillage（主数据集，已下载）

- 来源：https://github.com/spMohanty/PlantVillage-Dataset
- 使用子集：`raw/color`
- 版本：`master` @ `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`（上游提交日期 2026-02-05）
- 下载日期：2026-11-11
- 本地相对路径：`data/plantvillage/<类别名>/<图片>`
- 类别数：38；图片数：54305（详细统计见 `docs/data_audit.md`）
- 许可证：上游仓库未声明 License（GitHub API 返回 `license: null`），使用与再分发前需按课程要求核对
- 引用：Mohanty, Hughes, Salathé (2016), *Using deep learning for image-based plant disease detection*, Frontiers in Plant Science, DOI 10.3389/fpls.2016.01419

## PlantDoc（外部评估集，已下载 test 划分）

- 来源：https://github.com/pratikkayal/PlantDoc-Dataset
- 版本：`master` @ `5467f6012d78d1c446145d5f582da6096f852ae8`（上游提交日期 2021-05-02）
- 获取日期：2026-11-20
- 使用范围：仅 `test` 划分，作为外部评估；不参与训练或调参
- 本地相对路径：`data/plantdoc/<类别名>/<图片>`
- 类别数：27；图片数：236（详见 `docs/plantdoc_mapping.md`）
- 许可：CC BY 4.0
- 引用：Singh, Jain, Jain, Kayal, Kumawat, Batra (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020, DOI 10.1145/3371158.3371196
- 类映射：`splits/plantdoc_class_map.json`；外部评估清单：`splits/plantdoc_eval.csv`

## 获取方式

- PlantVillage（主数据集）：优先官方仓库；如用 Kaggle 镜像，需核对类别数（目标 38 类）与目录结构。
- PlantDoc（可选外部评估）：官方仓库；仅用于 E5 外部评估。

## 目录约定

- PlantVillage 放在 `data/plantvillage/<类别名>/<图片>`。
- PlantDoc 放在 `data/plantdoc/<类别名>/<图片>`。
- 两人目录结构必须一致（划分 CSV 中的 path 相对项目根）。

## 引用与许可

- 使用前核实数据集的官方引用信息（作者、年份、许可）并记录到本文件；不确定的信息不要臆造。
- 数据集本体不提交 Git；划分清单与类别映射放在 `splits/`（提交）。

## 相关文档

- 参考项目与借鉴方式：`docs/RELATED_WORK.md`
- 数据审计记录：`docs/data_audit.md`
