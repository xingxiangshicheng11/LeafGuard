# 方案 B：PlantVillage + PlantWild 合并数据

## 目的

在 PlantVillage（干净背景）之外加入 PlantWild v2（wild 背景、更多物种/病害），让模型看到更多作物和杂乱背景。  
本方案**不保留合并内部 test**：原 test 全部并入 train，最终评估使用 B 持有的 PlantDoc 外部测试集。

## 数据来源

- PlantVillage：https://github.com/spMohanty/PlantVillage-Dataset
  - commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`
  - 论文：https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full
  - License：上游未声明
- PlantWild v2：https://huggingface.co/datasets/uqtwei2/PlantWild
  - 论文：https://arxiv.org/abs/2408.03120
  - 代码：https://github.com/tqwei05/MVPDR
  - License：CC BY-NC-ND 4.0（非商业、禁止演绎；需确认课程允许）

## 标签空间

- 合并后 **132 类**
- PlantVillage 38 类保留原名称
- PlantWild 115 类中：
  - 21 个与 PlantVillage 重叠的病害类合并到 PlantVillage 类名
  - 94 个新类使用 `PlantWild::` 前缀
- PlantWild 无 healthy；健康类只来自 PlantVillage（12 类）

映射文件：`splits_combined/plantwild_class_map.json`

## 划分（方案 1：无内部 test）

| Split | 图片数 | 说明 |
|---|---:|---|
| train | 55,625 | PlantVillage 46,129 + PlantWild 9,496 |
| val | 9,878 | PlantVillage 8,155 + PlantWild 1,723 |
| test | 0 | 不保留；最终评估用 PlantDoc 外部集 |

- 132 类在 train/val 中均有覆盖
- PlantWild 已去除 269 个完全重复副本
- 近重复组整组进入同一 split

最终评估：
- PlantDoc 外部测试集：2,572 张、28 类（不参与训练/调参）

## 文件

- `splits_combined/train.csv`
- `splits_combined/val.csv`
- `splits_combined/class_to_idx.json`
- `splits_combined/plantwild_class_map.json`

CSV 列固定为 `path,label,class_name`；path 相对项目根：
- `data/plantvillage/<类别名>/<图片>`
- `data/plantwild/<类别名>/<图片>`

## 复现

```powershell
python scripts/build_combined_splits.py --test-to-train
```

## 注意

- 方案 B 的 `splits_combined/` 与 38 类基线 `splits/` 并存；不要混用。
- 训练用 train，val 用于早停/调参；没有内部 test。
- PlantWild 许可为 CC BY-NC-ND 4.0，需确认课程允许再用于训练。
- PlantWild 类别分布不均衡（最少 7 张），训练时可考虑 class weight 或采样。
