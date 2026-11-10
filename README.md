# LeafGuard
**Plant Leaf Disease Recognition and Generalization Using Transfer Learning**

双人课程项目：植物叶片病害识别、迁移学习和跨数据集泛化研究。

## 分工
- 成员 A：模型与实验，负责 `src/models/`、`src/training/`、`docs/model_notes.md`。
- 成员 B：数据与应用，负责 `src/data/`、`scripts/predict.py`、`docs/data_audit.md`。
- 共同：接口合同、集成测试、论文、PPT 和演示。

## 环境
建议 Windows 11 使用 Python 3.11/3.12 虚拟环境。PyTorch CUDA 安装命令以 https://pytorch.org/get-started/locally/ 当前选择器为准。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
先按官网安装匹配的 torch/torchvision，再运行 `pip install -r requirements.txt`。

## 开发顺序
1. 双方确认 `docs/INTERFACE_CONTRACT.md`。
2. B 完成数据审计、类别映射和固定划分。
3. A 使用统一数据接口完成 CNN / ResNet18 训练。
4. B 调用统一模型接口完成单图预测。
5. 共同集成、复核实验、写论文和 PPT。

数据集、虚拟环境和模型权重不提交 Git。不得编造实验结果。
