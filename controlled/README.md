# 受控训练实验：嵌入路径的因果检验（v4，进行中）

v3（Study 2）的证据来自档案中的观察数据。本目录自己训练小模型，**只改变嵌入（d_embed、div_val），解码器完全固定**，直接检验冻结的 v3 对嵌入的预测。课题 PDF 扩展方向 1 指出，受控训练比档案中的异构配置证据更强。新训练的结果是独立的新标签，不使用 final split。

## 设置

- 数据：WikiText-103 raw（Hugging Face `Salesforce/wikitext`），GPT-2 BPE 分词（50257 个 token，与档案 n_token = 50264 同一词表）。训练集 117,920,140 token，验证集 247,289，测试集 283,287。`prepare_data.py` 校验每个文件的 SHA-256。
- 模型（`txl.py`）：按档案的约定实现 Transformer-XL。相对位置注意力加记忆，Post-LN，绑定权重的自适应嵌入和自适应 softmax，头投影不绑定、尾投影绑定，cutoffs [19997, 39997]，权重初始化 N(0, 0.02)、投影初始化 N(0, 0.01)，dropout 0.1。注意力和块结构参考 NSC 作者的训练脚本（MIT 许可）。
- 训练（`train.py`）：Adam，学习率 1e-3，5% 线性预热后余弦衰减，梯度裁剪 0.25，bf16，32 条并行流 × 192 token，记忆长度 192。在完整验证集上带记忆计算 PPL。
- 硬件：NVIDIA GeForce RTX 5070 Ti Laptop GPU（12 GB），PyTorch 2.9.1 + CUDA 12.8。

```powershell
pip install -r controlled/requirements.txt   # torch 需要从 CUDA 源安装，见文件内注释
.\.venv\Scripts\python.exe controlled\prepare_data.py
.\.venv\Scripts\python.exe controlled\train.py --name pilot_d256_e256 --d_model 256 --n_layer 8 --n_head 8 --d_inner 1024 --d_embed 256 --div_val 1 --tokens 2e7 --seed 0
```

## 1. 计时试跑（2026-11-03）

d_model = 256，8 层，8 头，FFN 1024（解码器 6,838,272 个参数），训练 2000 万 token（约为训练集的 1/6）。

| 运行 | d_embed | div_val | 种子 | 总参数 | 验证 PPL | 训练时间 |
|---|---:|---:|---:|---:|---:|---:|
| pilot_d256_e256 | 256 | 1 | 0 | 19.75 M | 76.88 | 370 s |
| pilot_d256_e256_s1 | 256 | 1 | 1 | 19.75 M | 78.11 | 369 s |
| pilot_d256_e64 | 64 | 1 | 0 | 10.14 M | 105.98 | 340 s |

- 吞吐约 5.4 万 token/s，显存峰值 4 GB。
- 种子差异：两个种子的 log PPL 相差 0.016（约 1.6%），与档案上从数据反推的噪声上界（约 2%）同一量级。只有两个种子，这只是粗略估计。
- 嵌入效应：d_embed 从 64 增加到 256，log PPL 降低 0.32，约为种子差异的 20 倍。在这个训练量下，嵌入效应可以被清楚地测到。
- 局限：短训练下，窄嵌入的劣势可能被放大（低频 token 的嵌入训练不足）。正式实验中需要用更长的训练检查效应是否保持。

## 2. v3 对受控面板的预测（不读任何训练结果）

解码器固定为上面的设置时，NSC = 2151.1，对所有嵌入变体**完全相同**（NSC 不看嵌入，无法区分它们）。v3 的嵌入路径容量 E：

| d_embed | div_val = 1 | div_val = 2 | div_val = 4 |
|---:|---:|---:|---:|
| 64 | 273.1 | 413.9 | 315.2 |
| 128 | 439.6 | 656.3 | 500.5 |
| 256 | 780.1 | 947.5 | 724.6 |
| 512 | 713.0 | 1205.0 | 947.4 |

可以区分 v3 与"数参数"的地方：div_val = 1 时 v3 预测 d_embed = 512 **不比 256 更好**（秩瓶颈下饱和），而总参数量预测 512 更好；v3 还预测同样 d_embed 下 div_val = 2 优于 div_val = 1。
