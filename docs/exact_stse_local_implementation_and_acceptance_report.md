# Exact-STSE 本地实现与验收报告

## 1. 实现范围

已按 `Exact-STSE_空间坐标消融_架构与实验设计.md` 实现隔离的论文风格复现分支：

- `Exact-STSE`：`degree + coord + signed-neighcoord + community embedding + delta_degree`，输入维度 24；
- `Exact-STSE-NoCoord`：`degree + community embedding + delta_degree`，输入维度 18；
- 输入 LayerNorm、线性投影、单残差 GELU-FFN、输出 LayerNorm；
- 节点均值池化，以及明确标记为复现假设的时间窗口均值池化；
- `64 → 64 → 32 → 2` ReLU/Dropout 分类头；
- 类别加权、AdamW、ReduceLROnPlateau、梯度裁剪、早停、严格 checkpoint 和断点续训；
- 输入审计、训练、评估、16 样本过拟合、逐层方差诊断和三 seed 配对汇总脚本。

该分支使用独立 Dataset；通用关键子图模型仍不读取坐标，也未重新引入社区编号 embedding。

## 2. 数据审计

冻结协议：`configs/data_protocol_exact_stse_coords.json`。

| 项目 | 结果 |
|---|---:|
| 总样本数 | 307 |
| Train / Validation / Test | 215 / 46 / 46 |
| Train 类别 0 / 1 | 139 / 76 |
| Validation 类别 0 / 1 | 30 / 16 |
| Test 类别 0 / 1 | 30 / 16 |
| 时间窗口范围 | 15–52 |
| 每窗口节点数 | 固定 116 |
| 坐标数组字节哈希数 | 1 |
| 最大社区编号 | 14 |
| 数据异常 | 0 |

307 个样本的全部 9,627 个时间窗口使用同一套有限、非全零的 `116×3` 坐标。

## 3. 程序验收

- 两分支真实样本 forward/backward：通过；
- 24/18 维输入检查：通过；
- 第一个窗口 `delta_degree=0`：通过；
- 带符号邻居坐标公式：通过；
- NoCoord 不读取坐标：通过；
- 节点一致置换不改变预测：通过；
- list-based 变长时间序列与节点 mask：通过；
- 类别加权在 `batch_size=1` 时不被抵消：已在训练器中显式处理；
- checkpoint 配置、协议哈希、resume：通过；
- CLI smoke、完整验证集评估和逐层诊断：通过；
- Exact-STSE 新增单元测试：9/9 通过；
- 项目回归（排除既有 Windows 本地挂起项）：233 项通过，1 项因 Windows 无符号链接权限跳过。

既有 `test_crossfit_fold_protocol` 会在本地 `outputs/` 临时目录操作中挂起，本次改动未触及该模块。

## 4. 16 样本过拟合验收

两分支使用相同的固定 8+8 样本、seed 42、200 epoch、dropout 0、weight decay 0。

| 模型 | 最佳 Epoch | Train Accuracy | Train BA | Train AUROC | Train Loss | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Exact-STSE | 200 | 1.0000 | 1.0000 | 1.0000 | 0.000301 | 通过 |
| Exact-STSE-NoCoord | 196 | 0.7500 | 0.7500 | 0.671875 | 0.679161 | 未通过 |

NoCoord 的公式、输入维度、社区索引、标签对应、梯度、变长 batching 和分类头均已通过独立检查，因此不能将失败解释为明显的数据管线或断梯度错误。

## 5. 坍缩定位

在 64 个训练样本上的诊断：

| 表示 | Coord 方差 | NoCoord 方差 |
|---|---:|---:|
| 编码节点 | 1.029574 | 1.002034 |
| 窗口均值 | 0.101174 | 0.008193 |
| 样本时间均值 | 0.013355 | 0.000365 |
| Logits | 19.63097 | 0.001519 |

NoCoord 的节点编码并未坍缩，但节点均值与时间均值连续压缩了样本间差异；其样本表示平均余弦相似度达到 0.995761。坐标版保留了更明显的窗口和样本差异，并能完成记忆。

这只是最小过拟合证据，不能直接证明坐标提高验证集或测试集泛化。正式结论必须来自固定划分下的 seed 42、43、44 配对实验。

## 6. 本地产物

本地验收产物位于 `.local_validation/exact_stse_*`，已由 `.gitignore` 排除，不应提交到 Git。
