# 无坐标、社区结构化完整短期分支

## 1. 实现目标

本分支复现原论文短期分支的完整处理链，同时移除两类不适合当前数据的输入：

- 不使用节点空间坐标及邻居坐标；
- 不使用 `community_id` 的 `nn.Embedding`。

社区编号仅在单个时间窗口内用于判断节点是否同属一个社区。不同窗口、样本或数据集之间不比较社区编号。

完整流程为：

```text
带符号动态图序列
  → 无坐标、社区结构化节点特征
  → 逐窗口节点编码与图池化
  → CLS + 位置编码 + Transformer
  → 可训练原型记忆读取
  → 序列统计特征
  → 二分类头
```

## 2. 输入约束

- 支持每个样本不同的窗口数 \(M_b\)；
- 支持每个窗口不同的节点数 \(N_b^{(m)}\)；
- 不截断原图；
- 使用 list-based batching，只在 Transformer 前对窗口表示临时 padding；
- `time_mask` 保证 Transformer 忽略 padding；
- 正边和负边都是有效连接，边存在条件统一为
  \[
  |A_{ij}|>\tau_{\text{edge}}.
  \]

阈值来自冻结的数据协议。

## 3. 社区结构化节点特征

节点 \(i\) 的 15 维特征为：

1. 绝对连接强度 \(d_i=\sum_j|A_{ij}|\)；
2. 正连接强度 \(d_i^+=\sum_j\max(A_{ij},0)\)；
3. 负连接幅值 \(d_i^-=\sum_j|\min(A_{ij},0)|\)；
4. 正连接比例；
5. 负连接比例；
6. 绝对连接强度时间差分；
7. 正连接强度时间差分；
8. 负连接幅值时间差分；
9. 所属社区节点比例；
10. 社区内部平均正连接强度；
11. 社区内部平均负连接幅值；
12. 社区外部平均正连接强度；
13. 社区外部平均负连接幅值；
14. 社区内部边密度；
15. 社区外部边密度。

社区内外强度均按可连接节点数归一化，避免模型仅根据社区规模作判断。首个窗口的所有差分定义为零；节点跨窗口对齐优先使用稳定节点名称，缺失节点不制造虚假变化。

## 4. 社区结构异常

原论文使用原始社区标签频率计算异常度，但独立社区发现产生的编号不具有跨窗口语义，因此本实现改用 6 维、编号重标记不变的窗口摘要：

- 社区数/节点数；
- 社区规模熵；
- 最大社区比例；
- 社区内平均正连接强度；
- 社区内平均负连接幅值；
- 社区内平均边密度。

只使用训练集拟合这些摘要的均值和标准差。窗口异常度定义为各维绝对标准化偏差的平均值。节点特征的均值和标准差也只在训练集拟合，并绑定数据协议 SHA-256。

## 5. 神经网络

### 5.1 窗口编码器

```text
15维节点特征
  → train-only standardization
  → LayerNorm
  → Linear
  → residual FFN (GELU)
  → LayerNorm
  → 有效节点均值池化
```

得到每个窗口的表示 \(z_m\)。

### 5.2 时序编码器

在窗口序列前加入可训练的 `CLS` token，再加入正弦位置编码。默认采用两层、四头 Transformer Encoder，并通过 padding mask 忽略无效窗口。最终取 `CLS` 输出作为短期时序表示。

### 5.3 原型记忆

默认使用 8 个可训练记忆槽。模型以 `CLS` 表示生成 query，对记忆槽进行缩放点积读取，再由门控单元融合 query 与读取结果。

记忆槽是普通模型参数，只由损失反向传播和优化器更新。普通 forward/evaluation 不执行隐式原地写操作，避免样本顺序泄漏和不可复现状态。

### 5.4 序列统计与分类头

序列统计共 6 维：

- 窗口平均连接强度的均值、标准差、最大值；
- 社区结构异常度的均值、标准差；
- \(\log(1+M_b)\)。

统计向量经 LayerNorm 和投影后，与 `CLS` 表示、原型记忆表示拼接，输入两层隐藏层的二分类头。

## 6. 训练与评估协议

- 类别加权交叉熵；
- AdamW、梯度裁剪、ReduceLROnPlateau 和 early stopping；
- 默认按 validation AUROC 选择最佳 checkpoint；
- 保存 `best_checkpoint.pt`、`last_checkpoint.pt`、`history.json`；
- 最佳 checkpoint 产生后，只在 validation 上拟合：
  - balanced-accuracy 阈值；
  - accuracy 阈值；
- test 只能复用已冻结的 validation 阈值。

训练日志同时记录 train/validation loss、BA、AUROC、学习率与记忆注意力归一化熵。

## 7. 程序入口

- `scripts/fit_structured_short_term_standardizer.py`：拟合训练集标准化状态；
- `scripts/train_structured_short_term.py`：正式训练和最佳模型选择；
- `scripts/evaluate_structured_short_term.py`：冻结阈值的 validation/test 评估。

核心实现：

- `src/keysubgraph/features/structured_short_term_features.py`；
- `src/keysubgraph/models/structured_short_term.py`；
- `src/keysubgraph/training/structured_short_term_trainer.py`。

## 8. 已固定的安全边界

- 不读取坐标；
- 不含任何社区 `Embedding`；
- 社区编号任意重标记不改变特征或预测；
- 节点一致置换不改变预测；
- padding 不参与 Transformer、池化、loss 或指标；
- validation/test 不参与标准化拟合；
- test 不用于模型选择或阈值选择。
