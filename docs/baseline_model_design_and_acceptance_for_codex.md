# 第一轮探索实验：统一基线模型设计与实现要求

## 0. 文档目的

本文档用于指导 Codex 实现第一轮探索实验所需的统一基线模型。

当前阶段的目标不是直接构建最终投稿模型，而是先完成一个结构简单、训练稳定、支持带符号图、支持可变长度图序列、支持每个时间窗多个关键子图，并且便于后续构造消融与对照实验的统一基线。

后续所有探索实验都应尽量复用同一套数据读取逻辑、子图编码器、训练流程、评估指标、checkpoint 规则和日志格式，仅通过配置项改变单个实验变量。

# 1. 实验背景

## 1.1 任务定义

第 \(b\) 个样本由多个时间窗组成：

\[
\mathcal{G}_b=\{G_b^{(1)},G_b^{(2)},\ldots,G_b^{(M_b)}\},
\]

其中：

- \(M_b\) 是样本 \(b\) 的有效时间窗数量；
- 不同样本允许具有不同的 \(M_b\)；
- 每个 \(G_b^{(m)}\) 是第 \(m\) 个时间窗的图；
- 图中的边权允许为正或负；
- 分类标签为：

\[
Y_b\in\{0,1\}.
\]

当前项目中，关键子图提取器已经完成。因此，基线模型不再负责从原图中学习关键子图，而是直接读取已经提取好的关键子图。

## 1.2 子图输入形式

对于样本 \(b\) 的第 \(m\) 个时间窗，可能存在多个关键子图：

\[
\mathcal{S}_b^{(m)}
=
\{S_{b,1}^{(m)},S_{b,2}^{(m)},\ldots,S_{b,K_{b,m}}^{(m)}\},
\]

其中每个关键子图写为：

\[
S_{b,k}^{(m)}
=
(V_{b,k}^{(m)},A_{b,k}^{(m)},X_{b,k}^{(m)}).
\]

其中：

- \(V_{b,k}^{(m)}\) 是节点集合；
- \(A_{b,k}^{(m)}\) 是带符号加权邻接矩阵；
- \(X_{b,k}^{(m)}\) 是节点特征矩阵；
- \(K_{b,m}\) 允许随样本和时间窗变化。

## 1.3 带符号边

邻接矩阵中的边权满足：

\[
A_{ij}\in\mathbb{R}.
\]

因此：

- \(A_{ij}>0\) 表示正连接；
- \(A_{ij}<0\) 表示负连接；
- \(A_{ij}=0\) 表示无连接或低于边存在阈值。

负边是有效结构信息，不能直接删除，也不能直接与正边混合进行普通 GCN 归一化。基线模型必须采用正负边分离的消息传递方式。

## 1.4 可变长度要求

基线必须支持：

1. 不同样本具有不同时间窗数量 \(M_b\)；
2. 每个时间窗具有不同关键子图数量 \(K_{b,m}\)；
3. 不同关键子图具有不同节点数；
4. padding 不得参与消息传递、池化、状态更新和损失计算；
5. 最终分类必须使用最后一个有效时间窗对应的隐藏状态。

推荐优先采用 list-based batching。若使用 padding，则必须显式维护：

- `time_mask`
- `subgraph_mask`
- `node_mask`
- `edge_mask`

## 1.5 当前阶段目标

第一轮探索主要用于判断：

- 历史信息是否有用；
- 时间顺序是否有用；
- 学习得到的关键子图是否优于其他子图；
- 结构统计先验是否有用；
- 子图扰动是否会影响最终分类。

因此基线模型必须保持简单，避免复杂模型本身掩盖这些机制。

# 2. 固定基线架构

统一基线采用：

\[
oxed{
	ext{Signed Subgraph Encoder}
ightarrow
	ext{Window-level Pooling}
ightarrow
	ext{GRU Evolution Module}
ightarrow
	ext{MLP Classifier}
}
\]

完整数据流为：

\[
S_{b,k}^{(m)}
ightarrow
h_{b,k}^{(m)}
ightarrow
ar h_b^{(m)}
ightarrow
u_b^{(m)}
ightarrow
r_b^{(m)}
ightarrow
\hat Y_b.
\]

结构流程：

```text
单个关键子图
    │
    ├── 正边聚合
    ├── 负边聚合
    └── 两层 Signed Message Passing
             │
             ▼
      Mean Pool + Max Pool
             │
             ▼
       单个子图表示
             │
             ▼
同一时间窗多个子图做有效均值池化
             │
             ▼
       时间窗子图表示
             │
             ▼
        输入投影与归一化
             │
             ▼
       单层单向 GRU
             │
             ▼
    最后一个有效隐藏状态
             │
             ▼
       两层 MLP 分类器
             │
             ▼
         二分类 logits
```

# 3. 子图编码器设计

## 3.1 正负邻接分解

\[
A_{ij}^{+}=\max(A_{ij},0),
\]

\[
A_{ij}^{-}=\max(-A_{ij},0).
\]

其中：

- \(A^+\) 保存正边权重；
- \(A^-\) 保存负边绝对幅值；
- 二者均为非负矩阵。

## 3.2 正负消息归一化

\[
\widetilde A_{ij}^{+}
=
rac{A_{ij}^{+}}
{\sum_t A_{it}^{+}+arepsilon},
\]

\[
\widetilde A_{ij}^{-}
=
rac{A_{ij}^{-}}
{\sum_t A_{it}^{-}+arepsilon}.
\]

若节点不存在正邻居或负邻居，对应消息应为零向量。

## 3.3 Signed Message Passing

第 \(l\) 层节点表示记为 \(h_i^{(l)}\)，初始：

\[
h_i^{(0)}=x_i.
\]

正边消息：

\[
m_i^{+(l)}
=
\sum_j
\widetilde A_{ij}^{+}
W_+^{(l)}h_j^{(l)}.
\]

负边消息：

\[
m_i^{-(l)}
=
\sum_j
\widetilde A_{ij}^{-}
W_-^{(l)}h_j^{(l)}.
\]

节点更新采用拼接后 MLP：

\[
h_i^{(l+1)}
=
\operatorname{MLP}_{\mathrm{msg}}^{(l)}
\left(
[h_i^{(l)};m_i^{+(l)};m_i^{-(l)}]
ight).
\]

第一版不建议直接使用 \(m_i^+-m_i^-\)，因为那会预设负消息必须以减法方式起作用。拼接方式更灵活。

## 3.4 层数与维度

默认配置：

- Signed Message Passing 层数：2；
- 节点隐藏维度：64；
- 激活函数：ReLU；
- 可使用 LayerNorm；
- dropout：0.1 或 0.2。

推荐：

\[
d_{\mathrm{node}}ightarrow64ightarrow64.
\]

## 3.5 残差连接

若维度一致，可使用：

\[
h_i^{(l+1)}
=
h_i^{(l)}
+
\operatorname{MLP}_{\mathrm{msg}}^{(l)}
([h_i^{(l)};m_i^{+(l)};m_i^{-(l)}]).
\]

# 4. 单个子图池化

对有效节点执行均值池化和最大池化：

\[
h_{b,k}^{(m),\mathrm{mean}}
=
rac{1}{|V_{b,k}^{(m)}|}
\sum_{i\in V_{b,k}^{(m)}}h_i,
\]

\[
h_{b,k}^{(m),\mathrm{max}}
=
\max_{i\in V_{b,k}^{(m)}}h_i.
\]

拼接得到：

\[
h_{b,k}^{(m)}
=
[h_{b,k}^{(m),\mathrm{mean}};
h_{b,k}^{(m),\mathrm{max}}].
\]

若节点隐藏维度为 64，则单子图表示维度为 128。

必须使用 mask-aware pooling，padding 节点不能参与均值或最大值。

# 5. 时间窗内多个子图聚合

基线采用有效子图均值：

\[
ar h_b^{(m)}
=
rac{1}{K_{b,m}}
\sum_{k=1}^{K_{b,m}}h_{b,k}^{(m)}.
\]

若使用 padding：

\[
ar h_b^{(m)}
=
rac{
\sum_k M_{b,k}^{(m)}h_{b,k}^{(m)}
}{
\sum_k M_{b,k}^{(m)}+arepsilon
}.
\]

第一版不使用注意力池化，避免引入额外的可学习子图选择机制。

# 6. 显式结构特征接口

第一版基线推荐默认关闭显式结构特征：

```yaml
use_structural_features: false
```

原因是统计结构先验属于后续待验证变量，不能一开始就作为基线默认启用。

后续若启用结构特征，时间窗结构向量记为：

\[
z_b^{(m)}\in\mathbb{R}^{d_z}.
\]

所有结构特征必须只使用训练集统计量进行标准化：

\[
\widehat z_{b,j}^{(m)}
=
rac{z_{b,j}^{(m)}-\mu_j^{\mathrm{train}}}
{\sigma_j^{\mathrm{train}}+arepsilon}.
\]

投影：

\[
e_b^{(m)}
=
\operatorname{MLP}_{\mathrm{struct}}
(\widehat z_b^{(m)}),
\]

推荐维度：

\[
d_zightarrow32.
\]

预留先验模式：

```yaml
prior_mode:
  - none
  - uniform
  - real
  - permuted
```

第一版默认：

```yaml
prior_mode: none
```

# 7. 时间窗输入投影

若不使用结构特征：

\[
u_b^{(m)}=ar h_b^{(m)}.
\]

若使用结构特征：

\[
u_b^{(m)}
=
[ar h_b^{(m)};e_b^{(m)}].
\]

统一映射到 GRU 输入维度：

\[
\widetilde u_b^{(m)}
=
\operatorname{LayerNorm}
\left(
\operatorname{ReLU}
(W_u u_b^{(m)}+b_u)
ight).
\]

推荐：

\[
\widetilde u_b^{(m)}\in\mathbb{R}^{128}.
\]

# 8. 演化模块设计

## 8.1 模块选择

使用单层、单向 GRU。

不使用：

- 双向 GRU；
- LSTM；
- Transformer；
- Temporal Attention；
- 多层复杂时序模块。

## 8.2 状态更新

\[
r_b^{(m)}
=
\operatorname{GRUCell}
(\widetilde u_b^{(m)},r_b^{(m-1)}),
\]

初始状态：

\[
r_b^{(0)}=\mathbf{0}.
\]

隐藏维度：

\[
r_b^{(m)}\in\mathbb{R}^{128}.
\]

默认配置：

```yaml
gru:
  input_size: 128
  hidden_size: 128
  num_layers: 1
  bidirectional: false
```

## 8.3 可变长度更新

若 \(T_b^{(m)}\in\{0,1\}\) 表示时间窗是否有效，则：

\[
r_b^{(m)}
=
egin{cases}
\operatorname{GRUCell}
(\widetilde u_b^{(m)},r_b^{(m-1)}),
&T_b^{(m)}=1,\
r_b^{(m-1)},
&T_b^{(m)}=0.
\end{cases}
\]

最终分类使用最后一个有效状态：

\[
r_b^{\mathrm{final}}=r_b^{(M_b)}.
\]

不能使用双向 GRU，因为双向结构会利用未来时间窗，破坏历史传递和前缀信息的解释。

# 9. 分类器设计

分类头：

\[
r_b^{\mathrm{final}}
ightarrow
\operatorname{Linear}(128,64)
ightarrow
\operatorname{ReLU}
ightarrow
\operatorname{Dropout}(0.2)
ightarrow
\operatorname{Linear}(64,2).
\]

输出：

\[
o_b=[o_{b,0},o_{b,1}]\in\mathbb{R}^2.
\]

模型内部不要执行 softmax。

训练损失：

\[
\mathcal L_{\mathrm{train}}
=
\operatorname{WeightedCrossEntropy}(o_b,Y_b).
\]

评估阶段额外计算：

- AUROC；
- balanced accuracy；
- unweighted log-loss；
- accuracy；
- confusion matrix。

理论相关 log-loss 必须使用未加权交叉熵。

# 10. 默认配置

```yaml
model:
  node_hidden_dim: 64
  signed_gnn_layers: 2
  signed_gnn_dropout: 0.1
  use_residual: true

  subgraph_pooling:
    mean: true
    max: true

  window_pooling: mean

  use_structural_features: false
  structural_hidden_dim: 32

  fusion_dim: 128

  gru:
    hidden_dim: 128
    num_layers: 1
    bidirectional: false

  classifier:
    hidden_dim: 64
    dropout: 0.2
    num_classes: 2
```

# 11. 建议代码结构

```text
src/
├── data/
│   ├── dataset.py
│   ├── collate.py
│   └── masks.py
├── models/
│   ├── signed_message_passing.py
│   ├── subgraph_encoder.py
│   ├── window_pooling.py
│   ├── evolution_gru.py
│   ├── classifier.py
│   └── baseline_model.py
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── checkpoint.py
├── configs/
│   └── baseline.yaml
└── tests/
    ├── test_dataset.py
    ├── test_signed_encoder.py
    ├── test_masks.py
    ├── test_gru.py
    └── test_end_to_end.py
```

# 12. 前向接口

建议：

```python
logits, aux = model(batch)
```

其中：

```python
aux = {
    "subgraph_embeddings": ...,
    "window_embeddings": ...,
    "hidden_states": ...,
    "final_hidden_state": ...,
    "time_mask": ...,
}
```

至少必须能够导出：

- 每个时间窗表示；
- 每个时间窗隐藏状态；
- 最终隐藏状态。

# 13. 基线验收条件

## 13.1 数据读取

必须确认：

1. 能正确读取单个样本；
2. 能正确读取多个时间窗；
3. 能正确读取每个时间窗的多个关键子图；
4. 能处理不同样本的不同 \(M_b\)；
5. 能处理不同时间窗的不同 \(K_{b,m}\)；
6. 能处理不同子图节点数；
7. 标签读取正确；
8. 张量 dtype 正确；
9. Dataset 内不随机重新划分数据。

## 13.2 带符号图

构造包含正边、负边、无正邻居节点和无负邻居节点的人工小图，检查：

1. \(A^+\) 仅保留正边；
2. \(A^-\) 仅保留负边绝对值；
3. 正消息和负消息均被计算；
4. 删除负边后输出发生变化；
5. 将负边改为正边后输出发生变化；
6. 不出现 NaN 或 Inf。

## 13.3 单样本前向

```python
logits, aux = model(sample_batch)
```

要求：

```text
logits.shape == [1, 2]
```

并确认：

- 所有有效时间窗均被处理；
- 最终状态来自最后有效时间窗；
- 输出无 NaN 和 Inf。

## 13.4 多样本 batch

batch 至少包含：

- 不同时间窗数量的样本；
- 不同子图数量的时间窗；
- 不同节点数的子图。

要求：

```text
logits.shape == [batch_size, 2]
```

并确认 mask、样本隔离和长度索引正确。

## 13.5 Padding 不变性

对同一样本分别增加无效时间窗、无效子图和无效节点 padding。在 `model.eval()` 下：

\[
\|o_{\mathrm{original}}-o_{\mathrm{padded}}\|<10^{-6}
\]

或采用合理的浮点容忍范围。

## 13.6 无效时间窗状态

对于 padding 时间窗：

\[
T^{(m)}=0
\]

必须满足：

\[
r^{(m)}=r^{(m-1)}.
\]

## 13.7 梯度

执行：

```python
loss.backward()
```

检查以下模块均有有限梯度：

- 正边参数；
- 负边参数；
- 子图编码器；
- 输入投影层；
- GRU；
- 分类器。

## 13.8 小数据过拟合

选取 8 或 16 个训练样本，关闭大部分正则化并训练足够轮次。

建议验收标准：

```text
training accuracy >= 95%
```

或训练损失接近 0。

如果无法过拟合，应检查标签、mask、最终状态索引、梯度、损失、空输入和错误 detach。

## 13.9 可重复性

固定：

- Python seed；
- NumPy seed；
- PyTorch seed；
- 数据顺序；
- 模型初始化；
- CUDA 可重复性设置。

同一配置重复运行结果应基本一致。

## 13.10 Checkpoint

必须支持：

1. 保存模型；
2. 保存优化器；
3. 保存 epoch；
4. 保存验证指标；
5. 保存配置；
6. 重新加载；
7. 加载后对同一输入产生相同输出。

## 13.11 指标

在人工数组上验证：

- accuracy；
- balanced accuracy；
- AUROC；
- unweighted log-loss；
- confusion matrix。

明确：

- 训练损失可以使用 weighted CE；
- 理论评估必须使用 unweighted log-loss；
- AUROC 输入为类别 1 概率。

# 14. 基线冻结条件

满足以下条件后才能冻结基线：

1. 所有前向测试通过；
2. 所有 mask 测试通过；
3. signed edge 测试通过；
4. backward 梯度通过；
5. 小样本可过拟合；
6. checkpoint 可恢复；
7. 固定 seed 可复现；
8. 在正式训练集和验证集上完整跑通一次；
9. 日志和指标文件完整；
10. 配置可追踪。

冻结后，以下内容不应在探索实验间随意改变：

- 数据划分；
- 子图导出文件；
- 节点特征定义；
- 模型隐藏维度；
- 优化器；
- 学习率；
- 训练轮数；
- early stopping；
- checkpoint 选择规则；
- 随机种子；
- 评估指标。

# 15. 后续实验接口

建议预留：

```yaml
experiment:
  history_mode: full
  temporal_order: ordered
  subgraph_source: key
  prior_mode: none
```

后续扩展：

```yaml
history_mode:
  - full
  - current_only
  - reset_state

temporal_order:
  - ordered
  - shuffled
  - bag

subgraph_source:
  - key
  - low_score
  - top_degree
  - random
  - raw_graph

prior_mode:
  - none
  - uniform
  - real
  - permuted
```

# 16. Codex 实现要求

1. 不要重写已有关键子图提取器；
2. 先检查当前项目目录和数据格式；
3. 不假设固定时间窗数量；
4. 不假设固定子图节点数；
5. 不使用 PyTorch Geometric；
6. 使用当前环境支持的 PyTorch；
7. 所有新增配置有默认值；
8. 核心模块带类型注解和 docstring；
9. docstring 明确张量 shape；
10. 所有 mask 语义统一；
11. 不静默忽略空子图或空时间窗；
12. 空输入应抛出明确异常或按配置处理；
13. 不在训练代码内部随机划分数据；
14. 不在验证集或测试集计算训练统计量；
15. 第一阶段不加入注意力、Transformer 或复杂时序模型；
16. 先完成测试，再运行完整训练。

# 17. 最终基线定义

\[
oxed{
	ext{两层正负分离消息传递}
+
	ext{Mean/Max 子图池化}
+
	ext{时间窗内子图均值聚合}
+
	ext{单层单向 GRU}
+
	ext{两层 MLP 分类器}
}
\]

数学形式：

\[
h_{b,k}^{(m)}
=
f_{\mathrm{signed}}(S_{b,k}^{(m)}),
\]

\[
ar h_b^{(m)}
=
\operatorname{MeanPool}_k(h_{b,k}^{(m)}),
\]

\[
u_b^{(m)}
=
f_{\mathrm{proj}}(ar h_b^{(m)}),
\]

\[
r_b^{(m)}
=
\operatorname{GRU}(u_b^{(m)},r_b^{(m-1)}),
\]

\[
\hat Y_b
=
f_{\mathrm{cls}}(r_b^{(M_b)}).
\]

第一版不默认启用真实结构先验，也不使用复杂时序模型。

只有基线通过全部验收后，才开始逐步实现：

- `current_only`
- `reset_state`
- `shuffled`
- `bag`
- 不同子图来源
- 不同统计先验
