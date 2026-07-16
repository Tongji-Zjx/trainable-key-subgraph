# 第一轮探索实验：统一基线模型设计与实现要求

## 0. 文档目的

本文档用于指导 Codex 实现第一轮探索实验所需的统一基线模型。

当前阶段的目标不是直接构建最终投稿模型，而是先完成一个结构简单、训练稳定、支持带符号图、支持可变长度图序列、支持每个时间窗多个关键子图，并且便于后续构造消融与对照实验的统一基线。

后续所有探索实验都应尽量复用同一套数据读取逻辑、子图编码器、训练流程、评估指标、checkpoint 规则和日志格式，仅通过配置项改变单个实验变量。

本设计区分两种证据等级：

- `exploratory_in_sample`：允许使用全样本训练的提取器和全样本导出，仅用于发现候选现象；
- `confirmatory_cross_fitted`：提取器、统计先验和分类基线均不得接触外层测试样本标签，用于验证机制和泛化性。

除非特别说明，文中“支持某项理论假设”均指第二种证据等级下的结果。全样本探索结果不能表述为独立验证或可泛化证明。

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

### 1.2.1 现有硬子图导出接口

现有硬子图 JSON 不直接保存完整的 \(X_{b,k}^{(m)}\)，而是保存：

- `original_graph_ref`；
- `time_index`；
- `node_ids` 和 `node_names`；
- 全局节点编号形式的 `edge_index`；
- `original_edge_weights`；
- 各类 mask、提取分数和候选池信息。

因此 Dataset 必须根据 `original_graph_ref` 读取对应 `.pt` 文件，并根据 `time_index`、`node_ids` 和 `edge_index` 重建子图。全局边端点必须显式重映射为子图局部节点编号，禁止直接将全局编号送入子图编码器。

加载时必须验证：

1. `node_ids` 无重复且均在原图范围内；
2. 每条边的两个端点均属于 `node_ids`；
3. `original_edge_weights` 与原图对应位置一致；
4. `edge_presence_threshold` 与数据协议一致；
5. 导出文件的 checkpoint hash 和 protocol hash 与当前实验 manifest 一致；
6. JSON 顶层标签只作为监督目标，不能进入任何特征构造函数。

### 1.2.2 节点特征上下文

默认使用：

```yaml
node_feature_context: induced_subgraph
include_temporal_deltas: false
use_node_identity: false
```

即在每个诱导子图内部重新计算具有跨样本一致语义的节点结构特征。默认包括：

- 绝对连接强度；
- 正连接强度；
- 负连接幅值；
- 正连接比例；
- 负连接比例；
- 社区相对规模；
- 社区内/外正连接平均强度；
- 社区内/外负连接平均幅值；
- 社区正/负连接密度。

社区编号只用于同社区判断和结构特征计算，不允许使用 `nn.Embedding(community_id)`。`node_ids`、`node_names`、ROI 名称和空间坐标默认也不作为身份特征输入。

默认在诱导子图内重算特征，是为了检验“子图自身是否包含判别结构”。如果后续增加：

```yaml
node_feature_context: original_graph
```

则必须作为独立消融，因为这种模式会把子图外部的原图连接信息带入节点特征，不能与纯子图结果混合解释。

以下提取器专属字段禁止作为分类模型输入：

- `node_scores`；
- `edge_scores`；
- `candidate_score`；
- `score_node`、`score_edge`、`score_connectivity`、`score_dynamic`；
- `seed_node`；
- 任何由真实标签直接构造的逐样本特征。

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

`edge_mask` 必须统一定义为：

\[
M_{ij}^{\mathrm{edge}}
=
\mathbf 1\left(|A_{ij}|>\tau_{\mathrm{edge}}\right),
\]

其中 \(\tau_{\mathrm{edge}}\) 来自冻结的数据协议。训练、控制子图构造、原图对照和统计模块必须使用同一个阈值。

空子图或空时间窗不得被静默丢弃。默认策略为抛出包含 `sample_id` 和 `time_index` 的明确异常。若某类控制子图匹配失败，则由预先生成的匹配 manifest 决定是否从所有来源的配对分析中共同排除该 tuple。

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
\boxed{
\text{Signed Subgraph Encoder}

\rightarrow
\text{Window-level Pooling}

\rightarrow
\text{GRU Evolution Module}

\rightarrow
\text{MLP Classifier}
}
\]

完整数据流为：

\[
S_{b,k}^{(m)}

\rightarrow
h_{b,k}^{(m)}

\rightarrow
\bar h_b^{(m)}

\rightarrow
u_b^{(m)}

\rightarrow
r_b^{(m)}

\rightarrow
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
\frac{A_{ij}^{+}}
{\sum_t A_{it}^{+}+\varepsilon},
\]

\[
\widetilde A_{ij}^{-}
=
\frac{A_{ij}^{-}}
{\sum_t A_{it}^{-}+\varepsilon}.
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

\right).
\]

第一版不建议直接使用 \(m_i^+-m_i^-\)，因为那会预设负消息必须以减法方式起作用。拼接方式更灵活。

正负通道分别归一化会保留各通道内部的相对邻接权重，但会弱化正、负连接总强度。因此，节点输入必须保留正连接强度、负连接幅值及其比例，不能只输入常数或无符号度。若节点不存在某一符号的邻居，该通道消息必须严格为零，不允许产生 NaN。

第一轮基线的消息传递只使用当前子图中的 \(A_{ij}\) 及其符号通道，不使用提取器的 node/edge score。时间差分边特征默认关闭，作为后续独立动态特征消融：

```yaml
include_temporal_deltas: false
```

当启用差分边特征时，至少保留：

\[
e_{ij}
=
[A_{ij},|A_{ij}|,\Delta A_{ij},|\Delta A_{ij}|],
\]

并使用 `delta_edge_mask` 排除无法跨时间对齐的边。

## 3.4 层数与维度

默认配置：

- Signed Message Passing 层数：2；
- 节点隐藏维度：64；
- 激活函数：ReLU；
- 可使用 LayerNorm；
- dropout：0.1 或 0.2。

推荐：

\[
d_{\mathrm{node}}
\rightarrow64
\rightarrow64.
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
\frac{1}{|V_{b,k}^{(m)}|}
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
\bar h_b^{(m)}
=
\frac{1}{K_{b,m}}
\sum_{k=1}^{K_{b,m}}h_{b,k}^{(m)}.
\]

若使用 padding：

\[
\bar h_b^{(m)}
=
\frac{
\sum_k M_{b,k}^{(m)}h_{b,k}^{(m)}
}{
\sum_k M_{b,k}^{(m)}+\varepsilon
}.
\]

第一版不使用注意力池化，避免引入额外的可学习子图选择机制。

均值聚合不把 \(K_{b,m}\) 作为额外特征，以避免模型仅根据提取成功数量判断类别。同一时间窗的多个子图允许部分重叠，但必须在数据审计中报告节点/边重复率。控制来源若产生完全重复子图，也必须记录重复率，不能静默去重后改变有效 \(K_{b,m}\)。

对于有效时间窗，必须满足 \(K_{b,m}\ge1\)。控制子图匹配失败不应由模型内部用零向量临时替代，而应在训练前由匹配 manifest 统一处理。

# 6. 显式结构特征接口

中性基线默认关闭显式结构特征：

```yaml
use_structural_features: false
```

原因是统计结构先验属于后续待验证变量，不能一开始就作为基线默认启用。为了在先验实验中保持参数量一致，关闭结构特征时仍保留同维度结构分支，但输入固定零向量，并禁止该分支偏置产生非零常量。

后续若启用结构特征，时间窗结构向量记为：

\[
z_b^{(m)}\in\mathbb{R}^{d_z}.
\]

所有结构特征必须只使用训练集统计量进行标准化：

\[
\widehat z_{b,j}^{(m)}
=
\frac{z_{b,j}^{(m)}-\mu_j^{\mathrm{train}}}
{\sigma_j^{\mathrm{train}}+\varepsilon}.
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
d_z
\rightarrow32.
\]

结构特征与先验必须作为两个独立实验因素。预留先验模式：

```yaml
prior_mode:
  - none
  - uniform
  - real
  - permuted
```

第一版中性基线默认：

```yaml
prior_mode: none
```

必须至少比较：

| 组别 | 结构分支输入 | 先验模式 | 目的 |
|---|---|---|---|
| A | 零向量 | `none` | 中性基线 |
| B | 真实结构特征 | `none` | 结构特征本身是否有效 |
| C | 真实结构特征 | `uniform` | 统一缩放对照 |
| D | 真实结构特征 | `real` | 真实先验 |
| E | 真实结构特征 | `permuted` | 保持权重分布但打乱维度对应关系 |

真实先验、结构特征标准化参数和置乱映射只能由当前训练折计算并冻结。validation/test 只能应用这些参数，不能参与估计。

如果先验使用可逆残差缩放：

\[
W_A=\operatorname{diag}(1+\beta a_j),
\]

则 `uniform` 可能只是可被线性层或 LayerNorm 吸收的统一缩放，因此核心先验比较是 `real` 对 `permuted`，而不是仅比较 `real` 对 `none`。所有先验模式必须保持结构分支维度、参数量和训练预算一致。

# 7. 时间窗输入投影

若不使用结构特征：

\[
u_b^{(m)}=\bar h_b^{(m)}.
\]

若使用结构特征：

\[
u_b^{(m)}
=
[\bar h_b^{(m)};e_b^{(m)}].
\]

统一映射到 GRU 输入维度：

\[
\widetilde u_b^{(m)}
=
\operatorname{LayerNorm}
\left(
\operatorname{ReLU}
(W_u u_b^{(m)}+b_u)

\right).
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
\begin{cases}
\operatorname{GRUCell}
(\widetilde u_b^{(m)},r_b^{(m-1)}),
&T_b^{(m)}=1,\\
r_b^{(m-1)},
&T_b^{(m)}=0.
\end{cases}
\]

最终分类使用最后一个有效状态：

\[
r_b^{\mathrm{final}}=r_b^{(M_b)}.
\]

不能使用双向 GRU，因为双向结构会利用未来时间窗，破坏历史传递和前缀信息的解释。

## 8.4 历史模式的严格定义

`history_mode` 必须使用以下互不混淆的定义。

### full

按原始顺序处理所有有效时间窗：

\[
r_b^{(m)}
=
\operatorname{GRUCell}(\widetilde u_b^{(m)},r_b^{(m-1)}).
\]

### current_only

只使用最后有效时间窗，并从零状态更新一次：

\[
r_b^{\mathrm{current}}
=
\operatorname{GRUCell}(\widetilde u_b^{(M_b)},0).
\]

该模式与 full 共享子图编码器、输入投影、GRUCell 和分类头参数规模。

### truncate_history

只保留最后 \(L_b\) 个有效时间窗，相当于在截断点前清除历史，然后继续递归：

\[
r_b^{(M_b-L_b)}=0.
\]

为支持可变长度，\(L_b\) 可以由统一保留比例 \(\rho\in(0,1]\) 决定：

\[
L_b=\max(1,\lceil\rho M_b\rceil).
\]

通过多个 \(\rho\) 可以得到历史保留曲线。禁止把“每个窗口前都清零、最终只读取最后状态”的 `reset_state` 当作独立实验，因为它与 `current_only` 数学等价。

### independent_bag

每个时间窗独立投影，不使用递归状态，然后以 permutation-invariant 的均值池化汇总所有有效窗口，再使用参数量匹配的 MLP 分类。该模式保留所有窗口内容，但不建模顺序或状态传递。

## 8.5 顺序模式的严格定义

主顺序实验必须设置：

```yaml
include_temporal_deltas: false
use_time_position: false
```

以避免 \(\Delta A\)、\(\Delta degree\) 或时间位置提前泄漏顺序。比较模式为：

- `ordered`：原始时间顺序；
- `shuffled`：保持窗口多重集合不变，仅置换顺序；
- `bag`：使用 `independent_bag`，对排列严格不变。

`shuffled` 必须使用冻结随机种子生成可复现置换。正式结果至少汇总多个独立置换，不能只依赖一次偶然打乱。若后续启用时间差分，必须作为“差分特征增益”独立实验，不能与纯顺序结论混合。

`bag` 与 GRU 基线应尽量匹配表示维度、分类头规模和训练预算；若参数量无法完全相同，必须在结果中报告参数量并增加容量匹配敏感性分析。

# 9. 分类器设计

分类头：

\[
r_b^{\mathrm{final}}

\rightarrow
\operatorname{Linear}(128,64)

\rightarrow
\operatorname{ReLU}

\rightarrow
\operatorname{Dropout}(0.2)

\rightarrow
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

主机制实验默认使用 validation unweighted log-loss 选择最佳 checkpoint，使模型选择指标与理论代理一致。AUROC 可以作为预先声明的替代选择指标，但同一批实验不得在看到 test 结果后切换规则。

accuracy、balanced accuracy 和 confusion matrix 所需分类阈值只能在 validation 上选择并写入 checkpoint；test 阶段必须复用该阈值。AUROC 和 unweighted log-loss 不依赖该硬阈值。

# 10. 默认配置

```yaml
model:
  node_feature_context: induced_subgraph
  include_temporal_deltas: false
  use_node_identity: false
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
  prior_mode: none

  fusion_dim: 128

  gru:
    hidden_dim: 128
    num_layers: 1
    bidirectional: false

  classifier:
    hidden_dim: 64
    dropout: 0.2
    num_classes: 2

experiment:
  evidence_level: exploratory_in_sample
  history_mode: full
  history_keep_ratio: 1.0
  temporal_order: ordered
  subgraph_source: key
  random_repeat_index: 0

evaluation:
  primary_metric: unweighted_log_loss
  secondary_metrics:
    - roc_auc
    - balanced_accuracy
    - accuracy
  bootstrap_unit: subject_id
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

## 13.12 数据模式与泄漏测试

必须增加以下自动检查：

1. 硬子图全局节点编号正确映射到局部连续编号；
2. JSON 边权与 `original_graph_ref` 对应原图一致；
3. 标签未传入节点/边特征构造函数；
4. 禁止字段未出现在模型输入中；
5. 所有来源使用相同节点特征定义；
6. 所有来源使用同一个 edge threshold；
7. validation/test 未参与标准化、先验、类别权重或阈值估计；
8. checkpoint、protocol、split 和 export hash 完全匹配。

可以构造“标签置乱但图不变”的人工测试，确认 Dataset 输出特征完全不变，只有监督标签变化。

## 13.13 历史模式测试

必须验证：

1. `current_only` 输出不受早期窗口变化影响；
2. `full` 在正常初始化下允许早期窗口影响最终状态；
3. `truncate_history` 只受保留窗口影响；
4. 每窗口清零且只取最后状态的实现与 `current_only` 数值相同，因此不得被注册为独立实验；
5. 当 \(M_b=1\) 时，`full` 与 `current_only` 应在 eval 模式下等价。

## 13.14 顺序与 Bag 测试

必须验证：

1. `bag` 对窗口排列保持不变；
2. `shuffled` 只改变窗口顺序，不改变窗口内容和有效数量；
3. 主顺序实验中不存在 delta 或位置特征；
4. 固定 seed 时置换结果可复现；
5. 不同置换 seed 确实产生不同的有效排列；
6. padding 时间窗不参与置换。

## 13.15 控制来源匹配测试

对每个 matched tuple，key、low-score、Top-degree 和 random 至少匹配：

- `sample_id`；
- `time_index`；
- `subgraph_index`；
- 节点数；
- 边数；
- 有效时间窗集合；
- 有效子图数量。

连通性可以作为附加匹配条件或协变量，但必须在实验前固定。若任一主要来源匹配失败，该 tuple 必须从所有配对来源共同排除，并在 manifest 中记录原因。

随机控制的每个 repeat 必须形成独立、完整的控制数据集。禁止将多个 random repeats 同时池化到一个时间窗。Top-degree 完全重复子图和 key 子图重叠率必须被统计并写入数据审计报告。

## 13.16 统计验收测试

在人工预测数组上验证：

1. paired log-loss difference 的方向；
2. subject-level bootstrap 的重采样单位；
3. 同一 subject 的多个 session 不会被拆开重采样；
4. 置信区间与固定 seed 可复现；
5. 多重比较 FDR 修正正确；
6. 缺失配对不会被当作零差异；
7. test 指标不会被用于选择 checkpoint、阈值或超参数。

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
10. 配置可追踪；
11. 数据、split、子图导出、checkpoint 和 protocol hash 均写入 manifest；
12. 所有实验明确标记 evidence level；
13. matched-control manifest 通过完整性检查；
14. 理论比较的 unweighted log-loss 与训练 weighted CE 分开记录；
15. validation 阈值能够随 checkpoint 保存和恢复。

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

对于 `confirmatory_cross_fitted` 实验，还必须冻结：

- 外层 group-aware split；
- 内层 validation 规则；
- 每个外层折对应的提取器 checkpoint；
- 每折训练数据估计的结构标准化参数和统计先验；
- 匹配 tuple 集合；
- 主要假设、主要指标和比较方向。

# 15. 后续实验接口

统一实验配置：

```yaml
experiment:
  evidence_level: exploratory_in_sample  # or confirmatory_cross_fitted
  history_mode: full                     # full | current_only | truncate_history | independent_bag
  history_keep_ratio: 1.0
  temporal_order: ordered                # ordered | shuffled | bag
  permutation_seed: 42
  subgraph_source: key                   # key | low_score | top_degree | random | raw_graph
  random_repeat_index: 0
  node_feature_context: induced_subgraph
  include_temporal_deltas: false
  use_structural_features: false
  prior_mode: none                       # none | uniform | real | permuted
```

## 15.1 历史信息实验

固定：

- `temporal_order: ordered`；
- `subgraph_source: key`；
- `include_temporal_deltas: false`；
- `prior_mode: none`。

比较：

1. `full`；
2. `current_only`；
3. 多个 `truncate_history` 保留比例；
4. `independent_bag`。

主要问题分别是：历史是否有用、保留多长历史才有用，以及所有窗口内容在没有递归传递时是否已经足够。

## 15.2 时间顺序实验

固定相同窗口集合和相同子图表示，关闭所有 delta 与位置特征，比较：

1. ordered GRU；
2. 多个可复现置换下的 shuffled GRU；
3. 参数量尽量匹配的 independent Bag。

只有 ordered 在未见样本上稳定优于 shuffled 和 Bag，才能支持“顺序贡献”假设。ordered 优于 current-only 但不优于 Bag，只能说明历史集合有用，不能说明顺序有用。

## 15.3 子图来源实验

主要来源：

- key；
- matched low-score；
- matched Top-degree；
- matched random；
- raw graph。

key、low-score、Top-degree 和 random 使用冻结的 matched-control manifest。每个配对 tuple 至少匹配样本、时间窗、节点数和边数；匹配失败时所有主要来源共同排除。

random 的每个 repeat 是独立数据集和独立实验重复，不能将多个 repeat 合并为同一时间窗的多个输入。raw graph 由于尺寸和子图数不匹配，只作为“相对原图压缩/去噪”的补充比较，不与尺寸匹配控制混为同一推断。

## 15.4 结构特征与统计先验实验

按照第 6 节 A～E 五组执行。核心问题分成两步：

1. 结构特征本身是否带来增益：B 对 A；
2. 真实先验是否比相同权重分布的错误维度映射更好：D 对 E。

不得仅通过 D 对 A 同时声称结构特征和统计先验均有效。

## 15.5 时间差分特征实验

在 ordered 模式下单独比较：

- `include_temporal_deltas: false`；
- `include_temporal_deltas: true`。

该实验检验显式变化特征是否改善有限模型可学习性。由于差分是原始窗口序列的确定性函数，结果不能解释为增加了理想完整历史的真实互信息。

## 15.6 子图扰动实验

扰动必须预先定义类型和强度，例如：

- 删除或替换一定比例的高分节点；
- 删除或替换一定比例的高分边；
- 只扰动正边或只扰动负边；
- 删除早期、中期或晚期时间窗；
- 将早期子图替换为尺寸匹配 random 子图。

建议强度为 \(0\%\)、\(10\%\)、\(25\%\)、\(50\%\)。每种目标扰动都必须配有相同数量和尺寸的随机扰动对照。主要观察 unweighted log-loss 是否产生稳定的剂量—反应关系。

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
16. 先完成测试，再运行完整训练；
17. 不把提取器分数或候选排序字段作为分类输入；
18. 全局 edge index 必须转为子图局部 index；
19. 同一实验矩阵复用完全相同的数据 tuple 和评估样本；
20. 随机控制、顺序置换和 bootstrap 均使用稳定、可追踪的分层 seed；
21. 所有输出写入 evidence level、split hash、export hash、配置 hash 和代码版本；
22. Python 3.7 和 PyTorch 1.13.1 下不得使用仅由新版本提供的 API。

为避免大量小子图导致 GPU 利用率过低，允许在 collate 阶段将一个 batch 内所有有效子图扁平化，并通过全局化局部边索引和 `index_add_` 聚合消息。该优化不得改变 list-based 语义，必须用逐子图参考实现验证数值等价。也可以使用带完整 mask 的密集 padding，但不得截断任何原始节点、子图或时间窗。

# 17. 最终基线定义

\[
\boxed{
\text{两层正负分离消息传递}
+
\text{Mean/Max 子图池化}
+
\text{时间窗内子图均值聚合}
+
\text{单层单向 GRU}
+
\text{两层 MLP 分类器}
}
\]

数学形式：

\[
h_{b,k}^{(m)}
=
f_{\mathrm{signed}}(S_{b,k}^{(m)}),
\]

\[
\bar h_b^{(m)}
=
\operatorname{MeanPool}_k(h_{b,k}^{(m)}),
\]

\[
u_b^{(m)}
=
f_{\mathrm{proj}}(\bar h_b^{(m)}),
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
- `truncate_history`
- `independent_bag`
- `shuffled`
- `bag`
- 不同子图来源
- 不同统计先验
- 时间差分特征
- 子图扰动

# 18. 交叉拟合与证据等级

## 18.1 探索性全样本模式

`exploratory_in_sample` 可以使用已经由全样本训练提取器导出的子图，用于：

- 检查工程流程；
- 估计计算成本；
- 发现候选历史、顺序或结构现象；
- 为正式实验选择合理的扰动范围。

该模式的报告必须标注“监督性样本内探索”，不得使用“独立验证”“泛化证明”或“已证实传递有效”等表述。

## 18.2 确认性交叉拟合模式

`confirmatory_cross_fitted` 的外层流程为：

1. 根据冻结的 sample index 建立 group-aware 外层划分，同一 `subject_id` 只能出现在一个集合；
2. 仅使用外层训练数据训练关键子图提取器；
3. 仅使用外层训练数据选择 checkpoint、分类阈值、标准化参数和统计先验；
4. 使用冻结提取器分别导出外层训练、validation 和未见 test 子图；
5. 在训练折构造控制匹配 manifest，并将规则无修改地应用于 test；
6. 训练统一基线及其对照；
7. 只在外层未见 test 上产生该折最终预测；
8. 汇总各外层折未见预测后进行配对统计。

如果同一 subject 有多个 session，外层划分和 bootstrap 均以 subject 为单位。应报告每折类别比例和站点分布；站点外推可以通过 leave-one-site-out 作为补充敏感性分析，但不替代主要 group-aware 结果。

# 19. 理论假设与统计验收

## 19.1 主要指标

主要指标为未见样本上的 unweighted log-loss。对目标模型 \(a\) 和对照模型 \(c\)，定义：

\[
\Delta_{\mathrm{LL}}
=
\mathcal L_{\mathrm{LL}}^{(c)}
-
\mathcal L_{\mathrm{LL}}^{(a)}.
\]

\(\Delta_{\mathrm{LL}}>0\) 表示目标模型具有更低的预测不确定性。AUROC 和 balanced accuracy 为次要指标，accuracy 和 confusion matrix 为描述性指标。

## 19.2 配对统计

所有主要比较必须在同一批外层 test 样本上使用配对预测。置信区间以 subject 为重采样单位进行 bootstrap。至少报告：

- 平均效应；
- 95% 置信区间；
- 原始 \(p\)-value；
- 多主要假设下的 FDR-adjusted \(q\)-value；
- 每个随机种子和外层折的结果；
- 有效配对样本数。

不允许将匹配失败或缺失预测当作零差异。模型选择、假设方向和主要指标必须在查看最终 test 结果前冻结。

## 19.3 假设支持规则

| 理论假设 | 主要比较 | 最低解释条件 |
|---|---|---|
| 历史信息有用 | full vs current-only | full 的 paired \(\Delta_{\mathrm{LL}}\) 置信区间高于 0 |
| 状态传递有用 | full vs truncate-history | 保留更多历史产生稳定改善，或形成合理历史长度曲线 |
| 顺序信息有用 | ordered vs shuffled 与 Bag | ordered 同时优于两类无序对照 |
| 关键子图具有特异性 | key vs matched random/low-score/Top-degree | key 在相同 tuple 上稳定优于主要匹配控制 |
| 统计先验有用 | real vs permuted | real 在相同结构特征和权重分布下更优 |
| 关键结构具有扰动敏感性 | targeted vs matched random perturbation | 定向扰动造成更强且随剂量增加的退化 |

如果 ordered 仅优于 current-only，但不优于 Bag，只能支持“历史集合有用”，不能支持“顺序有用”。如果 key 仅优于 raw graph、但不优于 Top-degree，则结果更可能说明压缩或高连接强度有用，不能充分支持学习到的关键子图特异性。

# 20. 输出与审计文件

每次实验至少输出：

- 完整配置快照；
- 数据协议和 split hash；
- 提取器 checkpoint 与硬子图 export hash；
- matched-control manifest；
- 训练历史和最佳 checkpoint；
- validation 阈值；
- 每样本概率、真实标签和 subject/site 元数据；
- weighted training loss 与 unweighted evaluation log-loss；
- 配对比较结果、bootstrap 区间和 FDR 表；
- 参数量、运行时间、峰值显存和软件版本；
- evidence level 与是否存在已知泄漏风险的声明。

同一实验 ID 的产物默认不可覆盖。若显式覆盖，必须在日志中记录旧、新配置 hash。
