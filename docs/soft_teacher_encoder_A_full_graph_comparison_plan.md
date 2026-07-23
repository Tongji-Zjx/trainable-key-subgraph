# 软教师强分类编码器方案 A：完整图分类能力对比设计与实验方案

## 1. 文档目的

本文档用于解决当前 TG-SGWNet 阶段一软教师分类能力不足的问题。

当前阶段一采用：

\[
\text{Signed GNN}
\rightarrow
\text{图级池化}
\rightarrow
\text{Masked TCN}
\rightarrow
\text{分类头}
\]

对完整图或软关键图序列进行分类。但现有实验中，训练集分类表现同样较差，说明当前问题首先不是一般意义上的泛化失败，并形成如下待验证的工作假设：

\[
\boxed{
\text{当前 Signed GNN + TCN 编码器可能无法有效提取脑动态图中的分类信息}
}
\]

现阶段时间有限，因此暂不引入完整 SGW 分类分支，也暂不修改理论。当前工作的唯一目标是：

\[
\boxed{
\text{构建一个计算成本可控、非 Transformer、分类能力更强的神经编码器，}
}
\]

并先在完整图上与当前 Baseline 进行公平对比。

本文将新编码器记为：

\[
\boxed{
\text{方案 A：SGG-BiGRU-Proto}
}
\]

即：

\[
\text{Signed Edge-Gated GNN}
+
\text{BiGRU}
+
\text{轻量可学习原型码本}
\]

---

# 2. 当前要解决的问题

## 2.1 当前现象

当前软教师模型在训练集上的分类效果同样较差，说明：

- 不能简单归因于验证集泛化失败；
- 当前分类器可能连训练数据中的图结构与时间演化信息都没有充分提取；
- 若软教师分类能力不足，分类损失无法为软图提取器提供可靠的判别梯度；
- 软图提取器可能主要受到预算、拉普拉斯与 GW 保真损失驱动，而不是类别监督驱动；
- 后续硬子图候选与硬学生分类也会受到上游软教师能力限制。

当前整体依赖链为：

```text
软教师编码器能力
→ 分类梯度质量
→ 节点与边重要性分数质量
→ 软图质量
→ 硬子图质量
→ 最终分类性能
```

因此，当前优先级应当是：

\[
\boxed{
\text{先确认并提升完整图分类编码器的能力，再重新训练软图提取器}
}
\]

## 2.2 当前 Baseline 的潜在不足

当前 Baseline 为：

\[
\text{Signed GNN}
\rightarrow
\text{图池化}
\rightarrow
\text{Masked TCN}
\rightarrow
\text{MLP}
\]

其可能存在以下问题。

### 2.2.1 边特征利用不足

当前边特征包括：

\[
e_{ij}^{(m)}
=
\left[
A_{ij}^{(m)},
|A_{ij}^{(m)}|,
\Delta A_{ij}^{(m)},
|\Delta A_{ij}^{(m)}|
\right]
\]

普通图卷积往往主要将邻接矩阵作为聚合权重，未必能充分利用：

- 原始有符号连接强度；
- 连接强度绝对值；
- 相邻窗口连接变化；
- 连接变化幅度。

### 2.2.2 正负边表达能力有限

脑功能图包含正边和负边。若编码器仅使用简单归一化聚合，可能无法区分：

- 正相关连接传递的模式；
- 负相关连接传递的模式；
- 正负连接组合形成的异常结构。

### 2.2.3 时间建模能力不足

当前 TCN 使用固定卷积核和扩张率建模窗口级表示。它可能难以捕获：

- 不同长度的时间依赖；
- 不规则状态变化；
- 跨较远窗口的关联；
- 个体化的脑状态演化轨迹。

### 2.2.4 缺少跨样本原型信息

时间建模和由全体训练样本共同学习的原型表示可能对 ADHD 分类有帮助。因此，仅依赖
单样本内部的 GNN 和 TCN，可能无法充分利用跨受试者重复出现的动态模式；这一点在
本阶段只作为待检验假设，不提前视为已证实结论。

---

# 3. 本阶段实验目标

本阶段只回答一个问题：

> 在完全相同的完整图输入、数据划分和训练条件下，方案 A 是否比当前 Signed GNN + TCN Baseline 具有更强的分类能力？

本阶段不回答：

- 软图是否保留分类信息；
- 提取器是否能够学到判别关键结构；
- SGW 理论分支是否有效；
- 硬子图分类是否优于完整图；
- 最终模型是否具有良好测试泛化性能。

本阶段只比较：

\[
\boxed{
\text{Baseline}
\quad \text{vs.} \quad
\text{SGG-BiGRU-Proto}
}
\]

---

# 4. 实验输入与统一约束

## 4.1 使用完整图

本阶段不经过软图提取器。

直接使用：

\[
A_{\mathrm{input}}^{(m)}
=
A^{(m)}
\]

等价于：

\[
p_i^{(m)}=1,
\qquad
p_{ij}^{(m)}=1
\]

但推荐在代码中直接绕过提取器，而不是仍然执行评分网络。

## 4.2 输入数据

每个样本为动态图序列：

\[
\mathcal G_b
=
\left\{
G_b^{(1)},
G_b^{(2)},
\ldots,
G_b^{(M_b)}
\right\}
\]

其中：

\[
G_b^{(m)}
=
\left(
A_b^{(m)},
X_b^{(m)},
E_b^{(m)}
\right)
\]

并允许：

- 不同样本的时间片数 \(M_b\) 不同；
- 不同时间片的节点数 \(N_b^{(m)}\) 不同；
- 图中包含正边和负边；
- 节点通过稳定 ROI 名称或原始 ID 对齐；
- 不允许通过零值伪造真实时间差分。

## 4.3 节点特征

沿用当前项目已经确定的 13 维节点特征：

\[
X_b^{(m)}
\in
\mathbb R^{N_b^{(m)}\times 13}
\]

本阶段不新增或删除特征，避免把性能变化与特征工程变化混淆。

## 4.4 边特征

沿用当前 4 维边特征：

\[
e_{ij}^{(m)}
=
\left[
A_{ij}^{(m)},
|A_{ij}^{(m)}|,
\Delta A_{ij}^{(m)},
|\Delta A_{ij}^{(m)}|
\right]
\in
\mathbb R^4
\]

边是否存在仍统一使用：

\[
|A_{ij}^{(m)}|
>
\tau_{\mathrm{edge}}
\]

其中阈值必须来自冻结协议中的 `edge_presence_threshold`。当前
`configs/data_protocol_strict_theory.json` 的正式值为 `0.0`，即所有非零正边和负边均为有效边。
`.pt` 中的 `global_threshold` 只作为来源元数据保留，不得参与本实验的边存在判断。

## 4.5 本阶段关闭的模块

关闭：

\[
\mathcal L_{\mathrm{budget}}
\]

\[
\mathcal L_L
\]

\[
\mathcal L_{\mathrm{GW,id}}
\]

\[
\mathcal L_{\mathrm{supcon}}
\]

不执行：

- 软图提取；
- 硬候选生成；
- 硬子图导出；
- SGW 分类分支；
- 知识蒸馏；
- 长期静态图分支。

本阶段仅优化：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{cls}}
}
\]

---

# 5. 对比模型一：当前 Baseline

Baseline 保持当前实现不变：

```text
完整图序列
→ 当前 Signed GNN
→ 当前图池化
→ 当前 Masked TCN
→ Masked Mean + Masked Max
→ 192 维序列表示
→ 192 → 128 → 64 → 2 分类头
```

Baseline 不应为了本实验而增加：

- 新的边门控；
- 新的记忆模块；
- 新的池化方式；
- 新的时间编码器；
- 新的损失项。

只有保持 Baseline 不变，才能准确判断方案 A 的增益来自新编码器。

这里的 Baseline 必须是新增的“完整图旁路”实现：直接将协议定义的完整 signed
adjacency 输入当前 Signed GNN，完全不实例化或执行节点/边评分器。现有
`classification_only` 消融仍会构造 \(A_{ij}p_ip_jp_{ij}\)，不能作为完整图 Baseline。

---

# 6. 方案 A：SGG-BiGRU-Proto

## 6.1 整体结构

方案 A 的完整计算流程为：

```text
每个完整图窗口
    │
    ├── 13 维节点特征
    ├── 4 维边特征
    └── 原始 signed adjacency
            │
            ▼
3 层 Signed Edge-Gated Residual GNN
            │
            ▼
Mean + Max + Gated Attention Pooling
            │
            ▼
96 维窗口表示
            │
            ▼
1 层 BiGRU
            │
            ▼
Masked Mean + Masked Max
            │
            ▼
192 维序列表示
            │
            ▼
16 槽可学习原型码本
            │
            ▼
投影为 192 维教师表示
            │
            ▼
192 → 128 → 64 → 2 分类头
```

数学形式为：

\[
G^{(1:M)}
\rightarrow
\operatorname{SGG}
\rightarrow
\{g^{(1)},\ldots,g^{(M)}\}
\rightarrow
\operatorname{BiGRU}
\rightarrow
h_{\mathrm{seq}}
\rightarrow
\operatorname{PrototypeCodebook}
\rightarrow
h_{\mathrm{teacher}}
\rightarrow
\hat y
\]

---

# 7. Signed Edge-Gated Residual GNN

## 7.1 节点输入投影

将 13 维节点特征投影至 64 维：

\[
h_i^{(0)}
=
\operatorname{GELU}
\left(
W_xx_i+b_x
\right)
\in
\mathbb R^{64}
\]

建议配置：

```yaml
node_input_dim: 13
node_hidden_dim: 64
activation: gelu
```

## 7.2 边输入投影

将 4 维边特征投影至 32 维：

\[
u_{ij}
=
\operatorname{GELU}
\left(
W_ee_{ij}+b_e
\right)
\in
\mathbb R^{32}
\]

建议配置：

```yaml
edge_input_dim: 4
edge_hidden_dim: 32
```

## 7.3 边门控

在第 \(\ell\) 层，对每条有效边计算门控：

\[
\alpha_{ij}^{(\ell)}
=
\sigma
\left(
MLP_g^{(\ell)}
\left[
h_i^{(\ell)}+h_j^{(\ell)};
\left|h_i^{(\ell)}-h_j^{(\ell)}\right|;
u_{ij}
\right]
\right)
\]

其中：

\[
\alpha_{ij}^{(\ell)}
\in(0,1)
\]

表示该边在当前消息传递层中的传播强度。

该门控属于分类编码器内部机制，与软图提取器的边保留分数 \(p_{ij}\) 不同。
上述输入关于 \(i,j\) 对称，因此无向图上必须满足
\(\alpha_{ij}^{(\ell)}=\alpha_{ji}^{(\ell)}\)。禁止使用有序拼接
\([h_i;h_j;u_{ij}]\) 后直接产生不对称门控。

## 7.4 正负边分离

定义：

\[
A_{ij}^{+}
=
\max(A_{ij},0)
\]

\[
A_{ij}^{-}
=
\max(-A_{ij},0)
\]

分别计算正边和负边消息。

正边消息：

\[
m_{i,+}^{(\ell)}
=
\frac{
\sum_j
A_{ij}^{+}
\alpha_{ij}^{(\ell)}
W_+^{(\ell)}
h_j^{(\ell)}
}{
\sum_j
A_{ij}^{+}
\alpha_{ij}^{(\ell)}
+\epsilon
}
\]

负边消息：

\[
m_{i,-}^{(\ell)}
=
\frac{
\sum_j
A_{ij}^{-}
\alpha_{ij}^{(\ell)}
W_-^{(\ell)}
h_j^{(\ell)}
}{
\sum_j
A_{ij}^{-}
\alpha_{ij}^{(\ell)}
+\epsilon
}
\]

注意：

- \(A^-_{ij}\) 表示负边强度绝对值；
- 正边与负边使用不同的可学习参数；
- 原始边符号不能被覆盖；
- 非边、padding 边和自环不参与门控与消息聚合。

## 7.5 节点更新

节点候选表示：

\[
\widetilde h_i^{(\ell+1)}
=
\operatorname{GELU}
\left(
W_0^{(\ell)}h_i^{(\ell)}
+
m_{i,+}^{(\ell)}
+
m_{i,-}^{(\ell)}
\right)
\]

残差更新：

\[
h_i^{(\ell+1)}
=
\operatorname{LayerNorm}
\left(
h_i^{(\ell)}
+
\operatorname{Dropout}
\left(
\widetilde h_i^{(\ell+1)}
\right)
\right)
\]

建议使用 3 层：

```yaml
signed_edge_gated_gnn:
  num_layers: 3
  hidden_dim: 64
  dropout: 0.15
  residual: true
  normalization: layer_norm
  activation: gelu
```

实现时必须向量化计算有效无向边，禁止 Python 逐边循环，也应避免为每层永久保留
\([N,N,160]\) 的巨大中间张量。当前正式边阈值为零，图可能较稠密，因此正式训练前
必须记录单窗口和整样本的耗时及峰值显存。

---

# 8. 窗口级图池化

对于时间片 \(m\) 的节点表示，使用三路池化。

## 8.1 Mean Pooling

\[
g_{\mathrm{mean}}^{(m)}
=
\operatorname{MeanMask}_i
h_i^{(m)}
\]

## 8.2 Max Pooling

\[
g_{\mathrm{max}}^{(m)}
=
\operatorname{MaxMask}_i
h_i^{(m)}
\]

## 8.3 Gated Attention Pooling

注意力分数：

\[
s_i^{(m)}
=
w_a^\top
\tanh
\left(
W_ah_i^{(m)}
\right)
\]

归一化权重：

\[
\beta_i^{(m)}
=
\frac{
\exp(s_i^{(m)})
}{
\sum_{j\in V^{(m)}}
\exp(s_j^{(m)})
}
\]

注意力池化：

\[
g_{\mathrm{attn}}^{(m)}
=
\sum_i
\beta_i^{(m)}
h_i^{(m)}
\]

三路结果拼接：

\[
g_{\mathrm{raw}}^{(m)}
=
\left[
g_{\mathrm{mean}}^{(m)};
g_{\mathrm{max}}^{(m)};
g_{\mathrm{attn}}^{(m)}
\right]
\in
\mathbb R^{192}
\]

再投影为 96 维窗口表示：

\[
g^{(m)}
=
\operatorname{LayerNorm}
\left(
W_pg_{\mathrm{raw}}^{(m)}
+b_p
\right)
\in
\mathbb R^{96}
\]

建议配置：

```yaml
graph_pooling:
  methods:
    - mean
    - max
    - gated_attention
  concatenated_dim: 192
  output_dim: 96
  dropout: 0.15
```

---

# 9. BiGRU 时间编码器

将窗口表示序列：

\[
g^{(1)},g^{(2)},\ldots,g^{(M_b)}
\]

输入一层双向 GRU。

建议配置：

```yaml
bigru:
  input_dim: 96
  hidden_dim_per_direction: 48
  num_layers: 1
  bidirectional: true
  dropout: 0.0
```

每个时间步输出：

\[
o^{(m)}
\in
\mathbb R^{96}
\]

对有效时间片进行 masked pooling：

\[
h_{\mathrm{mean}}
=
\operatorname{MeanMask}_m
o^{(m)}
\]

\[
h_{\mathrm{max}}
=
\operatorname{MaxMask}_m
o^{(m)}
\]

最终序列表示：

\[
\boxed{
h_{\mathrm{seq}}
=
\left[
h_{\mathrm{mean}};
h_{\mathrm{max}}
\right]
\in
\mathbb R^{192}
}
\]

实现时优先使用：

```python
torch.nn.utils.rnn.pack_padded_sequence
```

并确保：

- 输入序列按真实长度打包；
- padding 时间片不参与 GRU；
- padding 时间片不参与 mean/max pooling；
- 恢复顺序后样本与标签仍一一对应。

---

BiGRU 使用完整序列的前后文，只适用于本项目当前的离线整序列二分类。其结果不得
解释为严格因果或在线单向时间传递。

---

# 10. 轻量可学习原型码本

## 10.1 设计原则

为了控制开发时间与训练复杂度，本阶段不实现动态写入式跨样本记忆，而采用可学习原型参数：

\[
M_{\mathrm{proto}}
\in
\mathbb R^{K\times d_m}
\]

默认：

\[
K=16,
\qquad
d_m=64
\]

这些原型作为普通参数，通过反向传播更新。
它们是由全部训练样本共同学习的静态参数码本，不保存具体受试者状态，也不在推理时
读取其他样本。因此实验只能检验“原型码本增强”的整体效果，不能表述为验证了动态
跨样本记忆机制。

## 10.2 查询与读取

将序列表示投影为查询：

\[
q
=
W_qh_{\mathrm{seq}}
\in
\mathbb R^{64}
\]

计算原型相似度：

\[
\alpha
=
\operatorname{softmax}
\left(
\frac{
qM_{\mathrm{proto}}^\top
}{
\sqrt{64}
}
\right)
\in
\mathbb R^{16}
\]

读取记忆：

\[
r
=
\alpha M_{\mathrm{proto}}
\in
\mathbb R^{64}
\]

融合：

\[
h_{\mathrm{teacher}}
=
\operatorname{LayerNorm}
\left(
W_f
\left[
h_{\mathrm{seq}};
r
\right]
+b_f
\right)
\in
\mathbb R^{192}
\]

建议配置：

```yaml
prototype_codebook:
  num_prototypes: 16
  prototype_dim: 64
  stateful_write: false
  output_dim: 192
```

---

# 11. 分类头

Baseline 和方案 A 使用完全相同的当前正式分类头：

\[
192
\rightarrow
128
\rightarrow
64
\rightarrow
2
\]

建议配置：

```yaml
classifier:
  input_dim: 192
  hidden_dims:
    - 128
    - 64
  activation: gelu
  dropout: 0.20
  output_dim: 2
```

输出 logits：

\[
z
=
MLP_{\mathrm{cls}}
\left(
h_{\mathrm{teacher}}
\right)
\]

概率：

\[
\hat p
=
\operatorname{softmax}(z)
\]

---

# 12. 训练损失

二分类采用加权交叉熵：

\[
\mathcal L_{\mathrm{cls}}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\sum_{c=0}^{1}
w_c
\mathbf 1(y_b=c)
\log \hat p_{b,c}
\]

其中类别权重 \(w_c\) 只能根据当前训练集计算，不能使用验证集或测试集标签比例。

本阶段总损失为：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{cls}}
}
\]

---

# 13. 实验步骤

## 13.1 步骤一：统一模型接口

保留两个编码器：

```python
class GraphSequenceClassifier(nn.Module):
    def __init__(self, encoder_type, config):
        super().__init__()

        if encoder_type == "signed_gnn_tcn":
            self.encoder = SignedGNNTCNEncoder(config)

        elif encoder_type == "sgg_bigru_proto":
            self.encoder = SignedGatedBiGRUPrototypeEncoder(config)

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.classifier = ClassificationHead(
            input_dim=192,
            hidden_dims=(128, 64),
            output_dim=2,
        )

    def forward(self, batch):
        representation = self.encoder(batch)
        logits = self.classifier(representation)

        return {
            "logits": logits,
            "representation": representation,
        }
```

本阶段配置：

```yaml
experiment:
  graph_mode: full_graph
  train_extractor: false
  compute_budget_loss: false
  compute_laplacian_loss: false
  compute_gw_loss: false
  compute_supcon_loss: false
```

## 13.2 步骤二：单元与前向检查

在正式训练前完成以下检查。

### Signed 邻接检查

```python
assert adjacency.shape[-1] == adjacency.shape[-2]
assert torch.allclose(adjacency, adjacency.T, atol=1e-6)
assert torch.all(adjacency.diag() == 0)
```

### 正负边检查

```python
positive = torch.clamp(adjacency, min=0)
negative = torch.clamp(-adjacency, min=0)

assert torch.all(positive >= 0)
assert torch.all(negative >= 0)
assert torch.allclose(adjacency, positive - negative)
```

### Mask 检查

```python
assert padded_nodes_do_not_affect_pooling
assert padded_windows_do_not_affect_bigru_output
assert sequence_lengths_match_time_mask
```

### 输出维度检查

```python
assert window_embedding.shape[-1] == 96
assert sequence_embedding.shape[-1] == 192
assert teacher_embedding.shape[-1] == 192
assert logits.shape[-1] == 2
```

### 梯度检查

执行一次前向与反向传播后确认：

```python
assert gnn_parameters_have_grad
assert bigru_parameters_have_grad
assert prototype_codebook_parameters_have_grad
assert classifier_parameters_have_grad
```

## 13.3 步骤三：16 样本过拟合测试

### 目的

判断模型是否具备基本学习能力，并快速排除：

- 标签错位；
- mask 错误；
- GRU 长度错误；
- 梯度中断；
- 优化器漏掉参数；
- 分类头实现错误。

### 数据

从训练集中固定选取 16 个样本：

- 两类都必须存在；
- 尽量类别平衡；
- 样本列表固定并写入日志。

### 配置

```yaml
overfit_test:
  num_samples: 16
  max_epochs: 200
  early_stopping: false
  dropout: 0.0
  weight_decay: 0.0
  loss: weighted_cross_entropy
```

### 验收条件

方案 A 应达到：

\[
\boxed{
\text{训练准确率约 95\% 以上}
}
\]

同时要求训练 AUROC 不低于 \(0.99\)，且 classification loss 明显下降并继续趋近于零。

若方案 A 无法拟合 16 个样本，不进入下一步，优先排查：

1. 样本与标签是否错位；
2. `pack_padded_sequence` 是否使用正确长度；
3. padding 是否参与池化；
4. 分类头是否加入优化器；
5. GNN、BiGRU 和原型码本是否收到梯度；
6. 是否出现 NaN 或 Inf；
7. 是否存在空图或全零图；
8. 边阈值是否错误地删除了大量有效边。

## 13.4 步骤四：固定划分上的主对比

为了节省时间，本阶段只使用当前项目中的一个固定训练—验证划分。

禁止：

- 根据测试集选择模型；
- 查看测试集结果后修改结构；
- 使用测试标签决定超参数。

### 对比模型

```text
Model 1: 当前 Signed GNN + TCN Baseline
Model 2: SGG-BiGRU-Proto
```

### 随机种子

```yaml
seeds:
  - 42
  - 43
  - 44
```

### 统一训练配置

```yaml
optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

training:
  max_epochs: 60
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  batch_size: use_current_stable_value

scheduler:
  name: reduce_lr_on_plateau
  monitor: validation_unweighted_loss
  factor: 0.5
  patience: 4
  min_learning_rate: 0.00001
```

不进行网格搜索。

## 13.5 步骤五：Checkpoint 选择

主选择指标：

\[
\text{Validation AUROC}
\]

若多个 epoch 的 Validation AUROC 相同，则使用：

\[
\text{Validation Unweighted Loss}
\]

较低者作为次级依据。Balanced Accuracy 继续报告，但不用于本阶段的学习率调度和
checkpoint 选择，避免固定 0.5 分类阈值导致长期并列。

不得使用：

- 训练集最佳值选择 checkpoint；
- 测试集最佳值选择 checkpoint；
- 多次查看测试集后回调模型。

## 13.6 步骤六：记录指标

每个 epoch 记录：

- Train Loss；
- Train Balanced Accuracy；
- Train AUROC；
- Validation Loss；
- Validation Balanced Accuracy；
- Validation AUROC；
- 当前学习率。
- 原型注意力熵、各槽位平均使用率和最大槽位占比（仅方案 A）。

每个随机种子训练结束后记录：

- 最佳 epoch；
- 最佳验证 Balanced Accuracy；
- 对应验证 AUROC；
- 训练集 Balanced Accuracy；
- 训练集 AUROC；
- 参数量；
- 平均单 epoch 时间；
- 峰值 GPU 显存。

表中的 Train/Validation 指标必须在训练结束后重新加载最佳 checkpoint，并在
`model.eval()` 下分别完整评估；不得直接使用含 Dropout 的训练过程统计值。

---

# 14. 实验结果表

最终只需要生成一张主表：

| 模型 | Train BA | Train AUC | Val BA | Val AUC | 参数量 | 时间/epoch | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Signed GNN + TCN | Mean ± Std | Mean ± Std | Mean ± Std | Mean ± Std | 数值 | 数值 | 数值 |
| SGG-BiGRU-Proto | Mean ± Std | Mean ± Std | Mean ± Std | Mean ± Std | 数值 | 数值 | 数值 |

并额外保留三个种子的明细：

| 模型 | Seed | Best Epoch | Train BA | Train AUC | Val BA | Val AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 42 |  |  |  |  |  |
| Baseline | 43 |  |  |  |  |  |
| Baseline | 44 |  |  |  |  |  |
| SGG-BiGRU-Proto | 42 |  |  |  |  |  |
| SGG-BiGRU-Proto | 43 |  |  |  |  |  |
| SGG-BiGRU-Proto | 44 |  |  |  |  |  |

---

# 15. 方案 A 的采用标准

若方案 A 同时满足以下条件，则将其替换为后续软教师编码器。

## 15.1 基本可训练性

16 样本过拟合测试达到：

\[
\text{Train Accuracy}
\geq
95\%
\]

## 15.2 完整训练集拟合能力改善

方案 A 的训练集 Balanced Accuracy 应明显高于 Baseline：

\[
\text{Train BA}_{A}
>
\text{Train BA}_{Baseline}
\]

若训练集仍接近随机水平，则说明：

- 方案 A 仍然无法有效提取信息；
- 或数据、标签、训练管线存在实现问题。

## 15.3 验证集性能改善

采用一个简单的工程判定标准：

\[
\boxed{
\text{Val BA 或 Val AUC 平均提高约 3 个百分点以上}
}
\]

同时：

- 3 个随机种子中至少 2 个优于 Baseline；
- 不能只依赖单个异常种子；
- 训练结果不能频繁出现 NaN 或严重不稳定。

该 3 个百分点不是理论阈值，只是为了避免因随机波动替换整个编码器。

## 15.4 计算成本可接受

建议：

\[
\boxed{
\text{方案 A 单 epoch 时间不超过 Baseline 的约 2 倍}
}
\]

若分类性能提升明显，可以接受适度增加显存。

---

# 16. 本阶段不做的内容

为了控制时间，本轮不做：

- 多个 GNN 层数搜索；
- 多个隐藏维度搜索；
- 不同原型槽数量搜索；
- 不同 BiGRU 层数搜索；
- 原型码本消融；
- Edge Gate 消融；
- 精确 SGW 分类分支；
- 长期静态图分支；
- 硬子图导出；
- 知识蒸馏；
- 五折完整测试；
- 大规模超参数搜索。

本轮唯一目标是：

\[
\boxed{
\text{确认 SGG-BiGRU-Proto 是否比当前 Signed GNN + TCN 更能从完整图中提取分类信息}
}
\]

---

# 17. 后续流程

若方案 A 在完整图上胜出，后续按以下顺序接回软教师：

```text
完整图上预训练方案 A
→ 保存最佳完整图编码器参数
→ 接入软图提取器
→ 初始化 p_i、p_ij 接近 1
→ 初期冻结或低学习率微调方案 A
→ 主要训练节点和边评分器
→ 逐步增加预算、拉普拉斯和 GW 损失
→ 最后联合微调
```

建议初始阶段：

```yaml
soft_teacher_stage:
  encoder_pretrained: true
  extractor_initial_scores_near_one: true
  freeze_encoder_epochs: 5
  encoder_learning_rate_scale: 0.1
  gradually_enable_structure_losses: true
```

---

# 18. 最终结论

当前需要解决的核心问题是：

\[
\boxed{
\text{当前 Signed GNN + TCN 编码能力不足，导致软教师无法有效分类，}
}
\]

进而使软图提取器缺少可靠的判别监督。

方案 A 使用：

\[
\boxed{
\text{Signed Edge-Gated Residual GNN}
+
\text{BiGRU}
+
\text{轻量可学习原型码本}
}
\]

分别增强：

- 带符号边和边特征的空间编码；
- 动态图序列的时间依赖建模；
- 跨样本重复模式的原型表示。

本阶段只在完整图上比较：

\[
\boxed{
\text{当前 Baseline}
\quad\text{vs.}\quad
\text{SGG-BiGRU-Proto}
}
\]

采用最简流程：

\[
\boxed{
\text{16 样本过拟合测试}
\rightarrow
\text{固定划分、3 个种子对比}
\rightarrow
\text{性能与成本判定}
}
\]

若方案 A 在完整图上明显优于 Baseline，再将其接入软图提取器，避免在编码器本身尚未验证时同时调试提取器和分类器。
