# Exact-STSE 与去空间坐标消融实验设计

## 文档状态

- **版本**：V1.0
- **目标**：严格复现原论文的短时状态编码器（Short-Term State Encoder, STSE），并增加“去掉空间坐标”的消融实验，用于验证原论文 STSE 的分类能力是否依赖 ROI 空间信息。
- **实验范围**：
  - 使用完整动态图序列；
  - 不加入关键子图提取；
  - 不加入谱–GW 分支；
  - 不加入 GNN、TCN、GRU、Transformer 或记忆模块；
  - 只比较原始 STSE 与去空间坐标 STSE；
  - 其余结构、数据划分和训练配置保持一致。

---

# 1. 实验目的

原论文在 ADHD-200 数据集上报告，仅使用短时状态编码器（STSE）即可达到：

\[
\mathrm{ACC}=65.69\%\pm0.69\%
\]

\[
\mathrm{AUC}=64.65\%\pm4.05\%
\]

STSE 使用的节点输入包括：

\[
\deg_i^{(m)},
\quad
\operatorname{coord}_i,
\quad
\operatorname{neighcoord}_i^{(m)},
\quad
\operatorname{emb}(c_i^{(m)}),
\quad
\Delta\deg_i^{(m)}.
\]

其中：

- \(\operatorname{coord}_i\) 是节点自身的三维 MNI 坐标；
- \(\operatorname{neighcoord}_i^{(m)}\) 是由邻接矩阵和邻居坐标计算的邻居空间中心。

因此，原论文 STSE 的分类能力可能不仅来自图结构和动态变化，也可能来自空间解剖先验。

本实验需要回答：

\[
\boxed{
\text{去掉节点坐标和邻居坐标后，STSE 的分类性能是否明显下降？}
}
\]

---

# 2. 对比模型

## 2.1 Exact-STSE

严格复现原论文短时状态编码器。

节点输入：

\[
f_i^{(m)}
=
\left[
\deg_i^{(m)};
\operatorname{coord}_i;
\operatorname{neighcoord}_i^{(m)};
\operatorname{emb}(c_i^{(m)});
\Delta\deg_i^{(m)}
\right].
\]

## 2.2 Exact-STSE-NoCoord

删除所有空间信息。

节点输入：

\[
\boxed{
f_{i,\mathrm{nc}}^{(m)}
=
\left[
\deg_i^{(m)};
\operatorname{emb}(c_i^{(m)});
\Delta\deg_i^{(m)}
\right].
}
\]

删除：

\[
\operatorname{coord}_i
\]

和：

\[
\operatorname{neighcoord}_i^{(m)}.
\]

除输入特征维度变化外，其余网络结构和训练方式保持一致。

---

# 3. 为什么必须同时删除两类坐标

只删除节点自身坐标 \(\operatorname{coord}_i\)，但保留邻居空间中心 \(\operatorname{neighcoord}_i^{(m)}\)，不能构成真正的去空间坐标消融。

因为：

\[
\operatorname{neighcoord}_i^{(m)}
=
\widehat A_i^{(m)}X_{\mathrm{coords}}
\]

仍然显式依赖所有邻居节点的空间坐标。

因此：

\[
\boxed{
\text{去空间坐标消融必须同时删除 coord 和 neighcoord。}
}
\]

---

# 4. 输入数据

对第 \(m\) 个时间窗口，Exact-STSE 输入：

\[
A_{\mathrm{curr}}^{(m)}\in\mathbb R^{N\times N},
\]

\[
A_{\mathrm{prev}}^{(m-1)}\in\mathbb R^{N\times N},
\]

\[
X_{\mathrm{coords}}\in\mathbb R^{N\times3},
\]

\[
c^{(m)}\in\mathbb Z^N.
\]

其中：

- \(A_{\mathrm{curr}}^{(m)}\)：当前窗口带符号邻接矩阵；
- \(A_{\mathrm{prev}}^{(m-1)}\)：上一窗口邻接矩阵；
- \(X_{\mathrm{coords}}\)：节点三维 MNI 坐标；
- \(c^{(m)}\)：当前窗口社区编号。

Exact-STSE-NoCoord 仍可接收 `coordinates` 参数以统一数据接口，但模型内部不读取该数据。

---

# 5. 第一个窗口的处理

对于第一个窗口：

\[
A_{\mathrm{prev}}^{(0)}
=
A_{\mathrm{curr}}^{(1)}.
\]

因此：

\[
\Delta\deg_i^{(1)}=0.
\]

这样与原论文伪代码保持一致。

---

# 6. 节点结构特征

## 6.1 当前窗口绝对加权度

\[
\deg_i^{(m)}
=
\sum_{j=1}^{N}
\left|
A_{ij}^{(m)}
\right|.
\]

这里使用的是绝对边权之和，而不是无权节点度。

## 6.2 上一窗口绝对加权度

\[
\deg_i^{(m-1)}
=
\sum_{j=1}^{N}
\left|
A_{ij}^{(m-1)}
\right|.
\]

## 6.3 节点度变化

\[
\Delta\deg_i^{(m)}
=
\deg_i^{(m)}-
\deg_i^{(m-1)}.
\]

该特征使 STSE 在不使用后续时间编码器的情况下，仍然能够利用相邻窗口的一阶动态信息。

---

# 7. 空间特征

## 7.1 节点自身坐标

\[
\operatorname{coord}_i\in\mathbb R^3.
\]

该坐标表示节点在标准脑空间中的位置。

## 7.2 邻居空间中心

原论文先定义：

\[
\widehat A_{ij}^{(m)}
=
\frac{A_{ij}^{(m)}}{\deg_i^{(m)}+\varepsilon}.
\]

再计算：

\[
\operatorname{neighcoord}_i^{(m)}
=
\sum_{j=1}^{N}
\widehat A_{ij}^{(m)}
\operatorname{coord}_j.
\]

矩阵形式为：

\[
\operatorname{NeighCoord}^{(m)}
=
\widehat A^{(m)}X_{\mathrm{coords}}.
\]

严格复现时：

- 分子使用带符号邻接矩阵 \(A^{(m)}\)；
- 分母使用绝对加权度；
- 不将分子替换为 \(|A^{(m)}|\)。

---

# 8. 社区嵌入

社区编号映射为：

\[
E_{\mathrm{comm},i}^{(m)}
=
\operatorname{Embedding}(c_i^{(m)}+1).
\]

其中：

- 原始社区编号 \(-1\) 映射到索引 0；
- 索引 0 对应专门的空社区向量；
- 其他社区编号整体加 1。

Exact-STSE 和 Exact-STSE-NoCoord 必须使用相同的社区词表大小、社区嵌入维度、初始化方式和优化方式。

---

# 9. 节点输入维度

设社区嵌入维度为 \(d_{\mathrm{emb}}\)。

## 9.1 Exact-STSE

\[
d_{\mathrm{in}}^{\mathrm{coord}}
=
d_{\mathrm{emb}}+8.
\]

## 9.2 Exact-STSE-NoCoord

\[
d_{\mathrm{in}}^{\mathrm{no\ coord}}
=
d_{\mathrm{emb}}+2.
\]

---

# 10. STSE节点编码器

两组模型均采用相同的编码结构。

## 10.1 输入LayerNorm

\[
\widetilde f_i^{(m)}
=
\operatorname{LayerNorm}(f_i^{(m)}).
\]

## 10.2 线性投影

\[
h_i^{(m)}
=
W_1\widetilde f_i^{(m)}+b_1
\in\mathbb R^d.
\]

## 10.3 残差前馈网络

按照原论文附录伪代码，使用 GELU：

\[
h_{i,\mathrm{ffn}}^{(m)}
=
W_3\operatorname{GELU}(W_2h_i^{(m)}+b_2)+b_3.
\]

残差输出：

\[
x_i^{(m)}
=
\operatorname{LayerNorm}
\left(
h_i^{(m)}+h_{i,\mathrm{ffn}}^{(m)}
\right).
\]

严格复现版不加入第二残差块、图卷积、GAT、节点注意力、边编码器或正负边双通道神经消息传递。

---

# 11. 窗口级节点池化

原论文采用单一节点平均池化：

\[
\boxed{
z^{(m)}
=
\frac{1}{N}
\sum_{i=1}^{N}x_i^{(m)}
}
\]

得到窗口表示：

\[
z^{(m)}\in\mathbb R^d.
\]

若数据存在变长节点，则使用 mask-aware mean：

\[
z^{(m)}
=
\frac{
\sum_iM_i^{(m)}x_i^{(m)}
}{
\sum_iM_i^{(m)}
}.
\]

不能在严格复现版中替换成 Mean+Max、Mean+Std、Attention Pooling 或 Gated Pooling。

---

# 12. 样本级时间聚合

原论文没有明确说明 Table 2 中 STSE-only 如何将多个窗口表示汇总为样本级表示。

本复现采用最小透明假设：

\[
\boxed{
z_{\mathrm{subject}}
=
\operatorname{MaskedMean}_m z^{(m)}
}
\]

即：

\[
z_{\mathrm{subject}}
=
\frac{
\sum_mM_m^{\mathrm{time}}z^{(m)}
}{
\sum_mM_m^{\mathrm{time}}
}.
\]

该部分必须在实验报告中标注为“原论文未明确给出的样本级聚合假设”。

---

# 13. 分类头

使用原论文风格的两层 ReLU + Dropout 分类头：

\[
h_1
=
\operatorname{Dropout}
\left(
\operatorname{ReLU}(W_1z_{\mathrm{subject}}+b_1)
\right),
\]

\[
h_2
=
\operatorname{Dropout}
\left(
\operatorname{ReLU}(W_2h_1+b_2)
\right),
\]

\[
z_{\mathrm{logit}}
=
W_oh_2+b_o.
\]

最终输出：

\[
z_{\mathrm{logit}}\in\mathbb R^2.
\]

---

# 14. 统一代码架构

建议使用一个编码器类，通过 `use_coordinates` 控制消融。

```python
import torch
from torch import nn
from torch.nn import functional as F


class ExactSTSEWindowEncoder(nn.Module):
    def __init__(
        self,
        community_vocab_size: int,
        community_embedding_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        use_coordinates: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.use_coordinates = use_coordinates
        self.eps = eps

        self.community_embedding = nn.Embedding(
            num_embeddings=community_vocab_size,
            embedding_dim=community_embedding_dim,
        )

        input_dim = community_embedding_dim + 2
        if use_coordinates:
            input_dim += 6

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.ffn_linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        current_adj: torch.Tensor,
        previous_adj: torch.Tensor,
        coordinates: torch.Tensor,
        communities: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        degree = current_adj.abs().sum(dim=-1)
        previous_degree = previous_adj.abs().sum(dim=-1)
        delta_degree = degree - previous_degree

        community_index = communities + 1
        community_embedding = self.community_embedding(community_index)

        feature_parts = [degree.unsqueeze(-1)]

        if self.use_coordinates:
            normalized_adj = current_adj / (
                degree.unsqueeze(-1) + self.eps
            )
            neighbor_coordinates = torch.matmul(
                normalized_adj,
                coordinates,
            )
            feature_parts.extend([
                coordinates,
                neighbor_coordinates,
            ])

        feature_parts.extend([
            community_embedding,
            delta_degree.unsqueeze(-1),
        ])

        features = torch.cat(feature_parts, dim=-1)
        hidden = self.input_projection(self.input_norm(features))
        residual = self.ffn_linear2(
            F.gelu(self.ffn_linear1(hidden))
        )
        node_output = self.output_norm(hidden + residual)

        if node_mask is None:
            return node_output.mean(dim=1)

        valid = node_mask.unsqueeze(-1).to(node_output.dtype)
        return (
            (node_output * valid).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1.0)
        )
```

---

# 15. 配置文件

## 15.1 Exact-STSE

```yaml
model:
  name: exact_stse
  use_coordinates: true

exact_stse:
  community_embedding_dim: 16
  hidden_dim: 64
  ffn_dim: 128

classifier:
  hidden_dims: [64, 32]
  dropout: 0.20
  output_dim: 2
```

## 15.2 Exact-STSE-NoCoord

```yaml
model:
  name: exact_stse_no_coord
  use_coordinates: false

exact_stse:
  community_embedding_dim: 16
  hidden_dim: 64
  ffn_dim: 128

classifier:
  hidden_dims: [64, 32]
  dropout: 0.20
  output_dim: 2
```

除 `use_coordinates` 外，其余配置保持一致。

---

# 16. 为什么不采用坐标置零

不建议通过将坐标和邻居坐标置零来实现去坐标消融。

原因是输入 LayerNorm 会对整个特征向量计算均值和方差，六个恒为零的通道仍会改变：

\[
\operatorname{Mean}(f_i)
\]

和：

\[
\operatorname{Var}(f_i).
\]

因此：

\[
\boxed{
\text{坐标置零不等价于删除坐标通道。}
}
\]

主实验应直接删除空间特征通道。

---

# 17. 实验分组

## 17.1 主消融

| 编号 | 模型 | 节点坐标 | 邻居坐标 | 社区嵌入 |
|---|---|---:|---:|---:|
| R0 | Exact-STSE | ✓ | ✓ | ✓ |
| R0-NC | Exact-STSE-NoCoord | × | × | ✓ |

## 17.2 可选统计增强

若实现社区异常和图统计分支，则保持成对比较：

| 编号 | 模型 |
|---|---|
| R1 | Exact-STSE+Stats |
| R1-NC | Exact-STSE-NoCoord+Stats |

不能比较 Exact-STSE+Stats 与 Exact-STSE-NoCoord，因为这会同时改变坐标和统计分支，无法进行单变量归因。

---

# 18. 实验步骤

## 18.1 步骤一：前向与维度检查

Exact-STSE：

```python
assert input_feature_dim == community_embedding_dim + 8
```

Exact-STSE-NoCoord：

```python
assert input_feature_dim == community_embedding_dim + 2
```

共同检查：

```python
assert window_embedding.shape[-1] == hidden_dim
assert logits.shape[-1] == 2
assert padded_nodes_do_not_affect_window_embedding
assert padded_windows_do_not_affect_subject_embedding
```

## 18.2 步骤二：16样本最小过拟合

使用同一批固定训练样本，两类都必须存在。

```yaml
overfit_test:
  num_samples: 16
  max_epochs: 200
  early_stopping: false
  dropout: 0.0
  weight_decay: 0.0
```

要求两组均达到：

\[
\text{Train Accuracy}\geq95\%.
\]

若 NoCoord 无法完成最小过拟合，应先检查输入维度、LayerNorm维度、社区索引、分类头参数、时间mask和标签对应，不能直接将失败归因于坐标缺失。

## 18.3 步骤三：固定划分主实验

使用完全相同的：

- `sample_index.csv`；
- `splits.csv`；
- 训练集和验证集；
- 类别权重；
- batch size；
- 优化器；
- 学习率；
- checkpoint规则。

建议随机种子：

```yaml
seeds: [42, 43, 44]
```

唯一变量是：

```yaml
use_coordinates: true / false
```

## 18.4 统一训练配置

```yaml
optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

training:
  max_epochs: 80
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
  batch_size: use_current_stable_value

scheduler:
  name: reduce_lr_on_plateau
  factor: 0.5
  patience: 5
  min_learning_rate: 0.00001
```

Checkpoint主指标：

\[
\text{Validation Balanced Accuracy}.
\]

次级指标：

\[
\text{Validation AUROC}.
\]

---

# 19. 记录指标

每个epoch记录：

- Train Loss；
- Train Accuracy；
- Train Balanced Accuracy；
- Train AUROC；
- Validation Loss；
- Validation Accuracy；
- Validation Balanced Accuracy；
- Validation AUROC；
- 当前学习率。

每次训练结束后记录：

- 最佳epoch；
- 参数量；
- 平均单epoch时间；
- 峰值显存；
- 混淆矩阵；
- Sensitivity；
- Specificity；
- F1。

---

# 20. 结果表

## 20.1 三种子汇总

| 模型 | Train BA | Train AUC | Val BA | Val AUC | 参数量 |
|---|---:|---:|---:|---:|---:|
| Exact-STSE | Mean ± Std | Mean ± Std | Mean ± Std | Mean ± Std |  |
| Exact-STSE-NoCoord | Mean ± Std | Mean ± Std | Mean ± Std | Mean ± Std |  |

## 20.2 单种子明细

| 模型 | Seed | Best Epoch | Train BA | Train AUC | Val BA | Val AUC |
|---|---:|---:|---:|---:|---:|---:|
| Exact-STSE | 42 |  |  |  |  |  |
| Exact-STSE | 43 |  |  |  |  |  |
| Exact-STSE | 44 |  |  |  |  |  |
| Exact-STSE-NoCoord | 42 |  |  |  |  |  |
| Exact-STSE-NoCoord | 43 |  |  |  |  |  |
| Exact-STSE-NoCoord | 44 |  |  |  |  |  |

---

# 21. 结果解释

## 21.1 去坐标后明显下降

若：

\[
\mathrm{AUC}_{\mathrm{coord}}
-
\mathrm{AUC}_{\mathrm{no\ coord}}
\geq3
\text{ 个百分点}
\]

并且三个随机种子中至少两个出现一致下降，则支持：

\[
\boxed{
\text{原STSE的分类能力明显依赖空间解剖信息。}
}
\]

此时不能将原文 STSE-only 的性能全部归因于图拓扑和动态变化。

## 21.2 两组表现接近

若两者差异很小并落在随机种子波动范围内，则说明：

\[
\deg,
\quad
\Delta\deg,
\quad
\operatorname{emb}(c)
\]

可能已经包含主要分类信息，空间坐标贡献有限。

## 21.3 去坐标后更好

可能说明：

- 坐标引入噪声；
- 坐标导致小样本过拟合；
- 坐标增强站点偏差；
- 坐标与邻接矩阵节点顺序不一致；
- 当前任务不需要强空间先验。

这将支持后续通用模型不使用坐标。

## 21.4 两组都接近随机

不能据此得出“空间坐标无效”。更准确的结论是：

\[
\boxed{
\text{当前数据管线尚未复现原论文STSE的有效分类能力。}
}
\]

此时无法可靠识别空间坐标的独立贡献。

---

# 22. 正式实验表述模板

> 为检验短时状态编码器是否依赖脑区空间先验，我们构建了 Exact-STSE-NoCoord。该变体同时删除节点三维坐标与由坐标计算的邻居空间中心，保留节点绝对加权度、相邻窗口度变化、社区嵌入、LayerNorm、线性投影、残差前馈编码器、节点均值池化和分类头不变。两种模型采用完全相同的数据划分、随机种子和训练设置进行比较。

---

# 23. 实验边界

该实验能够回答：

\[
\boxed{
\text{在当前数据和复现设置下，空间坐标是否提高STSE分类性能？}
}
\]

不能直接回答：

- 原作者最终模型是否依赖坐标；
- 原论文报告的全部性能提升是否来自空间信息；
- 坐标是否在所有脑疾病任务上有效；
- 去坐标模型是否必然具有更好的跨领域泛化；
- STSE-only 的原始作者实现是否与本复现的时间均值聚合完全一致。

---

# 24. 最终架构总结

## Exact-STSE

```text
当前图 + 前一窗口图 + 坐标 + 社区标签
→ degree
→ delta_degree
→ neighbor_coordinates
→ community_embedding
→ 特征拼接
→ LayerNorm
→ Linear
→ Residual FFN
→ 节点Mean Pooling
→ 窗口序列Mean Pooling
→ 两层ReLU分类头
→ 二分类结果
```

## Exact-STSE-NoCoord

```text
当前图 + 前一窗口图 + 社区标签
→ degree
→ delta_degree
→ community_embedding
→ 特征拼接
→ LayerNorm
→ Linear
→ Residual FFN
→ 节点Mean Pooling
→ 窗口序列Mean Pooling
→ 两层ReLU分类头
→ 二分类结果
```

两者之间唯一的设计变量是：

\[
\boxed{
\text{是否使用节点自身坐标和邻居空间坐标。}
}
\]
