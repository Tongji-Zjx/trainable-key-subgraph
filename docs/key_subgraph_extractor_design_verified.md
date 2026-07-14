# 关键子图提取器设计规范

> **文档状态：经原始 Word 文档、真实 `.pt` 数据检查、当前对话设计与工程可实现性复核后的实现级规范。**
>
> Codex 应以本文中的张量接口、训练/导出双路径、边界条件和测试要求为准。Word 文档用于追溯原始设计与视觉参考。若公式、真实 `.pt` 数据结构和可微性要求发生冲突，必须先报告冲突，不得自行猜测。


## 1. 模块目标

本模块用于图序列分类任务中的关键子图提取。第 \(b\) 个样本输入为一个按时间顺序排列、长度可变的图序列：

\[
\mathcal{G}_b=\left\{G_b^{(1)},G_b^{(2)},\ldots,G_b^{(M_b)}\right\}
\]

其中第 \(m\) 个时间片图表示为：

\[
G_b^{(m)}=\left(V_b^{(m)},A_b^{(m)},X_b^{(m)},C_b^{(m)}\right)
\]

其中：

- \(V_b^{(m)}=\{v_{b,1}^{(m)},\ldots,v_{b,N_b^{(m)}}^{(m)}\}\) 为节点集合；
- \(A_b^{(m)}\in\mathbb{R}^{N_b^{(m)}\times N_b^{(m)}}\) 为加权邻接矩阵；
- \(X_b^{(m)}\in\mathbb{R}^{N_b^{(m)}\times d}\) 为节点特征矩阵；
- \(C_b^{(m)}=\{c_{b,1}^{(m)},\ldots,c_{b,N_b^{(m)}}^{(m)}\}\) 为社区标签集合；
- \(M_b\) 为样本 \(b\) 的有效时间片数量；
- \(N_b^{(m)}\) 为样本 \(b\) 在时间片 \(m\) 的有效节点数量。

本文后续为简化公式而使用的 \(M\) 和 \(N\)，除非明确说明为 padded batch 维度，均应理解为当前样本相关的 \(M_b\) 和当前时间片相关的 \(N_b^{(m)}\)。不得假设同一 batch 内所有样本具有相同的 \(M\) 或 \(N\)。

本模块的目标不是无监督压缩原图，而是在分类监督下，从每个时间片图中提取若干对最终类别判断有用的关键子图。每个输出子图应尽量满足以下性质：

1. 保留与类别相关的判别信息；
2. 结构紧凑；
3. 节点与边尽量连通；
4. 同一时间片内多个关键子图之间不过度重复；
5. 能够稳定导出为离散节点集合与边集合。

## 2. 输入接口、变长批处理与数据有效性

原始数据不一定直接包含 \(X^{(m)}\)。本规范区分：

- **原始输入**：邻接矩阵、社区标签、稳定节点标识、样本标签；节点坐标即使存在也只作为源数据元信息保留，不进入模型；
- **派生特征**：连接强度、时间变化、社区结构特征；
- **最终节点特征**：完成构造后的 \(X^{(m)}\)。

在检查真实 `.pt` 文件前，不得假定字段名、对象类型或维度顺序。数据适配层应保留每个样本的原始时间片数、节点数、节点身份和图结构，禁止通过截断改变原始图。

### 2.1 首选 list-based batching

第一版首选 list-based batching。一个 batch 表示为长度为 \(B\) 的样本列表，每个样本包含长度为 \(M_b\) 的时间片列表：

```text
batch[b].graphs[m].adjacency       # [N_b^(m), N_b^(m)]
batch[b].graphs[m].community       # [N_b^(m)]
batch[b].graphs[m].node_ids        # 原始稳定节点标识
batch[b].label                     # 0 或 1
```

每个图独立编码为固定维度表示，随后在样本内进行 masked 时间池化，最后将样本表示堆叠为 \([B,F_h]\)。该方案不要求为计算方便而伪造节点或时间片。

### 2.2 可选 padded batching

若某一模块必须使用 padding，则 padded batch 可以表示为：

\[
A\in\mathbb{R}^{B\times M_{max}\times N_{max}\times N_{max}}
\]

\[
C\in\mathbb{Z}^{B\times M_{max}\times N_{max}}
\]

并必须同时提供：

- `time_mask`：\([B,M_{max}]\)；
- `node_mask`：\([B,M_{max},N_{max}]\)；
- `edge_mask`：\([B,M_{max},N_{max},N_{max}]\)，只标记满足 \(|A_{ij}|>\tau_{edge}\) 的真实非自环边；
- `subgraph_mask`：硬候选或 Top-\(K\) 输出的有效性 mask。

所有 pooling、loss、预算比例和统计计算必须忽略 mask 为假的部分。padding 社区值不得与任何真实社区编号混用。空间坐标不得进入节点或边特征，也不得通过邻居聚合等间接形式进入模型。

### 2.3 跨时间节点对齐

时间差分要求相邻时间片中的节点由稳定原始 ID 对齐。若 \(N_b^{(m)}\neq N_b^{(m-1)}\)，不得按局部矩阵行号直接相减。应构造相邻时间片节点并集和 `presence_mask`，按原始节点 ID 对齐；只在定义有效的对齐项上计算差分，缺失项由 mask 排除，不得将缺失节点静默当作真实零连接节点。为保持张量接口，未定义的差分位置可以存储安全零值，但必须同时输出 `delta_degree_mask` 或 `delta_edge_mask`，后续 pooling、loss、导出和统计不得把该安全值解释为真实的零变化。

### 2.4 当前数据有效性规则

- 社区标签异常样本 `data/adhd_5_0.5/NeuroIMAGE/1/NeuroIMAGE_3808273_1.pt` 直接排除，并在 exclusion manifest 中记录路径、样本 ID 和原因；
- 空间坐标不是模型输入，也不是样本有效性的必要条件；`.pt` 中坐标缺失、全零或格式异常均不得单独导致样本被排除；
- 标签从受控的数据索引读取；若标签来自目录，必须验证目录值只能为 `0` 或 `1`；
- 所有排除必须可复现且显式记录，不得静默跳过加载失败或异常样本。

### 2.5 带符号边与边存在性

邻接矩阵可以同时包含正边和负边。负边表示有效连接，而不是不存在的边。统一定义边存在 mask：

\[
edge\_mask_{ij}^{(m)}
=
\mathbf 1\left(i\neq j\right)
\mathbf 1\left(\left|A_{ij}^{(m)}\right|>\tau_{edge}\right)
\]

其中 \(\tau_{edge}\geq0\) 为边存在阈值。若输入邻接矩阵已经完成阈值化并以精确零表示无边，可以使用 \(\tau_{edge}=0\)。阈值必须在查看 held-out 结果前固定，并在配置、checkpoint 和导出元数据中记录数值及来源。

以下规则在训练、硬导出和统计模块中必须完全一致：

1. 禁止使用 \(A_{ij}>0\) 或 `soft_adj > 0` 判断边存在；
2. 拓扑判断只使用 \(|A_{ij}|>\tau_{edge}\)；
3. 连接强度和归一化分母使用绝对权重；
4. 消息传递和软选择邻接矩阵保留原始边符号；
5. 无向图只统计一次边，统一使用 \(i<j\)；
6. padding 和自环始终不属于有效边。

## 3. 节点特征构造

### 3.1 节点连接强度

对于第 \(m\) 个时间片中的节点 \(v_i\)，定义节点连接强度：

\[
d_i^{(m)}=\sum_{j\in V_b^{(m)}}\left|A_{ij}^{(m)}\right|
\]

该特征反映节点 \(v_i\) 在当前时间片与其他节点的总体连接强度。

单图输出语义为：

\[
d_b^{(m)}\in\mathbb{R}^{N_b^{(m)}}
\]

参考 PyTorch 语义：

```python
degree = adj.abs().sum(dim=-1)
```

### 3.2 相邻时间片节点连接变化

对于 (m>1) 且节点 (i) 在相邻时间片可按原始 ID 对齐时，定义：

\[
\Delta d_i^{(m)}=d_i^{(m)}-d_i^{(m-1)}
\]

对于第一个时间片：

\[
\Delta d_i^{(1)}=0
\]

固定节点集合时的参考 PyTorch 语义：

```python
delta_degree = torch.zeros_like(degree)
delta_degree[:, 1:] = degree[:, 1:] - degree[:, :-1]
```

### 3.3 空间坐标排除规则

空间坐标不属于本模型的节点特征。禁止将原始坐标、坐标差分、空间距离或邻居坐标聚合以直接或间接形式输入节点评分器、边评分器、图编码器或分类头。数据适配器不得要求 `coords` 字段存在，也不得因为坐标缺失、全零、非有限或形状异常而拒绝一个其他字段有效的样本。样本索引可将坐标状态作为审计元数据记录，但 `coords_valid` 不参与 `included` 判定。放宽坐标条件后必须重建样本索引；旧索引和旧数据协议不得与新实验混用。

### 3.4 社区结构特征

社区编号可能由每个时间片独立运行的社区发现算法生成，不保证跨样本或跨时间对齐。因此严禁将原始 `community_id` 直接输入 `nn.Embedding`。社区标签只用于同社区判断、社区覆盖、每社区 Top-\(q\) 种子选择和社区结构统计。

令节点 \(i\) 所属社区为：

\[
\mathcal C_i^{(m)}=\{v_j\mid c_j^{(m)}=c_i^{(m)}\},\qquad n_{c,i}^{(m)}=|\mathcal C_i^{(m)}|
\]

定义通过边存在阈值后的正连接强度和负连接幅值：

\[
[a]_{+,\tau}=a\,\mathbf 1(a>\tau_{edge}),\qquad
[a]_{-,\tau}=|a|\,\mathbf 1(a<-\tau_{edge})
\]

其中 \([a]_{-,\tau}\) 表示有效负边的绝对幅值。使用以下七个具有跨样本一致语义的社区结构特征。

社区相对规模：

\[
s_{c,i}^{(m)}=\frac{n_{c,i}^{(m)}}{N_b^{(m)}+\varepsilon}
\]

节点的社区内正连接平均强度：

\[
w_{intra,+,i}^{(m)}=
\frac{\sum_{j\neq i,\,c_j^{(m)}=c_i^{(m)}}[A_{ij}^{(m)}]_{+,\tau}}
{\max(n_{c,i}^{(m)}-1,1)}
\]

节点的社区内负连接平均幅值：

\[
w_{intra,-,i}^{(m)}=
\frac{\sum_{j\neq i,\,c_j^{(m)}=c_i^{(m)}}[A_{ij}^{(m)}]_{-,\tau}}
{\max(n_{c,i}^{(m)}-1,1)}
\]

节点的跨社区正连接平均强度：

\[
w_{inter,+,i}^{(m)}=
\frac{\sum_{j:\,c_j^{(m)}\neq c_i^{(m)}}[A_{ij}^{(m)}]_{+,\tau}}
{\max(N_b^{(m)}-n_{c,i}^{(m)},1)}
\]

节点的跨社区负连接平均幅值：

\[
w_{inter,-,i}^{(m)}=
\frac{\sum_{j:\,c_j^{(m)}\neq c_i^{(m)}}[A_{ij}^{(m)}]_{-,\tau}}
{\max(N_b^{(m)}-n_{c,i}^{(m)},1)}
\]

所属社区的正边无向拓扑密度：

\[
density_{c,+,i}^{(m)}=
\frac{2|\{(u,v):u<v,\,u,v\in\mathcal C_i^{(m)},\,A_{uv}^{(m)}>\tau_{edge}\}|}
{n_{c,i}^{(m)}(n_{c,i}^{(m)}-1)+\varepsilon}
\]

所属社区的负边无向拓扑密度：

\[
density_{c,-,i}^{(m)}=
\frac{2|\{(u,v):u<v,\,u,v\in\mathcal C_i^{(m)},\,A_{uv}^{(m)}<-\tau_{edge}\}|}
{n_{c,i}^{(m)}(n_{c,i}^{(m)}-1)+\varepsilon}
\]

单节点社区的正、负社区密度安全定义为 0。正负连接必须分别构造，不得先对带符号边权求和或平均后作为社区强度，因为该操作会发生正负抵消。上述强度按可连接节点数归一化，以降低不同 \(N_b^{(m)}\) 和社区规模造成的直接偏差；社区相对规模单独保留。

### 3.5 最终节点特征

\[
x_i^{(m)}=
\left[
d_i^{(m)};
\Delta d_i^{(m)};
s_{c,i}^{(m)};
w_{intra,+,i}^{(m)};
w_{intra,-,i}^{(m)};
w_{inter,+,i}^{(m)};
w_{inter,-,i}^{(m)};
density_{c,+,i}^{(m)};
density_{c,-,i}^{(m)}
\right]
\]

节点特征维度为：

\[
F_x=1+1+7=9
\]

单图节点特征张量为：

\[
X_b^{(m)}\in\mathbb{R}^{N_b^{(m)}\times F_x}
\]

list-based batch 保留这些变长张量；padded batch 才表示为 \([B,M_{max},N_{max},F_x]\)，并必须配合 `time_mask` 与 `node_mask`。

## 4. 节点重要性评分

\[
p_i^{(m)}=\sigma\left(MLP_v(x_i^{(m)})\right)
\]

其中：

\[
p_i^{(m)}\in[0,1]
\]

所有节点分数组成：

\[
p_v^{(m)}=\left[p_1^{(m)},p_2^{(m)},\ldots,p_N^{(m)}\right]
\]

单图输出形状为：

\[
P_{v,b}^{(m)}\in\mathbb{R}^{N_b^{(m)}}
\]

## 5. 边特征与边重要性评分

### 5.1 相邻时间片边权变化

对于 \(m>1\)：

\[
\Delta A_{ij}^{(m)}=A_{ij}^{(m)}-A_{ij}^{(m-1)}
\]

对于第一个时间片：

\[
\Delta A_{ij}^{(1)}=0
\]

### 5.2 边特征

\[
e_{ij}^{(m)}=
\left[
x_i^{(m)};
x_j^{(m)};
A_{ij}^{(m)};
\left|A_{ij}^{(m)}\right|;
\Delta A_{ij}^{(m)};
\left|\Delta A_{ij}^{(m)}\right|;
\mathbf{1}\left(c_i^{(m)}=c_j^{(m)}\right)
\right]
\]

必须同时保留 \(A_{ij}^{(m)}\) 与 \(|A_{ij}^{(m)}|\)、\(\Delta A_{ij}^{(m)}\) 与 \(|\Delta A_{ij}^{(m)}|\)。前者保留符号方向，后者提供不发生正负抵消的幅值信息。不得只保留其中一种。

其中：

\[
\mathbf{1}\left(c_i^{(m)}=c_j^{(m)}\right)=
\begin{cases}
1,&c_i^{(m)}=c_j^{(m)}\\
0,&c_i^{(m)}\neq c_j^{(m)}
\end{cases}
\]

### 5.3 边重要性评分

\[
p_{ij}^{(m)}=\sigma\left(MLP_e(e_{ij}^{(m)})\right)
\]

对于无向图，应保证：

\[
p_{ij}^{(m)}=p_{ji}^{(m)}
\]

可通过对称化实现：

\[
P_e^{(m)}\leftarrow
\frac{P_e^{(m)}+\left(P_e^{(m)}\right)^\top}{2}
\]

单图输出形状为：

\[
P_{e,b}^{(m)}\in\mathbb{R}^{N_b^{(m)}\times N_b^{(m)}}
\]

## 6. 软选择图构建

\[
\bar{A}_{ij}^{(m)}=
A_{ij}^{(m)}\cdot p_i^{(m)}\cdot p_j^{(m)}\cdot p_{ij}^{(m)}
\]

矩阵形式为：

\[
\bar{A}^{(m)}=
A^{(m)}\odot
\left(p_v^{(m)}\left(p_v^{(m)}\right)^\top\right)
\odot P_e^{(m)}
\]

软选择图表示为：

\[
\bar{G}^{(m)}=\left(V,\bar{A}^{(m)},X^{(m)}\right)
\]

参考 PyTorch 语义：

```python
node_pair_score = node_score.unsqueeze(-1) * node_score.unsqueeze(-2)
soft_adj = adj * node_pair_score * edge_score
```

由于 \(p_i,p_j,p_{ij}\geq0\)，每条有效边的 `soft_adj` 必须保留 \(A_{ij}\) 的符号。若图编码器需要归一化，归一化分母可以使用绝对连接强度，但消息权重不得无条件替换为 \(|\bar A_{ij}|\)，否则正连接与负连接会失去区别。

## 7. 社区感知种子节点选择

设第 \(m\) 个时间片图包含 \(R\) 个社区：

\[
\mathcal{C}^{(m)}=\left\{C_1^{(m)},C_2^{(m)},\ldots,C_R^{(m)}\right\}
\]

其中：

\[
C_r^{(m)}=\left\{v_i\mid c_i^{(m)}=r\right\}
\]

在每个社区内部选择节点重要性最高的 \(q\) 个节点作为种子：

\[
Seed_r^{(m)}=
Topq_{v_i\in C_r^{(m)}}\left(p_i^{(m)}\right)
\]

所有社区种子节点的并集为：

\[
Seed^{(m)}=\bigcup_{r=1}^{R}Seed_r^{(m)}
\]

## 8. 候选子图生成

对于每个种子节点 \(s\in Seed^{(m)}\)，在带符号图的无向边存在拓扑上进行固定 \(L\)-hop 邻域扩展。定义：

\[
T_{ij}^{(m)}=\mathbf 1\left(i\neq j\right)
\mathbf 1\left(|A_{ij}^{(m)}|>\tau_{edge}\right)
\]

\(L\)-hop 距离使用 \(T^{(m)}\) 上的无权最短路，不使用带符号权重作为距离；负权最短路不具有这里所需的 hop 语义。

\[
\widetilde{V}_s^{(m)}=
\left\{v_i\mid dist_{T^{(m)}}(v_i,s)\leq L\right\}
\]

\[
\widetilde{E}_s^{(m)}=
\left\{(v_i,v_j)\mid v_i,v_j\in\widetilde{V}_s^{(m)},|A_{ij}^{(m)}|>\tau_{edge}\right\}
\]

\[
\widetilde{S}_s^{(m)}=
\left(\widetilde{V}_s^{(m)},\widetilde{E}_s^{(m)}\right)
\]

拓扑存在性由 \(|A_{ij}|>\tau_{edge}\) 决定，不得使用 \(A_{ij}>0\) 或 `soft_adj > 0`，否则会错误删除负权边。软选择权重用于可微编码与重要性排序；导出时保留原始边权符号。无向边统一规范为 \(i<j\)，自环不进入候选边集。

## 9. 候选子图压缩

### 9.1 节点筛选

\[
V_s^{(m)}=
Topn_{v_i\in\widetilde{V}_s^{(m)}}\left(p_i^{(m)}\right)
\]

并满足：

\[
|V_s^{(m)}|\leq n_{\max}
\]

### 9.2 边筛选

\[
E_s^{(m)}=
Topb_{\substack{(v_i,v_j),\,v_i,v_j\in V_s^{(m)}\\|A_{ij}^{(m)}|>\tau_{edge}}}
\left(p_{ij}^{(m)}\right)
\]

并满足：

\[
|E_s^{(m)}|\leq b_{\max}
\]

压缩后的候选子图定义为：

\[
S_s^{(m)}=\left(V_s^{(m)},E_s^{(m)}\right)
\]

## 9.3 候选有效性约束

候选子图至少应满足：

\[
|V_s^{(m)}|\geq n_{\min}
\]

\[
|E_s^{(m)}|\geq e_{\min}
\]

推荐第一版设置 \(n_{\min}\geq 2\)、\(e_{\min}\geq 1\)。不满足条件的候选应被丢弃。

若某个时间片不足 \(K\) 个有效候选，必须保存 `num_valid_subgraphs` 与 `subgraph_mask`。填充项不得进入分类池化或结构统计。不得复制候选补足 \(K\)，也不得通过截断原图制造固定候选规模。

## 10. 候选子图编码

\[
z_s^{(m)}=Enc(S_s^{(m)})=Pool\left(GNN(S_s^{(m)})\right)
\]

其中 \(z_s^{(m)}\in\mathbb{R}^{F_z}\) 为候选子图表示。

## 11. 候选子图局部分类头

\[
o_s^{(m)}=MLP_{score}\left(z_s^{(m)}\right)
\]

\[
Conf(S_s^{(m)})=\max_c softmax\left(o_s^{(m)}\right)_c
\]

## 12. 候选子图综合评分

\[
Score(S_s^{(m)})=
\lambda_v Score_v+
\lambda_e Score_e+
\lambda_c Score_c+
\lambda_d Score_d+
\lambda_y Score_y
\]

其中：

\[
Score_v=
\frac{1}{|V_s^{(m)}|}
\sum_{v_i\in V_s^{(m)}}p_i^{(m)}
\]

\[
Score_e=
\frac{1}{|E_s^{(m)}|}
\sum_{(v_i,v_j)\in E_s^{(m)}}p_{ij}^{(m)}
\]

\[
Score_c=
\frac{|LCC(S_s^{(m)})|}{|V_s^{(m)}|}
\]

\[
Score_d=
\frac{1}{|V_s^{(m)}|}
\sum_{v_i\in V_s^{(m)}}\left|\Delta d_i^{(m)}\right|
\]

\[
Score_y=Conf(S_s^{(m)})
\]

若边集合为空，应返回安全值并记录异常情况。

## 13. 候选子图去重

\[
Overlap(S_a^{(m)},S_b^{(m)})=
\frac{|V_a^{(m)}\cap V_b^{(m)}|}{|V_a^{(m)}\cup V_b^{(m)}|+\varepsilon}
+
\frac{|E_a^{(m)}\cap E_b^{(m)}|}{|E_a^{(m)}\cup E_b^{(m)}|+\varepsilon}
\]

若：

\[
Overlap(S_a^{(m)},S_b^{(m)})>\tau_o
\]

则仅保留综合评分更高的候选子图。

## 14. 最终 Top-\(K\) 子图选择

去重后的候选集合记为 \(\widetilde{\mathcal{S}}^{(m)}\)。最终选择：

\[
\mathcal{S}^{(m)}=
\arg\max_{\mathcal{S}\subseteq\widetilde{\mathcal{S}}^{(m)},\,|\mathcal{S}|=K}
\sum_{S_k^{(m)}\in\mathcal{S}}Score(S_k^{(m)})
\]

并要求：

\[
Overlap(S_a^{(m)},S_b^{(m)})\leq\tau_o,\qquad a\neq b
\]

最终输出：

\[
\mathcal{S}^{(m)}=\left\{S_1^{(m)},S_2^{(m)},\ldots,S_K^{(m)}\right\}
\]

## 15. 训练路径与硬提取路径

本模块必须区分可微训练路径与冻结后的硬子图导出路径。

### 15.1 可微训练路径

训练阶段使用软选择图：

\[
\bar A_{ij}^{(m)}
=
A_{ij}^{(m)}
p_i^{(m)}
p_j^{(m)}
p_{ij}^{(m)}
\]

对每个时间片进行软图编码：

\[
g^{(m)}
=
Pool
\left(
GNN
\left(
V,\bar A^{(m)},X^{(m)}
\right)
\right)
\]

对第 (b) 个样本的有效时间片池化：

\[
h_b
=
\frac{
\sum_{m=1}^{M_{max}}
time\_mask_{b,m}\,g_b^{(m)}
}{
\sum_{m=1}^{M_{max}}time\_mask_{b,m}+\varepsilon
}
\]

最终预测：

\[
\hat y
=
MLP_{cls}(h_b)
\]

该路径不执行硬 Top-\(q\)、硬 \(L\)-hop、硬 Top-\(n_{\max}\)、硬 Top-\(b_{\max}\) 或最终硬 Top-\(K\)，因此分类梯度能够传播到节点评分网络和边评分网络。

### 15.2 硬子图导出路径

验证、测试和结构差异实验阶段冻结参数，并执行：

\[
G^{(m)}
\rightarrow
Seed^{(m)}
\rightarrow
\widetilde S_s^{(m)}
\rightarrow
S_s^{(m)}
\rightarrow
\mathcal S^{(m)}
\]

该路径输出离散节点集合与边集合，不参与梯度更新。

### 15.3 与原始 Word 方案的关系

原始方案使用最终硬子图表示：

\[
h_{hard}
=
\frac{
\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}
subgraph\_mask_{bmk}\,Enc(S_{b,k}^{(m)})
}{
\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}subgraph\_mask_{bmk}+\varepsilon
}
\]

该式描述了理想的子图级分类目标，但硬选择不可微。第一版实现默认采用：

```yaml
training_mode: soft_graph
```

只有在显式实现并测试 straight-through 或 Gumbel 近似后，才允许：

```yaml
training_mode: straight_through
```

不得将硬 Top-\(K\) 直接放入训练路径后声称其可端到端反向传播。

## 16. 训练目标

### 16.1 第一版基线的必选损失

第一版稳定基线使用：

\[
\mathcal L_{base}
=
\mathcal L_{cls}
+
\alpha_{budget}\mathcal L_{budget}
\]

分类损失：

\[
\mathcal L_{cls}
=
CE(\hat y,y)
\]

节点保留比例：

\[
r_{v,b}^{(m)}
=
\frac{
\sum_i node\_mask_{bmi}p_{b,i}^{(m)}
}{
\sum_i node\_mask_{bmi}+\varepsilon
}
\]

边保留比例：

\[
r_{e,b}^{(m)}
=
\frac{
\sum_{i,j}edge\_mask_{bmij}p_{b,ij}^{(m)}
}{
\sum_{i,j}edge\_mask_{bmij}+\varepsilon
}
\]

对于无向图，`edge_mask` 应采用统一的 \(i<j\) 边表示以避免双计数；若实现中保留对称矩阵计算，分子和分母必须采用完全一致的对称 mask。`edge_mask` 只覆盖满足 \(|A_{ij}|>\tau_{edge}\) 的原始正边和负边，不覆盖 padding、阈值内边或自环。

预算损失：

\[
\mathcal L_{budget}
=
\frac{
\sum_{b=1}^{B}\sum_{m=1}^{M_{max}}time\_mask_{b,m}
\left[
\left|r_{v,b}^{(m)}-\rho_v\right|
+
\left|r_{e,b}^{(m)}-\rho_e\right|
\right]
}{
\sum_{b=1}^{B}\sum_{m=1}^{M_{max}}time\_mask_{b,m}+\varepsilon
}
\]

该约束避免所有分数全部趋近 1 或全部塌缩为 0。

### 16.2 原始设计中的扩展目标

原始 Word 方案定义：

\[
\mathcal{L}
=
\mathcal{L}_{cls}
+
\alpha_1\mathcal{L}_{sub}
+
\alpha_2\mathcal{L}_{sparse}
+
\alpha_3\mathcal{L}_{conn}
+
\alpha_4\mathcal{L}_{div}
\]

其中：

\[
\mathcal{L}_{sub}
=
\frac{
\sum_n\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}
subgraph\_mask_{nmk}\,CE(o_{n,k}^{(m)},y_n)
}{
\sum_n\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}
subgraph\_mask_{nmk}+\varepsilon
}
\]

\[
\mathcal{L}_{conn}
=
\frac{
\sum_n\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}
subgraph\_mask_{nmk}
\left(
1-
\frac{|LCC(S_{n,k}^{(m)})|}{|V_{n,k}^{(m)}|+\varepsilon}
\right)
}{
\sum_n\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}
subgraph\_mask_{nmk}+\varepsilon
}
\]

\[
\mathcal{L}_{div}
=
\frac{
\sum_n\sum_{m=1}^{M_{max}}\sum_{a\neq k}
subgraph\_mask_{nma}subgraph\_mask_{nmk}
Overlap(S_{n,a}^{(m)},S_{n,k}^{(m)})
}{
\sum_n\sum_{m=1}^{M_{max}}\sum_{a\neq k}
subgraph\_mask_{nma}subgraph\_mask_{nmk}+\varepsilon
}
\]

这些公式保留为研究设计目标。但在默认 `soft_graph` 基线中：

- \(\mathcal L_{sub}\) 默认关闭；
- \(\mathcal L_{conn}\) 默认作为硬导出质量指标；
- \(\mathcal L_{div}\) 默认作为硬导出质量指标；
- 只有实现多个可微子图 mask 及相应代理损失后，才能将它们加入训练。

### 16.3 不允许的实现

1. 对硬选择结果 `detach()` 后仍声称分类损失训练了提取器；
2. 只最小化 \(\sum_i p_i+\sum_{ij}p_{ij}\)，导致全部分数趋近 0；
3. 使用测试集标签选择 \(\rho_v\)、\(\rho_e\)、\(K\)、阈值或损失权重；
4. 将填充子图计入平均池化。

## 17. 候选评分与训练监督的职责划分

综合评分：

\[
Score(S_s^{(m)})
=
\lambda_v Score_v
+
\lambda_e Score_e
+
\lambda_c Score_c
+
\lambda_d Score_d
+
\lambda_y Score_y
\]

主要用于冻结后的硬候选排序和 Top-\(K\) 导出。

\[
Score_y
=
\max_c softmax(o_s^{(m)})_c
\]

最大置信度不等于预测正确率。局部分类头只能在训练集上训练，并在验证集上选择模型；测试集标签不得用于候选排序。

第一版允许：

- 采用 `soft_graph` 基线时令 \(\lambda_y=0\)；
- 只有在训练数据候选上训练并验证局部分类头后，才启用 \(\lambda_y>0\)。

配置文件必须显式记录：

```yaml
use_local_confidence_score: false
```

## 18. Codex 实现检查表

编码前必须确认：

1. `.pt` 文件实际对象类型、字段名、shape 和 dtype；
2. 邻接矩阵是否有向、是否对称、是否包含负权，以及 \(\tau_{edge}\) 的数值和来源；
3. 逐样本、逐时间片记录 \(M_b\) 与 \(N_b^{(m)}\)，不得假设固定；
4. 社区标签是否有效，异常样本是否进入 exclusion manifest；确认空间坐标未进入任何模型特征；
5. 相邻时间片的原始节点 ID 是否可对齐；
6. 标签编码方式及站点、受试者分组信息；
7. 默认采用 `training_mode: soft_graph`；
8. 原始社区编号未输入 `nn.Embedding`；
9. 硬提取只用于冻结后的导出与统计；
10. 所有 pooling、loss 和统计支持 mask；
11. 所有除法使用 \(\varepsilon\)；
12. 输出保留原始节点编号与名称；
13. 未通过截断改变原始时间片或图结构。

最小单元测试：

```python
assert degree.shape == (N_bm,)
assert delta_degree.shape == (N_bm,)
assert community_features.shape == (N_bm, 7)
assert node_features.shape == (N_bm, 9)
assert edge_features.shape == (N_bm, N_bm, 23)
assert torch.isfinite(community_features).all()
assert node_score.min() >= 0
assert node_score.max() <= 1
assert edge_score.min() >= 0
assert edge_score.max() <= 1
assert torch.allclose(
    edge_score,
    edge_score.transpose(-1, -2),
    atol=1e-6,
)
assert soft_adj.shape == adj.shape
assert torch.isfinite(soft_adj).all()
assert torch.equal(edge_mask, adj.abs() > edge_presence_threshold)
assert torch.all(torch.sign(soft_adj[edge_mask]) == torch.sign(adj[edge_mask]))
assert edge_features.shape[-1] == 2 * node_feature_dim + 5
assert not model_uses_raw_community_embedding
assert padding_invariance_error < tolerance
assert truncated_node_count == 0
assert truncated_time_count == 0
```

还必须用至少两个不同 \(M_b\)、不同 \(N_b^{(m)}\) 的样本组成同一 batch，验证 list-based 与 padded+mask 两种等价实现得到一致结果；向 padding 区域写入任意有限值不得改变 pooling、loss 或导出结果。

## 19. 导出格式

每个样本、时间片和关键子图至少保存：

- `sample_id`
- `site`
- `label`
- `split`
- `fold`
- `time_index`
- `subgraph_index`
- `node_ids`
- `node_names`
- `edge_index`
- `original_edge_weights`
- `edge_presence_threshold`
- `node_scores`
- `edge_scores`
- `candidate_score`
- `community_labels`
- `delta_degree`
- `delta_edge_weight`
- `delta_degree_mask`
- `delta_edge_mask`
- `time_mask`
- `node_mask`
- `subgraph_mask`
- `num_valid_subgraphs`
- `original_graph_ref`
- `candidate_pool_ref`

`original_edge_weights` 必须保存带符号原始边权，不得用绝对值覆盖；需要绝对幅值时由下游显式派生。`edge_presence_threshold` 必须与训练时的 \(\tau_{edge}\) 完全一致。

不得丢失原始节点编号或名称。Low-score 对照需要访问同一样本、同一时间片的完整有效候选池，因此应额外保存候选池，或在硬导出阶段同时保存按相同有效性和去重规则选出的 Low-score 对照。Random 与 Top-degree 对照必须能够通过 `sample_id`、`site` 和 `time_index` 回查原图。

## 20. 当前实现边界

当前只实现：

1. 图序列数据适配；
2. 节点特征构造；
3. 节点与边重要性评分；
4. 软选择图；
5. 候选子图生成；
6. 候选子图压缩与评分；
7. 最终 Top-\(K\) 子图导出；
8. 简单分类头监督训练。

当前不实现：

- 跨时间子图匹配；
- 子图演化网络；
- 原型记忆；
- 长期图分支；
- 多分支融合。


## 21. 与原始设计的一致性说明

本文完整保留了原始设计中的图序列输入、连接与动态特征、节点与边重要性评分、软选择图、社区感知种子节点、\(L\)-hop 候选扩展、节点与边压缩、候选编码与评分、重叠去重、最终 Top-\(K\) 离散子图以及分类监督目标。由于真实社区编号不具备跨样本和跨时间语义，原始社区 ID embedding 已替换为社区相对规模、社区内/外正负平均连接强度和社区内正负边密度；该修订保留社区结构信息并消除编号置换依赖。

变长 list-based batching、可选 padding mask、节点 ID 对齐、训练/导出双路径、负权边拓扑 mask、有效候选约束和异常样本清单属于工程澄清，用于消除真实数据变长结构、离散操作与反向传播之间的歧义，不改变模块目标。

## 22. 当前全样本探索性训练配置

当前项目将提取器视为结构探索的中间工具，采用独立的
`all_samples_exploratory` 协议：938 个有效样本全部进入显式的 `all`
分区，同时用于分类监督训练和冻结后的硬关键子图提取。该配置不得伪装成
train/validation/test 划分，也不得把同一队列上的分类指标解释为泛化性能。

每个 epoch 后，模型以 `eval()` 模式在完整队列上重新计算分类损失与预算损失，
以最低完整队列 loss 保存 `best_checkpoint.pt`，并始终保存
`last_checkpoint.pt`。history 和 checkpoint 使用 `cohort`、
`selection_partition: cohort` 记录该过程，不使用 `validation` 命名。

该配置只改变数据使用和 checkpoint 选择策略，不改变软选择公式、预算约束、
带符号边处理、硬候选生成或 Top-\(K\) 导出定义。旧的严格划分协议仍保留为
历史预测实验路径，但其索引、checkpoint 与全样本协议不得混用。
