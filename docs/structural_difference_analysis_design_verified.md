# 类别关键子图结构差异统计模块设计规范

> **文档状态：经原始 Word 文档、真实 `.pt` 数据检查、当前对话设计与统计实现要求复核后的实现级规范。**
>
> 本模块只分析冻结模型在 held-out 样本上导出的离散子图，不参与提取器训练。


## 1. 实验目标

本模块用于验证训练完成的关键子图提取器是否从两类图序列中提取出了具有类别区分能力的有效结构。

实验不以分类准确率作为唯一证据，而是直接比较两类样本中关键子图的结构统计分布。若两类关键子图在规模、密度、连接强度、动态变化和社区结构等维度上存在稳定且显著的差异，并且这种差异强于随机子图、Top-degree 子图和 Low-score 子图，则认为关键子图提取器捕捉到了与类别相关的有效结构。

## 2. 数据定义

第 \(n\) 个变长图序列样本记为：

\[
\mathcal{G}_n=\left\{G_n^{(1)},G_n^{(2)},\ldots,G_n^{(M_n)}\right\}
\]

对应标签为：

\[
y_n\in\{0,1\}
\]

其中时间片 \(m\) 的节点数记为 \(N_n^{(m)}\)。不同样本的 \(M_n\) 和 \(N_n^{(m)}\) 均可不同，不得通过截断统一时间片数或节点数。

训练完成后的关键子图提取器对每个有效时间片最多输出 \(K\) 个关键子图：

\[
G_n^{(m)}\rightarrow
\mathcal{S}_n^{(m)}=
\left\{S_{n,1}^{(m)},S_{n,2}^{(m)},\ldots,S_{n,K}^{(m)}\right\}
\]

其中：

\[
S_{n,k}^{(m)}=\left(V_{n,k}^{(m)},E_{n,k}^{(m)}\right)
\]

对第 \(n\) 个样本，其全部关键子图集合为：

\[
\mathcal{S}_n^{1:M_n}=
\left\{S_{n,k}^{(m)}\mid m=1,\ldots,M_n;\,k=1,\ldots,K;\,w_{nmk}=1\right\}
\]

其中 \(w_{nmk}\) 由 `time_mask` 与 `subgraph_mask` 共同确定。无效时间片、无效候选和 padding 项不属于统计样本。本文后续简写的 \(M\) 和 \(N\) 均应理解为样本相关变量。

### 2.1 数据有效性与排除清单

- 社区标签异常样本 `data/adhd_5_0.5/NeuroIMAGE/1/NeuroIMAGE_3808273_1.pt` 直接排除；
- 空间坐标不进入提取器或结构统计模块；坐标缺失、全零或格式异常不得单独作为排除样本的理由；
- 所有排除规则和 exclusion manifest 必须在查看 held-out 统计结果前固定，并对所有类别、折和对照组一致应用；
- 不得根据类别差异显著性选择性排除样本、时间片或子图。

## 3. 核心实验假设

设 \(\phi(S)\) 为子图 \(S\) 的结构统计向量。若提取器有效，则：

\[
P\left(\phi(S)\mid Y=0\right)
\neq
P\left(\phi(S)\mid Y=1\right)
\]

## 4. 结构统计向量

对于一个来自第 \(m\) 个时间片的子图：

\[
S=(V_S,E_S)
\]

定义：

\[
\phi(S)=
\left[
N_V(S),
N_E(S),
Density(S),
W_{abs,avg}(S),
W_{abs,sum}(S),
W_{+,avg}(S),
W_{+,sum}(S),
W_{-,avg}(S),
W_{-,sum}(S),
D_{node}(S),
D_{edge}(S),
R_{intra}^{+}(S),
R_{inter}^{+}(S),
R_{intra}^{-}(S),
R_{inter}^{-}(S)
\right]
\]

## 5. 子图规模指标

\[
N_V(S)=|V_S|
\]

\[
N_E(S)=|E_S|
\]

这里使用 \(N_E(S)\) 表示边数量，避免与边集合 \(E_S\) 混淆。其含义与原始 Word 文档中的边规模一致。

## 5.1 无向边计数约定

若邻接矩阵以对称矩阵保存无向图，一条无向边只能统计一次。推荐只统计 \(i<j\) 的边。不得同时统计 \((i,j)\) 和 \((j,i)\)，否则边数量、总边权和密度会翻倍。

邻接矩阵允许同时包含正边和负边。负边是有效连接，不是无边。统一定义：

\[
E_S=\{(i,j):i<j,\,i,j\in V_S,\,|A_{ij}^{(m)}|>\tau_{edge}\}
\]

\[
E_S^+=\{(i,j)\in E_S:A_{ij}^{(m)}>\tau_{edge}\}
\]

\[
E_S^-=\{(i,j)\in E_S:A_{ij}^{(m)}<-\tau_{edge}\}
\]

禁止使用 \(A_{ij}>0\) 作为全部边的存在条件。\(\tau_{edge}\) 必须与提取器训练和硬导出使用完全相同的值，并从 checkpoint 或导出元数据读取，不得在 held-out 统计阶段重新选择。

自环默认不计入边集合，除非真实任务明确要求保留。

## 6. 子图密度

对于无向子图：

\[
Density(S)=
\frac{2|E_S|}{|V_S|(|V_S|-1)+\varepsilon}
\]

对于有向子图：

\[
Density_{directed}(S)=
\frac{|E_S|}{|V_S|(|V_S|-1)+\varepsilon}
\]

实现前必须明确图类型。

## 7. 子图连接强度指标

### 7.1 总体平均绝对边权

\[
W_{abs,avg}(S)=
\frac{1}{|E_S|}
\sum_{(i,j)\in E_S}\left|A_{ij}^{(m)}\right|
\]

### 7.2 总体绝对连接强度

\[
W_{abs,sum}(S)=
\sum_{(i,j)\in E_S}\left|A_{ij}^{(m)}\right|
\]

### 7.3 正连接强度

\[
W_{+,avg}(S)=
\frac{1}{|E_S^+|}
\sum_{(i,j)\in E_S^+}A_{ij}^{(m)}
\]

\[
W_{+,sum}(S)=
\sum_{(i,j)\in E_S^+}A_{ij}^{(m)}
\]

### 7.4 负连接幅值

\[
W_{-,avg}(S)=
\frac{1}{|E_S^-|}
\sum_{(i,j)\in E_S^-}\left|A_{ij}^{(m)}\right|
\]

\[
W_{-,sum}(S)=
\sum_{(i,j)\in E_S^-}\left|A_{ij}^{(m)}\right|
\]

所有强度指标均为非负量。禁止计算或报告：

\[
\frac{1}{|E_S|}\sum_{(i,j)\in E_S}A_{ij}^{(m)}
\]

作为连接强度，因为正负边会相互抵消。总体指标必须使用绝对边权，正边和负边必须另外分开报告。

硬提取阶段原则上通过 \(e_{\min}\geq 1\) 排除空边子图。若 \(|E_S|=0\)，全部边相关指标记为 `NaN` 并记录原因。若 \(|E_S^+|=0\) 或 \(|E_S^-|=0\)，相应的平均值记为 `NaN`，相应的总强度记为 0；不得把“没有该符号的边”和“该符号边的平均强度为 0”混为一谈。

## 8. 子图动态变化指标

### 8.1 节点连接变化

\[
\Delta d_i^{(m)}=d_i^{(m)}-d_i^{(m-1)}
\]

\[
\Delta d_i^{(1)}=0
\]

\[
D_{node}(S)=
\frac{1}{|V_S|}
\sum_{i\in V_S}\left|\Delta d_i^{(m)}\right|
\]

### 8.2 边权变化

\[
\Delta A_{ij}^{(m)}=A_{ij}^{(m)}-A_{ij}^{(m-1)}
\]

\[
\Delta A_{ij}^{(1)}=0
\]

\[
D_{edge}(S)=
\frac{1}{|E_S|}
\sum_{(i,j)\in E_S}
\left|A_{ij}^{(m)}-A_{ij}^{(m-1)}\right|
\]

上述时间差分只对能够通过原始节点 ID 在相邻时间片对齐的节点和边定义。若节点集合发生变化，统计模块必须读取提取器导出的对齐结果、`delta_degree_mask` 和 `delta_edge_mask`，不得按局部矩阵行号直接相减，也不得将缺失节点或边的安全占位零值作为真实零变化参与统计。

## 9. 社区结构指标

社区编号只在当前样本、当前时间片内用于相等性判断，不假定社区 ID 跨样本或跨时间具有一致语义。统计模块不得比较原始社区编号大小，也不得对社区编号做 embedding。

### 9.1 正连接的社区内部与跨社区比例

\[
R_{intra}^{+}(S)=
\frac{\left|\left\{(i,j)\in E_S^+\mid c_i^{(m)}=c_j^{(m)}\right\}\right|}
{|E_S^+|}
\]

\[
R_{inter}^{+}(S)=
\frac{\left|\left\{(i,j)\in E_S^+\mid c_i^{(m)}\neq c_j^{(m)}\right\}\right|}
{|E_S^+|}
\]

当 \(|E_S^+|>0\) 时，应满足：

\[
R_{intra}^{+}(S)+R_{inter}^{+}(S)\approx 1
\]

### 9.2 负连接的社区内部与跨社区比例

\[
R_{intra}^{-}(S)=
\frac{\left|\left\{(i,j)\in E_S^-\mid c_i^{(m)}=c_j^{(m)}\right\}\right|}
{|E_S^-|}
\]

\[
R_{inter}^{-}(S)=
\frac{\left|\left\{(i,j)\in E_S^-\mid c_i^{(m)}\neq c_j^{(m)}\right\}\right|}
{|E_S^-|}
\]

当 \(|E_S^-|>0\) 时，应满足：

\[
R_{intra}^{-}(S)+R_{inter}^{-}(S)\approx 1
\]

若对应符号的边集合为空，该符号的社区比例指标记为 `NaN` 并进入逐指标有效 mask，不得替换为 0。正负边必须分别统计；只报告混合后的社区内/跨社区比例不足以描述带符号网络结构。

## 10. 样本级聚合

同一样本内部数量可变的有效子图不能被视为相互独立的统计样本。不得假设每个样本恰好具有 \(M\times K\) 个有效子图。

对第 \(n\) 个样本：

\[
\bar{\phi}_n=
\frac{
\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}
w_{nmk}\phi\left(S_{n,k}^{(m)}\right)
}{
\sum_{m=1}^{M_{max}}\sum_{k=1}^{K}w_{nmk}+\varepsilon
}
\]

其中：

\[
w_{nmk}=time\_mask_{nm}\cdot subgraph\_mask_{nmk}\in\{0,1\}
\]

类别 0：

\[
\Phi_0=\left\{\bar{\phi}_n\mid y_n=0\right\}
\]

类别 1：

\[
\Phi_1=\left\{\bar{\phi}_n\mid y_n=1\right\}
\]

主统计比较对象为 \(\Phi_0\) 与 \(\Phi_1\)。子图级结果仅作为补充描述。

若某个结构指标因空边或其他已记录原因成为 `NaN`，该指标必须使用逐指标有效 mask：

\[
\bar{\phi}_{n,r}
=
\frac{
\sum_{m,k}w_{nmk,r}\phi_r(S_{n,k}^{(m)})
}{
\sum_{m,k}w_{nmk,r}+\varepsilon
}
\]

其中 \(w_{nmk,r}=1\) 仅当时间片、子图和指标 \(r\) 均有效。填充子图和缺失指标不得进入分子或分母，缺失值不得替换为 0。每个样本、每个指标都应保存实际有效子图数。主分析的聚合方式必须在查看 held-out 结果前固定。

## 11. 数据划分与防止泄漏

推荐流程：

- 训练集：训练子图提取器与简单分类头；
- 验证集：调整超参数与选择最佳模型；
- 测试集：冻结模型后提取子图并进行最终结构差异分析。

禁止使用测试集标签调节提取器参数、评分权重、结构指标、对照组规则和显著性阈值。

若数据量较小，使用 \(K\)-fold 交叉验证。每一折只分析 held-out 样本，最终汇总所有 out-of-fold 结果。

多站点数据必须在划分表中显式保存 `site` 和稳定受试者 ID。若同一受试者存在多个扫描，必须使用 group split，禁止同一受试者跨 train/validation/test 或跨交叉验证折。应尽量在类别允许的范围内保持站点分布；对于仅含单一类别或极端不平衡的站点，必须单独报告并在结论中讨论站点混杂，不能仅依赖全局随机分层。

## 12. 对照组设计

所有对照组应尽量匹配关键子图的有效子图数量、节点数、边数、时间片分布、样本分布和 mask。对照组必须来自同一样本、同一有效时间片，不得通过截断原图或复制候选完成匹配。生成 Random 与 Top-degree 对照时必须能够通过 `sample_id`、`site` 和 `time_index` 回查原始图。

### 12.1 Random Subgraph

对每个关键子图 \(S_{n,k}^{(m)}\)，随机对照必须来自同一样本、同一时间片，并匹配：

\[
|V_{rand}|=|V_{key}|
\]

\[
|E_{rand}|=|E_{key}|
\]

若关键子图要求连通，随机对照也应尽量采用连通采样。随机对照至少重复 \(R_{rand}\) 次，推荐 \(R_{rand}\geq 100\)，并保存可复现随机种子。

### 12.2 Top-degree Subgraph

根据：

\[
d_i^{(m)}=\sum_j\left|A_{ij}^{(m)}\right|
\]

选择连接强度最高的节点，并形成规模匹配的诱导子图。

### 12.3 Low-score Subgraph

在同一样本、同一时间片、同一候选池中选择综合评分最低的有效候选。它必须满足与关键子图相同的 \(n_{\min}\)、\(e_{\min}\)、\(n_{\max}\)、\(b_{\max}\) 和去重规则。

若有效低分候选不足 \(K\) 个，应记录实际数量并使用 mask，不得复制候选补足。

Low-score 对照要求硬导出阶段保存同一样本、同一时间片的完整有效候选池，或同时导出已经按相同有效性与去重规则选出的 Low-score 候选。仅保存最终 Top-\(K\) 关键子图不足以事后构造该对照。

## 13. 单变量统计检验

对每一个结构指标 \(\phi_r\)，主检验采用双侧 Mann–Whitney U 检验。

原假设：

\[
H_0:
P(\phi_r\mid Y=0)=P(\phi_r\mid Y=1)
\]

备择假设：

\[
H_1:
P(\phi_r\mid Y=0)\neq P(\phi_r\mid Y=1)
\]

原始显著性阈值：

\[
\alpha=0.05
\]

Mann–Whitney U 用于判断两组分布位置是否存在差异；在没有额外假设时，不能简单解释为均值差异检验。主结果应报告 median、IQR、U 统计量和双侧 \(p\)-value。

## 14. 多重比较校正

对多个指标的原始 \(p\)-value 使用 Benjamini–Hochberg FDR 校正。

\[
q<0.05
\]

结果表格应同时保存原始 \(p\)-value、FDR \(q\)-value、显著性和差异方向。

## 14.1 缺失值与差异方向

每个指标在检验前必须报告两类有效样本数。缺失值不得静默替换为 0。

类别差异方向定义为：

\[
Direction_r
=
sign(T_{1,r}-T_{0,r})
\]

其中主分析推荐：

\[
T_{c,r}=median(\Phi_{c,r})
\]

若使用交叉验证，应预先规定方向稳定阈值，例如至少 \(80\%\) 的折方向一致。

## 15. 标准化类别差异度

\[
\Delta_{\phi_r}=
\frac{|\mu_{1,r}-\mu_{0,r}|}
{\sigma_{0,r}+\sigma_{1,r}+\varepsilon}
\]

其中 \(\mu_{0,r}\)、\(\mu_{1,r}\) 为两类均值，\(\sigma_{0,r}\)、\(\sigma_{1,r}\) 为两类标准差。

整体结构差异度：

\[
\Delta_{total}=
\frac{1}{P}\sum_{r=1}^{P}\Delta_{\phi_r}
\]

## 16. 推荐附加效应量

可同时报告 Cliff's delta：

\[
\delta=P(X_1>X_0)-P(X_1<X_0)
\]

它不替代 \(\Delta_{\phi_r}\)，而是作为补充。

## 17. 可选多变量检验

可对完整样本级结构向量执行 Maximum Mean Discrepancy：

\[
MMD^2(X,Y)=
\frac{1}{n(n-1)}\sum_{i\neq i'}k(x_i,x_{i'})+
\frac{1}{m(m-1)}\sum_{j\neq j'}k(y_j,y_{j'})-
\frac{2}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m}k(x_i,y_j)
\]

显著性通过 permutation test 获得。MMD 仅作为整体分布差异补充。

## 18. 有效性判定标准

关键子图提取器在结构统计层面被认为有效，需要同时满足：

1. 至少一个结构指标在 FDR 校正后显著；
2. 关键子图的显著指标数量大于 Random Subgraph；
3. 关键子图的显著指标数量大于 Top-degree Subgraph；
4. 关键子图的显著指标数量大于 Low-score Subgraph；
5. 关键子图的 \(\Delta_{total}\) 大于三个对照组；
6. 不同数据划分或交叉验证折中的差异方向基本一致。

若只满足部分条件，则只能得出关键子图在部分结构维度上捕捉到类别差异的有限结论。

## 19. 实验流程

1. 在训练集上训练关键子图提取器、子图编码器和简单分类头；
2. 使用验证集选择最佳 checkpoint；
3. 冻结模型参数；
4. 对 held-out 样本的每个有效时间片导出最多 \(K\) 个关键子图及全部 mask；
5. 对每个关键子图计算 \(\phi(S)\)；
6. 按 `time_mask`、`subgraph_mask` 和逐指标有效 mask 聚合样本的所有有效子图，得到 \(\bar{\phi}_n\)；
7. 按标签构建 \(\Phi_0\) 与 \(\Phi_1\)；
8. 执行 Mann–Whitney U、FDR、\(\Delta_{\phi_r}\) 和 \(\Delta_{total}\)；
9. 对三个对照组重复完全相同的流程；
10. 比较显著指标数量、差异强度和折间稳定性，并形成结论。

## 20. 结果表格

### 20.1 逐指标结果

| 指标 | 类别 0 统计量 | 类别 1 统计量 | 原始 p-value | FDR q-value | 差异方向 | \(\Delta_{\phi_r}\) | 效应量 | 是否显著 |
|---|---:|---:|---:|---:|---|---:|---:|---|

对于非正态数据，建议报告 median 和 IQR，同时可保留 mean 和 std。

### 20.2 对照组比较

| 子图来源 | 显著指标数量 | 平均 FDR q-value | \(\Delta_{total}\) | 方向稳定性 | 结论 |
|---|---:|---:|---:|---|---|
| Key Subgraph |  |  |  |  |  |
| Random Subgraph |  |  |  |  |  |
| Top-degree Subgraph |  |  |  |  |  |
| Low-score Subgraph |  |  |  |  |  |

## 21. 可视化输出

- 箱线图或小提琴图：展示每个指标的类别分布；
- 差异热力图：展示不同子图来源的 \(\Delta_{\phi_r}\)；
- UMAP 或 t-SNE：展示样本级结构向量的二维分布，仅作为描述性证据；
- 折间稳定性图：展示不同折中的差异方向和效应强度。

## 22. 输出文件建议

```text
outputs/
├── exclusion_manifest.csv
├── split_manifest.csv
├── subgraph_level_metrics.csv
├── sample_level_metrics.csv
├── univariate_test_results.csv
├── control_group_comparison.csv
├── fold_stability.csv
├── mmd_results.json
└── figures/
    ├── boxplots/
    ├── discrepancy_heatmap.png
    ├── umap.png
    └── fold_stability.png
```

## 23. Codex 实现接口

统计模块至少读取：

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

`original_edge_weights` 必须是带符号原始边权；`edge_presence_threshold` 必须与训练、硬导出所用的 \(\tau_{edge}\) 一致。统计模块应在入口处验证二者，不得自行取绝对值覆盖原字段，也不得重新估计阈值。

推荐拆分为：

```python
compute_subgraph_metrics(...)
aggregate_sample_metrics(...)
generate_random_controls(...)
generate_top_degree_controls(...)
select_low_score_controls(...)
run_mannwhitney_tests(...)
apply_bh_fdr(...)
compute_discrepancy(...)
compute_effect_sizes(...)
summarize_fold_stability(...)
```

最小测试：

```python
assert node_count >= 2
assert edge_count >= 1
assert 0.0 <= density <= 1.0 + tolerance
assert (edge_weights.abs() > edge_presence_threshold).all()
if positive_edge_count > 0:
    assert (positive_edge_weights > edge_presence_threshold).all()
    assert 0.0 <= positive_intra_ratio <= 1.0
    assert 0.0 <= positive_inter_ratio <= 1.0
    assert abs(positive_intra_ratio + positive_inter_ratio - 1.0) <= tolerance
if negative_edge_count > 0:
    assert (negative_edge_weights < -edge_presence_threshold).all()
    assert 0.0 <= negative_intra_ratio <= 1.0
    assert 0.0 <= negative_inter_ratio <= 1.0
    assert abs(negative_intra_ratio + negative_inter_ratio - 1.0) <= tolerance
assert abs_connection_sum >= 0.0
assert positive_connection_sum >= 0.0
assert negative_connection_magnitude_sum >= 0.0
assert sample_level_table["sample_id"].is_unique
assert padding_invariance_error < tolerance
assert truncated_node_count == 0
assert truncated_time_count == 0
```

无向图必须额外验证没有同时重复统计 \((i,j)\) 与 \((j,i)\)，负边不会被边存在判断删除，且统计实现中不存在正负边权直接求平均导致的抵消。仅当相应符号边集合非空时，才执行社区比例和为 1 的断言。还必须验证：改变 padding 区域中的任意有限值不会改变子图级指标、样本级聚合或统计检验；有效子图数量不同的样本不会因固定 \(M\times K\) 分母而产生偏差。

## 24. 与原始设计的一致性说明

本文保留了原始 Word 方案中的规模、密度、连接强度、动态变化和社区结构指标，并针对真实带符号网络将连接强度与社区结构扩展为总体绝对量、正连接量和负连接幅值，避免正负抵消。样本级聚合、三类对照组、Mann–Whitney U、Benjamini–Hochberg FDR、\(\Delta_{\phi_r}\)、\(\Delta_{total}\)、判定标准、结果表格、可视化和结论边界保持不变。

唯一边计数、变长序列、节点身份对齐、缺失值规则、随机重复、mask 聚合、异常样本清单和代码接口属于实现澄清，不改变实验假设。

## 25. 结论边界

本模块能够验证的是关键子图是否在结构统计层面保留了与类别相关的差异。

本模块不能单独证明：

- 完整分类框架最优；
- 后续子图演化模块有效；
- 关键子图具有因果意义；
- 结构差异在所有数据集上普遍成立。

当主要判定标准满足时，可表述为：训练后的关键子图提取器在 held-out 样本中提取出了具有稳定类别结构差异的关键子图，并且这种差异强于随机、Top-degree 和 Low-score 对照组，说明提取器保留了与分类任务相关的结构信息。

## 26. 当前全样本探索性分析配置

当前实现另行冻结 `all_samples_exploratory` 协议，将 938 个有效样本全部用于
提取器训练，并从同一批样本导出关键子图进行 A/B 类结构差异统计。此配置是
用户明确选择的队列内探索路径，是第 11 节严格泛化评估流程的替代实验配置，
而不是 held-out 验证。

该路径仍必须固定样本索引、边阈值、硬提取参数、对照组规则、统计指标和随机
种子；所有导出与报告必须标记 `split: all` 和
`exploratory_in_sample_analysis: true`。允许报告“在当前分析队列中观察到的结构
差异”，但不得使用“测试集性能”“外部泛化”“未见样本上的稳定差异”等表述。
分类 AUROC、balanced accuracy 和 F1 只能作为训练诊断，不能作为泛化证据。

如果未来需要确认差异能否推广到未见样本，应恢复第 11 节的 group-aware
held-out 或 out-of-fold 设计，并使用全新的协议和 checkpoint；不得复用本路径
所得的队列内显著性作为独立验证结果。
