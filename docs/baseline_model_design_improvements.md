# 统一基线模型设计修订说明

## 1. 修订对象与目的

本次修订以 `docs/baseline_model_design_and_acceptance_for_codex_fixed.md` 为正式设计文档，在保留原有 Signed Subgraph Encoder、窗口内聚合、单向 GRU 和 MLP 分类头主流程的前提下，解决实验可识别性、数据泄漏、控制匹配和统计验收方面的问题。

本次只修订设计文档，没有实现或修改模型程序。

## 2. 主要改进

### 2.1 增加证据等级

新增两种证据等级：

- `exploratory_in_sample`：允许使用全样本训练提取器的导出结果，只用于发现候选现象；
- `confirmatory_cross_fitted`：提取器、先验和分类基线均不能接触外层测试标签，用于机制验证和泛化判断。

明确规定全样本探索结果不能表述为独立验证或子图传递有效性的证明。

### 2.2 明确硬子图数据接口

现有硬子图 JSON 没有直接保存完整节点特征，因此补充了以下加载流程：

1. 根据 `original_graph_ref` 加载原始 `.pt`；
2. 根据 `time_index` 选择时间窗；
3. 根据 `node_ids` 重建子图节点；
4. 将全局 `edge_index` 重映射为子图局部编号；
5. 核对边权、阈值、checkpoint hash 和 protocol hash。

同时明确标签只能作为监督目标，不能进入特征构造。

### 2.3 冻结节点特征语义

中性基线默认设置为：

```yaml
node_feature_context: induced_subgraph
include_temporal_deltas: false
use_node_identity: false
```

即在诱导子图内部重新计算正负连接强度、比例和社区结构特征，以检验子图本身的判别结构。

原图上下文节点特征被保留为独立消融，不能与纯子图结果混合解释。社区编号仅用于同社区判断和结构特征计算，禁止直接 embedding。

### 2.4 禁止提取分数泄漏

明确禁止把以下字段作为分类模型输入：

- node/edge score；
- candidate score；
- seed node；
- connectivity/dynamic 等候选排序分数；
- 由真实标签直接构造的逐样本字段。

这样可以避免 key 子图比控制子图获得额外输入。

### 2.5 统一带符号边存在规则

统一规定：

\[
M_{ij}^{\mathrm{edge}}
=
\mathbf 1(|A_{ij}|>\tau_{\mathrm{edge}}).
\]

训练、控制子图、原图对照和统计模块必须复用数据协议中的同一个阈值。

### 2.6 补充 Signed Message Passing 的信息约束

指出正负通道分别归一化会弱化各通道总强度，因此节点输入必须保留：

- 正连接强度；
- 负连接幅值；
- 正负连接比例。

默认消息传递不使用提取分数，也不使用时间差分边特征。差分边特征被调整为独立消融变量。

### 2.7 修正历史模式定义

删除了把“每个时间窗都清零状态”作为独立 `reset_state` 实验的设计，因为在最终只读取最后状态时，它与 `current_only` 数学等价。

重新定义为：

- `full`：使用完整递归历史；
- `current_only`：只使用最后有效窗口；
- `truncate_history`：按比例保留最后若干窗口，形成历史保留曲线；
- `independent_bag`：使用全部窗口，但不使用递归或顺序。

这使历史内容、历史长度、递归传递和顺序贡献能够被分别检验。

### 2.8 排除顺序实验中的差分泄漏

主顺序实验强制关闭：

```yaml
include_temporal_deltas: false
use_time_position: false
```

随后比较 ordered、多个 shuffled permutation 和 permutation-invariant Bag。

差分特征被单独设置为 ordered 模式下的可学习性消融，不能再被当作纯顺序证据。

### 2.9 改进多来源控制匹配

规定 key、low-score、Top-degree 和 random 的主要比较必须基于冻结的 matched-control manifest，至少匹配：

- sample；
- time window；
- subgraph index；
- node count；
- edge count；
- 有效窗口和有效子图集合。

任一主要来源匹配失败时，该 tuple 从所有配对来源共同排除，不能在某一来源中静默缺失。

每个 random repeat 被定义为独立控制数据集，禁止把多个随机重复同时池化到同一时间窗。raw graph 被定位为补充的压缩/去噪对照，而不是尺寸匹配控制。

### 2.10 分离结构特征和统计先验作用

将原来的先验比较拆分为五组：

| 组别 | 结构输入 | 先验 |
|---|---|---|
| A | 零向量 | none |
| B | 真实结构特征 | none |
| C | 真实结构特征 | uniform |
| D | 真实结构特征 | real |
| E | 真实结构特征 | permuted |

由 B 对 A 检验结构特征本身，由 D 对 E 检验真实先验维度映射。避免同时改变结构特征和先验后无法判断改进来源。

### 2.11 增加扰动剂量实验

增加以下目标扰动方向：

- 高分节点；
- 高分边；
- 正边和负边；
- 不同时间阶段；
- 早期关键子图替换。

建议使用 0%、10%、25%、50% 的扰动强度，并与相同数量和尺寸的随机扰动比较，以检验是否存在稳定剂量—反应关系。

### 2.12 增加交叉拟合流程

补充了确认性实验的完整外层流程：

1. group-aware 外层划分；
2. 只用训练折训练提取器；
3. 只用训练折估计标准化参数、先验、阈值和类别权重；
4. 用冻结提取器导出未见 test 子图；
5. 在外层未见样本上产生最终预测；
6. 汇总跨折未见预测进行配对统计。

同一 subject 的 session 必须留在同一集合，并以 subject 为 bootstrap 单位。

### 2.13 增加统计验收规则

将未见样本上的 unweighted log-loss 设为主要理论指标，定义配对差异：

\[
\Delta_{\mathrm{LL}}
=
\mathcal L_{\mathrm{control}}
-
\mathcal L_{\mathrm{target}}.
\]

新增要求包括：

- subject-level paired bootstrap；
- 95% confidence interval；
- 原始 p-value；
- FDR-adjusted q-value；
- 每个 seed 和 fold 的结果；
- 有效配对样本数。

同时规定了历史、顺序、key 特异性、真实先验和目标扰动各自最低限度的支持条件。

### 2.14 改进 checkpoint 和分类阈值规则

默认使用 validation unweighted log-loss 选择机制实验的最佳 checkpoint。需要硬分类的指标必须在 validation 上选择阈值，并把阈值写入 checkpoint；test 不得重新调节。

### 2.15 扩充自动测试和审计

新增以下验收类别：

- 数据模式和标签泄漏测试；
- 历史模式等价性测试；
- Bag 排列不变性；
- shuffle 内容守恒；
- matched-control 完整性；
- paired statistics 和 FDR 测试；
- artifact hash 一致性；
- validation/test 不参与训练统计量估计。

### 2.16 增加工程性能建议

允许在不使用 PyTorch Geometric 的条件下，将 batch 内子图扁平化，通过局部边索引全局化和 `index_add_` 进行消息聚合，以减少大量小图的 Python 循环开销。

优化实现必须与逐子图参考实现进行数值等价测试，且不得截断节点、子图或时间窗。

### 2.17 增加输出产物要求

每次实验至少记录：

- 完整配置；
- split、protocol、export 和 checkpoint hash；
- matched-control manifest；
- validation threshold；
- 每样本概率和真实标签；
- weighted train loss 与 unweighted evaluation log-loss；
- bootstrap/FDR 结果；
- 参数量、时间、显存和软件版本；
- evidence level 和已知泄漏风险。

## 3. 保持不变的核心设计

以下主流程没有改变：

```text
两层正负分离消息传递
→ Mean/Max 子图池化
→ 时间窗内有效子图均值聚合
→ 单层单向 GRU
→ 两层 MLP 分类器
```

仍然保持：

- 支持 signed graph；
- 支持可变时间窗、子图数和节点数；
- 不使用 PyTorch Geometric；
- Python 3.7、PyTorch 1.13.1 兼容；
- 训练可使用 weighted CE；
- 理论评估使用 unweighted log-loss；
- 第一阶段不引入 Transformer、Attention 或复杂时序模块。

## 4. 文档检查结果

修订后设计文档完成了以下静态检查：

- UTF-8 读取正常；
- ASCII 非法控制字符数量为 0；
- Markdown fenced code block 数量成对；
- `\[` 与 `\]` 数量一致；
- `\begin{cases}` 与 `\end{cases}` 数量一致；
- 两处残留的 `\right` 公式错误已修复；
- 可变长度 GRU 分段公式的换行符已修复。

## 5. 修订后的作用

修订后的设计能够更明确地区分以下结论：

- 历史窗口集合有用；
- 递归状态传递有用；
- 时间顺序有用；
- 关键子图优于尺寸匹配控制；
- 结构特征本身有用；
- 真实统计先验映射有用；
- 关键结构对定向扰动具有特异敏感性。

这些区分能够避免把普通历史累积、模型容量差异、标签泄漏或子图尺寸差异错误解释为新的子图演化机制，为后续扩展理论和形成可验证创新点提供更可靠的实验基础。
