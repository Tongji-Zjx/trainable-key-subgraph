# D3-B Variation-Temporal Residual：加入学习型时序编码器的架构与实验设计

## 文档状态

- **版本**：V1.0
- **任务**：带符号动态脑图序列二分类
- **基础架构**：D3-B Variation-Only Exact-Head
- **扩展目标**：在尽量保留现有有效分类路径的前提下，引入学习型时序编码器，利用谱变化序列中的时间顺序信息提升分类性能
- **推荐模型名称**：

\[
\boxed{
\text{D3-B Variation-Temporal Residual}
}
\]

---

# 0. 冻结实施修正（V1.1）

以下修正优先于本文后续可能存在的简化表述，并作为程序实现、测试和
实验验收的冻结契约：

1. **独立时序缓存**：现有Exact-SGW缓存只包含样本级
   `core/variation/representation`和transition mask，不包含BiGRU需要的
   逐转移16维序列。新增独立时序缓存与manifest，不修改现有Exact-SGW
   artifact schema。
2. **严格复现基础logit**：T4使用的 \(z_B\) 必须由当前冻结selector、
   Proxy、Exact train-only scaler和Exact分类头按正式B路径计算并缓存；
   禁止用Exact variation近似替代Proxy variation。
3. **逐步特征来源**：时序分支使用冻结硬关键图上重新计算的Exact谱分位
   绝对差，不计算Exact GW；其masked mean必须与现有Exact 16维variation
   在数值容差内一致。
4. **非连续mask压缩**：无效transition可能出现在序列中间。进入GRU前，
   必须按原时间顺序压缩所有有效transition，再padding和packing，不能把
   transition数量直接当作前缀长度。
5. **空序列恒等回退**：没有有效transition的样本必须强制
   \(h_T=0,z_T=0\)，从而保证T4中
   \(z_{\mathrm{final}}=z_B\)。禁止MLP或分类头偏置为其产生非零修正。
6. **scaler为确定性拟合**：逐转移16维scaler只由train中的有效transition
   统计得到，不是可训练参数；validation和test不得参与拟合。
7. **最终阈值重新冻结**：原D3-B阈值0.509430只属于T0。T4概率改变后必须
   在validation重新选择阈值，再原样用于test。
8. **T1不是严格参数匹配对照**：判断时间顺序贡献时，除比较T3与T1外，
   必须加入固定seed的样本内顺序打乱诊断；不得只凭T3>T1归因于顺序。
9. **测试集使用边界**：T0–T4架构选择、early stopping和消融判断只使用
   train/validation；test仅用于最终冻结候选的正式评估。
10. **基础工程保持不变**：现有D3-B、all-34 Proxy、selector、Exact-SGW
    scaler和分类头代码及产物继续保留。第一阶段时序实验不得覆盖这些文件。

---

# 1. 设计背景

当前已验证架构为 D3-B Variation-Only Exact-Head。

其正式推理路径为：

```text
带符号动态图序列
→ 学习型硬关键图选择器
→ 硬关键子图序列
→ 34维Proxy表示
→ 屏蔽前18维core
→ 仅保留18–33维variation
→ 冻结Exact-SGW scaler
→ 冻结Exact-SGW分类头
→ validation冻结阈值
→ 二分类结果
```

当前正式结果：

\[
\text{Test AUROC}=0.611798
\]

\[
\text{Test Accuracy}=0.607143
\]

\[
\text{Test Balanced Accuracy}=0.599111.
\]

当前较好结果主要来自16维谱变化幅度：

\[
h_{\mathrm{variation},k}
=
\operatorname{mean}_m
\left|
q_{m+1,k}-q_{m,k}
\right|,
\qquad
k=0,\ldots,15.
\]

该设计能够描述整个样本中各谱分位的平均变化强度，但在时间平均过程中丢失了：

- 变化发生的先后顺序；
- 变化持续的时间；
- 变化是否集中在某一阶段；
- 前期异常与后期异常的区别；
- 局部波动是否具有连续性。

因此，引入学习型时序编码器是可行且有明确动机的。

---

# 2. 核心设计原则

## 2.1 不直接替换现有D3-B路径

现有D3-B已经取得较好结果，因此不建议立即删除原始variation平均路径。

推荐保留：

\[
z_B
\]

作为原有D3-B基础分类logit，再增加一个学习型时序分支：

\[
z_T.
\]

最终通过残差方式融合：

\[
\boxed{
z_{\mathrm{final}}
=
z_B+\alpha z_T.
}
\]

这样可以最大程度保留当前有效决策边界。

## 2.2 时序编码器必须放在时间平均之前

当前样本级variation表示为：

\[
H_{\mathrm{variation}}
=
\operatorname{mean}_m
|\Delta Q^{(m)}|.
\]

一旦已经求平均，时间顺序就无法恢复。

因此，学习型时序编码器的输入应为逐步变化序列：

\[
V_b
=
\left[
v_b^{(1)},
\ldots,
v_b^{(T_b)}
\right],
\]

其中：

\[
v_b^{(m)}
=
\left|
Q_b^{(m+1)}-Q_b^{(m)}
\right|
\in\mathbb R^{16},
\]

\[
T_b=M_b-1.
\]

## 2.3 第一版只编码16维variation序列

第一版不建议直接使用全部34维逐步SGW信息。

推荐输入：

\[
\boxed{
v^{(m)}
=
|\Delta Q^{(m)}|
\in\mathbb R^{16}
}
\]

原因：

1. 当前实验已经表明variation-only优于all-34 Proxy；
2. 16维输入更容易训练；
3. 参数量较小；
4. 更容易判断性能提升是否来自时间顺序；
5. 避免谱方向、spectral speed和GW proxy speed引入额外噪声。

---

# 3. 总体架构

```text
可变长度带符号动态图序列
              │
              ▼
     冻结的硬关键图选择器
              │
              ▼
      硬关键子图序列 U(1:M)
              │
              ▼
   每个窗口计算16维谱分位 Q(m)
              │
              ▼
逐步variation序列 |Q(m+1)-Q(m)|
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
原D3-B静态分支   学习型时序分支
mean variation   BiGRU
        │           │
冻结Exact Head   Mean/Max Pooling
        │           │
        ▼           ▼
       z_B         z_T
        │           │
        └─────┬─────┘
              ▼
       残差Logit融合
              ▼
       z_final = z_B + α z_T
              ▼
        二分类概率与标签
```

---

# 4. 输入与数据协议

每个样本为：

\[
\mathcal G_b
=
\left\{
G_b^{(1)},
\ldots,
G_b^{(M_b)}
\right\},
\]

其中：

\[
G_b^{(m)}
=
\left(
A_b^{(m)},
C_b^{(m)},
I_b^{(m)}
\right).
\]

- \(A_b^{(m)}\)：带符号加权邻接矩阵；
- \(C_b^{(m)}\)：社区编号；
- \(I_b^{(m)}\)：节点名称或稳定ID；
- \(M_b\)：样本相关窗口数；
- \(N_b^{(m)}\)：窗口相关节点数。

必须保持：

- 不使用空间坐标；
- 不截断时间窗口；
- 不截断节点；
- 使用list-based variable-length batching；
- padding不参与时序编码和池化；
- 同一主体不跨训练、验证和测试集合；
- 所有标准化统计量只由训练集拟合。

---

# 5. 冻结的硬关键图选择器

第一版时序扩展中，选择器保持冻结。

选择器包括：

- 15维节点特征；
- 6维边特征；
- 节点残差评分器；
- 边残差评分器；
- 社区覆盖节点Top-k；
- 候选边Top-k；
- STE硬选择；
- 节点保留比例 \(r_n=0.50\)；
- 边保留比例 \(r_e=0.30\)。

得到硬关键子图序列：

\[
\mathcal U_b
=
\left\{
U_b^{(1)},
\ldots,
U_b^{(M_b)}
\right\}.
\]

第一版不重新训练selector，以避免：

- 硬图发生变化；
- 现有D3-B基线失效；
- 重新导出所有子图；
- 重新拟合scaler；
- 重新训练Exact分类头；
- 无法归因性能变化。

---

# 6. 逐窗口谱分位状态

对每个硬关键图 \(U_b^{(m)}\) 计算带符号正则化拉普拉斯。

绝对度矩阵：

\[
D_{ii}^{(m)}
=
\sum_j
|A_{ij}^{(m)}|.
\]

带符号归一化拉普拉斯：

\[
\mathcal L_{\eta}^{(m)}
=
I
-
\left(
D^{(m)}+\eta I
\right)^{-1/2}
A^{(m)}
\left(
D^{(m)}+\eta I
\right)^{-1/2}.
\]

其中：

\[
\eta=10^{-3}.
\]

对拉普拉斯特征值经验分布提取16个固定分位点：

\[
Q_b^{(m)}
=
\left[
q_{b,m,1},
\ldots,
q_{b,m,16}
\right]
\in\mathbb R^{16}.
\]

推荐使用与当前D3-B一致的分位点配置。

---

# 7. 时序输入构造

相邻窗口的绝对谱变化：

\[
v_b^{(m)}
=
\left|
Q_b^{(m+1)}
-
Q_b^{(m)}
\right|
\in
\mathbb R^{16}.
\]

整个样本形成：

\[
V_b
=
\left[
v_b^{(1)},
v_b^{(2)},
\ldots,
v_b^{(T_b)}
\right],
\]

其中：

\[
T_b=M_b-1.
\]

当样本没有有效相邻窗口时：

- 时序分支输出零向量；
- time mask全部为无效；
- 不允许产生NaN或Inf；
- 该样本仍可通过基础D3-B分支完成预测。

---

# 8. 逐转移特征标准化

逐步variation序列必须单独拟合train-only scaler。

对第 \(k\) 维：

\[
\mu_k^{\mathrm{step}}
=
\operatorname{mean}_{b,m\in\mathrm{train}}
v_{b,m,k},
\]

\[
\sigma_k^{\mathrm{step}}
=
\sqrt{
\operatorname{mean}_{b,m\in\mathrm{train}}
\left(
v_{b,m,k}
-
\mu_k^{\mathrm{step}}
\right)^2
+\epsilon
}.
\]

标准化：

\[
\widetilde v_{b,m,k}
=
\frac{
v_{b,m,k}
-
\mu_k^{\mathrm{step}}
}{
\sigma_k^{\mathrm{step}}
+\epsilon
}.
\]

注意：

\[
\boxed{
\text{不能直接复用现有34维样本级Exact-SGW scaler。}
}
\]

因为：

- 当前scaler作用于样本级34维特征；
- 新scaler作用于逐时间步16维特征；
- 两者统计对象、维度和分布不同。

---

# 9. 学习型时序编码器

## 9.1 推荐结构：一层BiGRU

输入：

\[
\widetilde V_b
\in
\mathbb R^{T_b\times16}.
\]

通过一层BiGRU：

\[
H_b^{(1:T_b)}
=
\operatorname{BiGRU}
\left(
\widetilde V_b
\right).
\]

配置：

```yaml
temporal_encoder:
  type: bigru
  input_dim: 16
  hidden_dim_per_direction: 32
  num_layers: 1
  bidirectional: true
  recurrent_dropout: 0.0
```

每个时间步输出：

\[
H_b^{(m)}
\in
\mathbb R^{64}.
\]

其中：

- 正向GRU输出32维；
- 反向GRU输出32维；
- 拼接后为64维。

## 9.2 为什么第一版使用小型BiGRU

当前训练样本数量有限，因此建议：

- 只使用一层；
- 每个方向32维；
- 不使用多层堆叠；
- 不使用大隐藏维度；
- 不在GRU内部使用dropout；
- 只在后续MLP中使用dropout。

这样能够降低过拟合风险。

---

# 10. 时间池化

第一版推荐：

\[
h_{\mathrm{mean}}
=
\operatorname{MeanMask}_m
H_b^{(m)},
\]

\[
h_{\mathrm{max}}
=
\operatorname{MaxMask}_m
H_b^{(m)}.
\]

拼接：

\[
h_{\mathrm{pool}}
=
\left[
h_{\mathrm{mean}};
h_{\mathrm{max}}
\right]
\in
\mathbb R^{128}.
\]

再投影为：

\[
h_{\mathrm{temporal}}
=
MLP_T
\left(
h_{\mathrm{pool}}
\right)
\in
\mathbb R^{32}.
\]

推荐：

```text
Linear(128,64)
→ GELU
→ Dropout(0.20)
→ Linear(64,32)
```

第一版暂不加入时间注意力。

原因：

- BiGRU已经增加学习能力；
- mean和max更稳定；
- 更容易判断是否真正存在时间顺序增益；
- 注意力可能在小样本下过拟合到少量窗口。

---

# 11. 时序辅助分类头

时序分支独立输出：

\[
z_T
=
Classifier_T
\left(
h_{\mathrm{temporal}}
\right)
\in
\mathbb R^2.
\]

推荐结构：

```text
Linear(32,32)
→ GELU
→ Dropout(0.20)
→ Linear(32,2)
```

该辅助头用于：

1. 保证BiGRU具有独立分类能力；
2. 防止残差融合系数过小时序分支学不到有效表示；
3. 支持Temporal-only实验；
4. 判断时序分支是否真正包含额外分类信息。

---

# 12. 原D3-B基础分支

保留当前正式D3-B路径：

1. 对Proxy 34维表示屏蔽前18维；
2. 仅保留18–33维variation；
3. 使用冻结的Exact-SGW train-only scaler；
4. 使用冻结的Exact-SGW分类头；
5. 输出基础logit：

\[
z_B
\in
\mathbb R^2.
\]

第一阶段中，以下模块全部冻结：

- selector；
- Exact-SGW scaler；
- Exact分类头；
- 当前D3-B静态推理路径。

---

# 13. 残差Logit融合

最终logit：

\[
\boxed{
z_{\mathrm{final}}
=
z_B
+
\alpha z_T
}
\]

其中：

\[
\alpha
=
\sigma(a)
\in(0,1).
\]

推荐初始化：

\[
\alpha_0=0.10.
\]

可令：

\[
a_0
=
\log
\frac{0.1}{0.9}.
\]

这样模型训练初期：

\[
z_{\mathrm{final}}
\approx
z_B,
\]

时序分支只对当前有效预测进行小幅修正。

## 13.1 为什么不直接特征拼接

不推荐第一版直接将：

\[
[H_{\mathrm{variation}};h_{\mathrm{temporal}}]
\]

拼接后重新训练一个全新分类头。

原因：

- 可能破坏当前已验证决策边界；
- 无法充分利用冻结Exact分类头；
- 可能导致模型重新依赖噪声特征；
- 更难判断性能变化来源；
- 在有限样本上更容易过拟合。

残差logit融合更适合作为第一版安全扩展。

---

# 14. 损失函数

最终分类损失：

\[
\mathcal L_{\mathrm{final}}
=
CE
\left(
z_{\mathrm{final}},
y
\right).
\]

时序辅助损失：

\[
\mathcal L_{\mathrm{temporal}}
=
CE
\left(
z_T,
y
\right).
\]

总损失：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{final}}
+
\lambda_T
\mathcal L_{\mathrm{temporal}}
}
\]

推荐：

\[
\lambda_T=0.30.
\]

训练集类别权重应与当前主实验一致。

---

# 15. 训练阶段设计

## 阶段一：冻结基础路径，只训练时序分支

冻结：

- 节点评分器；
- 边评分器；
- selector；
- 硬选择预算；
- Exact-SGW scaler；
- Exact分类头；
- D3-B基础分支。

训练：

- 逐步variation scaler；
- BiGRU；
- 时间池化MLP；
- 时序辅助分类头；
- 残差融合系数 \(\alpha\)。

目标：

\[
\boxed{
\text{验证时间顺序是否能够在当前D3-B基础上提供额外增益。}
}
\]

## 阶段二：可选解冻Exact分类头

只有当阶段一验证集结果稳定优于D3-B时，才进行。

解冻：

- Exact分类头；
- 可选D3-B最后一层。

保持冻结：

- selector；
- 硬关键图；
- 谱特征计算；
- 逐步variation定义。

建议学习率：

```yaml
learning_rates:
  temporal_encoder: 1.0e-3
  temporal_pooling: 1.0e-3
  temporal_head: 1.0e-3
  fusion_alpha: 1.0e-3
  exact_head: 1.0e-4
  selector: 0.0
```

## 阶段三：是否联合微调selector

第一轮不推荐。

只有当：

1. BiGRU时序分支稳定有效；
2. Temporal-only明显高于随机；
3. 残差融合稳定优于D3-B；
4. 多随机种子结果一致；

才考虑解冻selector。

一旦解冻selector，必须重新：

- 训练selector；
- 导出硬关键图；
- 计算Proxy与Exact特征；
- 拟合scaler；
- 训练分类头；
- 选择验证阈值；
- 评估全部对照实验。

---

# 16. 推荐实验分组

## T0：当前D3-B

```text
mean(|ΔQ|)
→ 冻结Exact Head
```

作为当前正式基线。

## T1：重新训练的Variation-Mean分类器

```text
mean(|ΔQ|)
→ 新训练MLP分类头
```

目的：

- 排除性能变化只是来自重新训练分类头；
- 为时序模型提供参数量更公平的对照。

## T2：Variation-UniGRU

```text
|ΔQ(1:T)|
→ 单向GRU
→ Mean/Max Pooling
→ 分类
```

目的：

- 判断只使用过去到未来的递归建模是否有效。

## T3：Variation-BiGRU

```text
|ΔQ(1:T)|
→ BiGRU
→ Mean/Max Pooling
→ 分类
```

目的：

- 判断双向上下文是否优于单向GRU。

## T4：D3-B + BiGRU残差融合

```text
冻结D3-B
+
Variation-BiGRU
→ z_final = z_B + αz_T
```

这是最终推荐模型。

---

# 17. 关键判定关系

需要满足：

\[
\boxed{
T3>T1
}
\]

才能说明性能提升主要来自时间顺序建模，而不是额外参数量。

需要满足：

\[
\boxed{
T4>T0
}
\]

才能说明学习型时序分支对当前正式D3-B提供了实际增益。

若：

\[
T3\approx T1,
\]

说明时间顺序贡献有限，性能变化主要来自新分类器。

若：

\[
T4\leq T0,
\]

说明时序分支引入噪声或融合方式不合适。

---

# 18. 推荐训练配置

```yaml
model:
  name: d3b_variation_temporal_residual
  num_classes: 2

selector:
  checkpoint: reuse_current_best
  frozen: true
  node_ratio: 0.50
  edge_ratio: 0.30

base_d3b:
  enabled: true
  frozen: true
  exact_head: reuse_current_frozen_head
  scaler: reuse_current_train_only_scaler
  threshold: reuse_current_validation_threshold

temporal_features:
  source: exact_hard_graph_spectral_quantiles
  feature: absolute_spectral_quantile_delta
  input_dim: 16
  standardize_from_train_only: true
  invalid_transition_policy: mask

temporal_encoder:
  type: bigru
  input_dim: 16
  hidden_dim_per_direction: 32
  num_layers: 1
  bidirectional: true
  recurrent_dropout: 0.0

temporal_pooling:
  methods:
    - masked_mean
    - masked_max
  projection_hidden_dim: 64
  output_dim: 32
  dropout: 0.20

temporal_classifier:
  hidden_dim: 32
  dropout: 0.20
  output_dim: 2

fusion:
  type: residual_logit
  formula: z_final = z_base + alpha * z_temporal
  learnable_alpha: true
  alpha_initial: 0.10

loss:
  final_ce_weight: 1.0
  temporal_aux_ce_weight: 0.30
  use_class_weights: true

optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

training:
  max_epochs: 60
  early_stopping_patience: 10
  gradient_clip_norm: 1.0
  checkpoint_metric: validation_auroc
  seeds:
    - 42
    - 43
    - 44
```

---

# 19. 变长序列与Mask处理

样本的有效转移数为：

\[
T_b=M_b-1.
\]

必须使用：

- `sequence_length`；
- `time_mask`；
- `pack_padded_sequence`或等价实现；
- mask-aware mean；
- mask-aware max。

禁止：

- 将padding输入BiGRU并当作真实数据；
- 用零向量替代真实缺失转移且不提供mask；
- 截断较长样本；
- 使用最后隐藏状态而忽略不同长度影响。

推荐不直接使用最后隐藏状态，而使用：

\[
\operatorname{MeanMask}
+
\operatorname{MaxMask}.
\]

---

# 20. 评价指标

主指标：

\[
\boxed{
\text{Validation/Test AUROC}
}
\]

同时报告：

- Accuracy；
- Balanced Accuracy；
- F1；
- Sensitivity；
- Specificity；
- 混淆矩阵；
- 每个随机种子的结果；
- 均值和标准差；
- 最佳epoch；
- 学习到的 \(\alpha\)；
- Temporal-only性能；
- 基础D3-B性能；
- 残差融合性能。

---

# 21. 结果记录表

| 模型 | Val AUROC | Test AUROC | Test BA | Test Accuracy | F1 | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| T0 D3-B |  |  |  |  |  |  |
| T1 Variation-Mean MLP |  |  |  |  |  |  |
| T2 Variation-UniGRU |  |  |  |  |  |  |
| T3 Variation-BiGRU |  |  |  |  |  |  |
| T4 D3-B + BiGRU Residual |  |  |  |  |  |  |

---

# 22. 必做消融

## 22.1 时序顺序打乱

在测试或验证中随机打乱variation序列顺序：

\[
[v^{(1)},\ldots,v^{(T)}]
\rightarrow
[v^{(\pi(1))},\ldots,v^{(\pi(T))}].
\]

若BiGRU性能明显下降，说明模型真正利用了顺序信息。

## 22.2 单向与双向对比

比较：

\[
\text{UniGRU}
\quad\text{vs.}\quad
\text{BiGRU}.
\]

用于判断反向上下文是否有效。

## 22.3 Mean-only与Mean+Max

比较：

- masked mean；
- masked max；
- mean + max。

用于验证池化方式。

## 22.4 固定Alpha与可学习Alpha

比较：

\[
\alpha=0.1
\]

与：

\[
\alpha=\sigma(a).
\]

用于判断自适应融合是否必要。

## 22.5 无时序辅助头

删除：

\[
CE(z_T,y).
\]

用于判断时序辅助监督是否防止分支被忽略。

---

# 23. 第二轮可扩展设计

在16维variation时序模型验证有效后，可进一步尝试：

## 23.1 18维Core时序输入

\[
\Gamma^{(m)}
=
\left[
\Delta Q^{(m)};
v_{\mathrm{spec}}^{(m)};
v_{\mathrm{GW}}^{(m)}
\right]
\in\mathbb R^{18}.
\]

## 23.2 34维逐步联合输入

\[
x^{(m)}
=
\left[
\Delta Q^{(m)};
|\Delta Q^{(m)}|;
v_{\mathrm{spec}}^{(m)};
v_{\mathrm{GW}}^{(m)}
\right]
\in\mathbb R^{34}.
\]

## 23.3 时间注意力

在BiGRU后增加：

\[
\alpha_m
=
\operatorname{softmax}_m
\left(
w^\top\tanh(WH^{(m)})
\right).
\]

只应在mean+max已经验证后再加入。

---

# 24. 理论边界

当前理论直接对应固定聚合后的SGW核心表示，例如：

\[
H_{\mathrm{core}}
=
\operatorname{mean}_m
\Gamma^{(m)}.
\]

加入BiGRU后：

\[
H_{\mathrm{BiGRU}}
=
f_\theta
\left(
\Gamma^{(1:T)}
\right)
\]

是学习型表示。

因此不能声称：

- BiGRU输出继承原有类别间隔下界；
- 最终残差logit继承SGW理论保证；
- 学习型时序编码必然保持谱–GW演化距离；
- 最终概率具有相同Wasserstein间隔。

正确表述是：

\[
\boxed{
\text{固定SGW路径保留理论对齐，BiGRU属于经验性能增强分支。}
}
\]

若后续需要扩展理论，可考虑：

- 对BiGRU施加谱归一化；
- 控制网络Lipschitz常数；
- 推导输入序列扰动到输出表示扰动的上界；
- 将固定SGW表示保留为理论旁路。

---

# 25. 主要风险

## 25.1 小样本过拟合

当前训练集规模有限，因此BiGRU可能增加方差。

应使用：

- 小隐藏维度；
- 一层网络；
- Dropout；
- early stopping；
- 多随机种子；
- 不过度调参。

## 25.2 序列长度或站点泄漏

BiGRU可能学习：

- 扫描长度；
- 窗口数量；
- 站点特定预处理差异。

应检查：

- 窗口数与标签的关系；
- 窗口数与站点的关系；
- 各站点长度分布；
- 仅长度基线的分类能力。

## 25.3 Proxy与Exact来源不一致

当前D3-B中：

- variation来自Proxy；
- 分类头在Exact-SGW特征上训练。

新时序分支推荐优先使用：

\[
\boxed{
\text{硬关键图上逐窗口计算的Exact谱分位序列。}
}
\]

若使用Proxy序列，则必须为时序分支重新训练分类头，不能直接复用Exact分类头。

## 25.4 阈值重新选择

新增时序分支后，最终概率分布发生变化。

因此：

- 原D3-B阈值不能直接视为最终模型阈值；
- 必须在validation重新选择；
- test上不得重新调阈值；
- 阈值策略必须在实验前固定。

---

# 26. 最终推荐方案

推荐第一版采用：

\[
\boxed{
\text{冻结D3-B基础路径}
+
\text{16维逐步variation}
+
\text{一层BiGRU}
+
\text{Mean/Max池化}
+
\text{时序辅助头}
+
\text{残差logit融合}
}
\]

核心公式：

\[
v^{(m)}
=
|Q^{(m+1)}-Q^{(m)}|,
\]

\[
H^{(1:T)}
=
\operatorname{BiGRU}
\left(
\widetilde V
\right),
\]

\[
h_T
=
MLP
\left[
\operatorname{MeanMask}(H);
\operatorname{MaxMask}(H)
\right],
\]

\[
z_T
=
Classifier_T(h_T),
\]

\[
\boxed{
z_{\mathrm{final}}
=
z_B+\alpha z_T.
}
\]

该方案的优势是：

1. 保留当前已经取得较好结果的D3-B；
2. 只增加时间顺序建模；
3. 性能变化容易归因；
4. 参数量适中；
5. 支持变长窗口；
6. 不需要第一轮重新训练selector；
7. 理论固定表示仍可保留为旁路；
8. 适合作为当前项目的下一阶段性能增强实验。
