# SV-HardSGW 神经网络模块升级路线设计（修订版）

## V1 → V2 → V3 渐进式架构升级方案

更新日期：2026-07-29

---

## 1. 当前模型、证据与升级目标

当前实现：

```text
signed_gin_multibranch_late_fusion
```

已经完成：

- 学习型 Hard Key Subgraph Selector；
- 保留正负边的硬关键子图导出；
- 修复后的 Signed GIN 关键子图编码；
- Static-spectral 分支；
- Variation 分支；
- 三分支独立监督与非负 logit 后期融合。

当前证据表明：

1. GIN 低秩化已经明显缓解；
2. Static-spectral 仍是 WMRC 上最强且较稳定的独立分支；
3. 当前全局 softmax 后期融合仍存在跨折负迁移；
4. 三个融合权重长期接近均匀，弱分支无法真正收缩到零；
5. selector 已经学习节点和边是否应进入硬关键子图，但下游编码器尚未充分学习
   冻结硬图内部的细粒度节点与边模式。

因此，更准确的升级目标是：

```text
已学习的关键子图选择
        +
硬子图内部结构表示学习
        +
不伤害稳定主干的安全融合
        +
判别性边模式学习
```

当前版本不使用：

- 空间坐标；
- 站点标签；
- ROI 名称 embedding；
- 原始社区编号 embedding。

---

## 2. 总体升级原则

### 2.1 不撤销已经验证有效的修复

当前 `mean + std` 节点读出、残差、Jumping Knowledge、紧凑时间聚合和
train-only BatchNorm 已经在三个 WMRC 外折中稳定提高 GIN 有效秩。

因此：

- 不用纯 Attention pooling 替换 `mean + std`；
- 不恢复高维冗余 GIN 表示；
- 不恢复静态、Variation、GIN 的直接特征拼接；
- 不删除 signed normalized message passing。

### 2.2 Static-spectral 作为安全主干

当前 WMRC 冻结 OOF 结果：

| 路径 | Pooled OOF AUROC | Mean fold AUROC ± SD |
|---|---:|---:|
| 当前融合 | 0.5634 | 0.5613 ± 0.0572 |
| GIN | 0.5303 | 0.5309 ± 0.0622 |
| Static-spectral | **0.5827** | **0.5792 ± 0.0220** |
| Variation | 0.5290 | 0.5327 ± 0.0296 |

后续架构以 Static-spectral 为主干，GIN、Variation 和 Attention 只能以残差形式
增加信息，不再强制占有固定比例的融合权重。

### 2.3 每次只引入一个可识别改动

升级顺序为：

```text
当前模型基线
→ V1A 安全残差融合
→ V1B 残差 Attention
→ V2 Edge-aware Signed message passing
→ V3 Prototype Memory（条件性进入）
```

每一步必须在前一步冻结后比较。不得一次加入 Attention、Edge-aware 和 Prototype，
否则无法判断性能变化来源。

### 2.4 单次 validation 只用于筛选

- test/outer-test 不参与超参数、分支、checkpoint 规则或架构选择；
- 单次 train/validation 只用于实现检查和廉价筛选；
- 正式结论以完整 3-fold OOF 为准；
- 多个候选中只有一个预先冻结的胜出方案进入正式 OOF。

---

## 3. 当前实验状态

### 3.1 WMRC

当前架构已经完成：

- 单次冻结划分；
- 3-fold 完整交叉拟合；
- GIN 表示诊断；
- 冻结分支贡献诊断；
- 与改进前模型的 OOF 对比。

### 3.2 ADHD

当前架构的 ADHD 3-fold OOF 已经启动。

本路线不再安排“ADHD 最新单划分实验”。当前正在运行的 OOF 将直接作为 ADHD
正式基线，完成后冻结：

- pooled OOF AUROC；
- site-stratified OOF AUROC；
- mean fold AUROC ± SD；
- Accuracy、Balanced Accuracy、F1；
- 每折 selector、分支 AUROC、GIN 有效秩和融合权重。

在 ADHD 基线 OOF 完成之前可以进行本地代码开发和单元测试，但不得使用尚未完成的
outer-test 结果迭代选择 V1/V2 架构。

---

## 4. V1A：Static Anchor + Zero-output Residual Experts

### 4.1 目标

首先只解决当前最明确的融合负迁移：

> 当 GIN 或 Variation 没有可靠增益时，它们不应破坏 Static-spectral 主干。

V1A 暂不增加 Attention，确保融合机制的作用可单独识别。

### 4.2 两阶段训练

#### 阶段一：训练 Static-spectral 主干

训练：

```text
Static-spectral
→ 16维投影
→ static classifier
→ static logits
```

仅使用 inner-train 训练，使用 inner-validation 选择 checkpoint。得到：

\[
\ell_s.
\]

选定后冻结 static 主干的：

- train-only scaler；
- static projection；
- static classifier；
- validation 阈值规则。

#### 阶段二：训练残差专家

在冻结 static 主干后训练：

- 修复后的 GIN 专家；
- Variation 专家；
- 两个非负残差门控。

最终输出：

\[
\ell
=
\ell_s
+
g_G\,\Delta\ell_G
+
g_V\,\Delta\ell_V.
\]

其中：

\[
g_G,g_V\ge 0.
\]

两个残差专家的最后输出层采用零权重、零偏置初始化：

\[
\Delta\ell_G(0)=0,\qquad
\Delta\ell_V(0)=0.
\]

因此初始模型严格满足：

\[
\ell(0)=\ell_s.
\]

门控使用接近零的非负参数化，并加入向零收缩。推荐：

\[
g_k=\operatorname{sigmoid}(\beta_k),
\qquad \beta_k(0)=-6.
\]

训练 checkpoint 候选必须包含 epoch 0 的纯 static anchor。若残差专家不能在
inner-validation 上改善预设指标，允许最终 checkpoint 保持为 static anchor。

### 4.3 分支结构

```text
                         frozen static anchor
                         ┌───────────────────┐
Static-spectral ────────►│ static classifier │────► l_s
                         └───────────────────┘

Repaired Signed GIN ─► GIN expert ─► Δl_G ─► g_G ─┐
                                                   ├─► l
Variation ───────────► Var expert ─► Δl_V ─► g_V ─┘
```

三个分支不共享投影层或分类头。

### 4.4 V1A 损失

\[
\mathcal L_{\mathrm{V1A}}
=
\mathcal L_{\mathrm{main}}(\ell,y)
+
\lambda_{\mathrm{aux}}
\frac{
\mathcal L_G+\mathcal L_V
}{2}
+
\lambda_{\mathrm{gate}}(g_G+g_V).
\]

说明：

- `main` 使用最终残差输出；
- 辅助分类损失保证专家即使在门控接近零时仍能学习；
- gate penalty 使无可靠增益的专家保持关闭；
- static anchor 在阶段二冻结，不被专家梯度改变。

### 4.5 V1A 实验

固定：

- 同一数据划分；
- 同一 selector checkpoint；
- 同一硬图缓存；
- 同一 train-only scaler；
- 同一训练种子和 checkpoint 规则。

比较：

| 模型 | 融合方式 |
|---|---|
| 当前架构 | 三分支全局 softmax logit 融合 |
| V1A | Static anchor + GIN/Variation 残差专家 |

必须报告：

- final、static、GIN、Variation AUROC；
- residual gate 数值；
- fusion regret；
- GIN 有效秩与样本间余弦；
- mean fold AUROC ± SD。

---

## 5. V1B：Residual Node Attention

### 5.1 目标

selector 已决定哪些节点进入硬关键子图；V1B 只回答：

> 在已经选出的硬关键子图内部，学习型节点加权是否能补充 `mean + std` 读出？

Attention 权重不能直接解释为因果节点重要性。

### 5.2 保留当前主读出

当前可靠读出：

\[
z_{\mathrm{ms}}
=
[\operatorname{mean}(h_i),\operatorname{std}(h_i)].
\]

新增：

\[
a_i
=
\operatorname{softmax}(\operatorname{MLP}(h_i)),
\qquad
z_{\mathrm{att}}
=
\sum_i a_i h_i.
\]

不使用 `z_att` 替换 `z_ms`，而是：

\[
z_G
=
z_{\mathrm{ms}}
+
g_A P_A(z_{\mathrm{att}}).
\]

其中 Attention 残差投影零输出初始化，且：

\[
g_A(0)\approx0.
\]

初始状态仍严格等价于当前修复后的 `mean + std` GIN。

### 5.3 Attention 诊断

必须记录：

- 归一化 attention 熵；
- 最大节点权重；
- 有效节点数；
- attention 与绝对度的 Spearman 相关；
- Attention 屏蔽前后 validation AUROC；
- 预算匹配的节点扰动结果。

仅观察 attention 热图不能证明节点重要性。

### 5.4 V1B 实验

比较：

| 模型 | GIN读出 | 融合 |
|---|---|---|
| V1A | `mean + std` | 安全残差融合 |
| V1B | `mean + std + residual attention` | 安全残差融合 |

V1B 只有在以下条件同时满足时才保留：

1. GIN 分支 validation/OOF AUROC 有增益；
2. final 不低于 V1A；
3. GIN 有效秩不重新下降；
4. attention 不重新退化为近似均匀分配；
5. 增益不是仅由单个 fold 产生。

---

## 6. V1 正式成功标准

V1A 和 V1B 先通过本地测试与 16 样本记忆实验，再进行单次 validation 廉价筛选。
只有一个冻结方案进入正式 OOF。

### 表示闸门

- GIN representation 归一化有效秩不低于 `0.10`；
- GIN projection 平均样本间余弦低于 `0.995`；
- 所有输出有限；
- variable-length mask 行为不变；
- 正负边符号保持。

### 融合闸门

- final 相对 static-spectral 的 OOF regret 不超过 `0.01`；
- 无增益专家的残差门控应接近零；
- final pooled OOF AUROC 不低于当前架构；
- fold AUROC SD 不高于对应数据集当前基线；
- 不允许依赖 outer-test 选择 V1A 或 V1B。

### 跨数据集闸门

进入 V2 至少要求：

1. V1 在 ADHD 或 WMRC 的 pooled OOF AUROC 有预先定义的实际增益；
2. 在另一个数据集上相对当前基线不发生超过 `0.01 AUROC` 的退化；
3. 融合不再系统性低于 static-spectral。

---

## 7. V2：Edge-aware Signed Message Passing

### 7.1 目标

V1 主要改进安全融合和节点级读出。V2 进一步学习硬关键子图中的正负连接模式。

现有缓存已经保存每个硬图窗口的带符号邻接矩阵，因此第一版 V2 可由现有缓存构造：

\[
[A_{ij},|A_{ij}|].
\]

不需要重新训练 selector，也不需要把负边当作无边。

### 7.2 正负消息通道

定义：

\[
A_{ij}^{+}=\max(A_{ij},0),
\qquad
A_{ij}^{-}=|\min(A_{ij},0)|.
\]

正边消息：

\[
m_i^{+}
=
\sum_j
A_{ij}^{+}
\phi_{+}
(
[h_i,h_j,|A_{ij}|]
).
\]

负边消息：

\[
m_i^{-}
=
\sum_j
A_{ij}^{-}
\phi_{-}
(
[h_i,h_j,|A_{ij}|]
).
\]

节点更新：

\[
h_i'
=
\psi
(
[h_i,m_i^{+},m_i^{-}]
).
\]

要求：

- \(\phi_+\) 与 \(\phi_-\) 参数独立；
- 共享函数与邻居求和保证节点排列等变性；
- 正负消息分别记录，不能先平均后抵消；
- 不允许把 \(A\) 全部替换成 \(|A|\)；
- 不创建原图或硬图中不存在的新边。

### 7.3 计算实现

硬图通常仍较稠密，V2 必须：

- 只在 `edge_mask` 有效边上构造 edge index；
- 使用向量化 gather/scatter；
- 避免 Python 节点双重循环；
- 保留 list-based variable-length batching；
- 在本地 dummy graph 验证 forward/backward；
- 在服务器比较显存、每 epoch 时间和吞吐。

### 7.4 V2 实验

#### V2-1：Edge-aware 增益

| 模型 | 消息 |
|---|---|
| V1 winner | signed normalized adjacency |
| V2 | edge-aware positive/negative channels |

#### V2-2：Signed 消融

严格匹配：

- selector；
- 硬图；
-参数量；
- 初始化；
-训练 seed；
- scaler；
- checkpoint 规则。

比较：

```text
Signed：A+ 与 A− 独立通道
Unsigned：只使用 |A|
```

#### V2-3：边扰动

预先冻结：

- 扰动比例；
- 扰动类型；
- 随机重复次数；
- 正负边比例保持规则；
- targeted 与 random 的预算匹配规则。

至少区分：

- 边权随机化；
- 边符号随机化；
- 拓扑重连。

边扰动性能下降只能证明模型使用了对应连接信息，不能单独证明因果性。

### 7.5 V2 成功标准

1. Signed V2 pooled OOF AUROC 高于 V1；
2. Signed 不低于参数匹配的 unsigned 对照；
3. 预算匹配边扰动导致可重复的性能下降；
4. final 不低于 static anchor；
5. 增益不是仅由一个 fold 或一个站点产生；
6. 在一个数据集改善时，另一个数据集不出现明显退化。

V2 通过后才能讨论 V3。

---

## 8. V3：受约束的 Prototype Memory

### 8.1 当前优先级

V3 暂不直接实施。

此前原型实验出现：

- 16 个原型实际只使用约 3–4 个；
- 单个原型占据约 58%–60% 权重；
- 原型融合后表示方差下降；
- 下游分类表示坍缩。

因此，简单加入：

\[
d_k=\|z-P_k\|
\]

不能自动证明学习到了跨样本关键子图模式。

### 8.2 进入条件

只有当 V2 在完整 OOF 中稳定通过后，V3 才进入程序实现。

### 8.3 必要约束

未来 V3 至少需要：

- 类别条件原型；
- 原型多样性或最小距离约束；
- 原型使用率下限；
- 防止单原型垄断的熵约束；
- 原型残差通道零输出初始化；
- prototype-only 和 no-prototype 对照；
- 原型空间有效秩与占用率诊断。

第一轮不搜索 `K=2/4/8/16`。应预注册一个小规模 \(K\)，避免 validation 多重搜索。

### 8.4 解释边界

潜在向量原型不是实际图原型。只有当原型能够映射回稳定的节点、边或结构统计模式时，
才能解释为“共享疾病相关关键结构”。

---

## 9. 暂不加入的模块

### 9.1 Temporal Encoder

暂缓。

此前 BiGRU/学习型时间编码实验出现 validation 增益但 test 或跨折未稳定复现。
当前主要瓶颈仍是：

- 安全融合；
- 子图内部结构编码；
- 边模式利用。

正确顺序：

```text
安全融合
→ 空间结构表示
→ 边模式学习
→ 再评估时间演化编码
```

### 9.2 Transformer

当前不推荐：

- 样本量有限；
- 参数量和实验搜索成本较高；
- 当前基础融合问题尚未解决；
- 容易增加划分敏感性和过拟合。

---

## 10. 修订后的实验顺序

### Step 0：完成当前 ADHD OOF

当前已经启动，继续运行至完整结束：

- 不追加 ADHD 最新单划分；
- 不因中途 fold 结果修改架构；
- 完成后冻结汇总和逐样本 OOF 预测；
- 与 WMRC 当前基线共同构成升级验收基线。

### Step 1：本地实现 V1A

- static anchor 两阶段训练；
- GIN/Variation 零输出残差专家；
- 非负收缩门控；
- epoch 0 anchor checkpoint；
- forward/loss/backward/save/load；
- variable-length、signed edge、padding/mask 和排列不变测试。

### Step 2：V1A 最小服务器验收

- CUDA smoke；
- 16 样本记忆；
- 一个冻结 train/validation 的廉价筛选；
- 不运行 test。

### Step 3：本地实现 V1B

- 保留 `mean + std`；
- 加入零输出 residual attention；
- attention 分布和屏蔽诊断；
- 与 V1A 仅相差 Attention 残差。

### Step 4：冻结一个 V1 winner

只使用 train/inner-validation：

- 比较 V1A 与 V1B；
- 冻结唯一胜出方案；
- 不根据 outer-test 选择方案。

### Step 5：V1 完整 OOF

优先复用当前 ADHD/WMRC OOF 中已经由 inner-train 训练、且未见 outer-test 的：

- fold assignments；
- fold-local selector；
- hard graph cache；
- train-only scaler。

复用这些冻结产物可以形成严格配对比较，并显著减少计算量。不得在观察 V1 outer-test
后继续修改同一确认性架构。

### Step 6：实现并筛选 V2

- 只替换 GIN 消息模块；
- static anchor 与安全残差融合保持不变；
- 先完成 signed/unsigned 和边扰动廉价诊断；
- 只让一个冻结 V2 方案进入 OOF。

### Step 7：条件性讨论 V3

只有 V2 的跨数据集 OOF 闸门通过后才进入。

---

## 11. 计算量控制

为了避免实验规模失控：

1. 不为每个候选都运行完整 OOF；
2. 本地只做 dummy、单元测试和最小流程；
3. 单次 validation 只筛选，不形成正式结论；
4. 每个版本只保留一个候选进入 OOF；
5. 当前 ADHD OOF 的 fold-local selector、缓存和 scaler 可用于配对下游架构比较；
6. V3 不进行四档原型数量网格搜索；
7. Temporal 和 Transformer 不与 V1/V2 并行开展。

---

## 12. 最终目标架构

只有各阶段逐步通过后，最终候选才可能是：

```text
Full Signed Graph Sequence
        │
        ▼
Learned Hard Key Subgraph Selector
        │
        ▼
Signed Key Subgraph Sequence
        │
        ├───────────────► Static-spectral anchor ──────► l_s
        │
        ├───────────────► Edge-aware Signed GIN
        │                    │
        │                    ├─ mean + std
        │                    └─ residual attention
        │                           │
        │                           ▼
        │                       Δl_G, g_G
        │
        └───────────────► Variation expert ────────────► Δl_V, g_V

final logits = l_s + g_G Δl_G + g_V Δl_V

Prototype Memory：
仅在 V2 OOF 稳定通过后，作为受约束残差模块条件性加入
```

该路线的核心不是不断增加模块，而是：

> 以稳定的 Static-spectral 为安全下限，只允许经过验证的关键子图神经信息以残差方式
> 提高分类，不能在没有增益时破坏主干。

---

## 13. 最终研究逻辑

```text
学习关键子图
→ 修复关键子图神经表示
→ 建立不会负迁移的安全融合
→ 学习正负边判别模式
→ 条件性探索跨样本结构原型
```

这一路线保留了原设计“关键子图表示学习框架”的目标，同时让每一步都具备：

- 可实现性；
- 可归因性；
- 可复现性；
- OOF 验证边界；
- 失败时的安全回退路径。
