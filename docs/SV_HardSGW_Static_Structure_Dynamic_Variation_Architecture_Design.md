# SV-HardSGW：静态结构＋动态 Variation 联合架构设计

## 1. 文档状态

- 版本：V1.0
- 日期：2026-07-28
- 状态：待实现、待验证的候选正式架构
- 任务：可变长度带符号动态脑图序列二分类
- 基线：D3-B Variation-Only Exact-Head

本文档定义在现有 D3-B 关键子图选择流程上新增“静态结构通道”的最小
架构。它不覆盖或删除已有 D3-B、all-34 Proxy、Exact-SGW 和时间编码器
实现。

本架构的核心假设是：

> 稳定的类别信息可能同时存在于关键子图的平均结构状态和跨窗口演化幅度
> 中。仅使用 Variation 会丢失静态结构基线；仅使用静态结构则无法表示
> 动态演化。

当前文档只给出预注册设计，不声明该架构已经提高分类性能。

---

## 2. 设计动机

已有诊断得到以下事实：

1. D3-B 在原 ADHD 划分上的 Test AUROC 为 0.611798，但在替代划分上
   明显下降；
2. 16 维 Variation 在不同划分和部分站点之间出现类别效应方向反转；
3. 恢复旧的 all-34 表示没有稳定改善，因为旧表示的 34 维均为动态量，
   并不包含真正的静态结构状态；
4. Proxy 与 Exact 的 Variation 高度一致，说明主要瓶颈不是
   Proxy–Exact 数值转换；
5. 因此下一步不应简单恢复 spectral speed 或 GW speed，而应补充与时间
   差分互补的静态带符号结构信息。

---

## 3. 总体流程

```text
可变长度带符号动态图序列
        │
        ▼
15维节点特征 + 6维边特征
        │
        ▼
节点/边残差评分器
        │
        ▼
社区覆盖节点Top-k + 候选边Top-k + STE
        │
        ▼
硬关键子图序列
        ├─────────────────────────────┐
        ▼                             ▼
静态结构通道 S（28维）          动态Variation通道 V（16维）
        │                             │
train-only scaler               train-only scaler
        │                             │
28→16投影                       16→16投影
        └──────────────┬──────────────┘
                       ▼
                  拼接为32维
                       │
                       ▼
               32→16→2分类头
                       │
                       ▼
                  正类概率
                       │
                       ▼
           validation确定并冻结阈值
```

原 D3-B Variation-only 路径继续保留，作为 SV0 对照和工程回退入口。

---

## 4. 输入与不可变约束

单个样本表示为：

\[
\mathcal G_b
=
\{G_b^{(1)},\ldots,G_b^{(M_b)}\},
\qquad
G_b^{(m)}
=
(A_b^{(m)},C_b^{(m)},I_b^{(m)}).
\]

- \(A_b^{(m)}\)：带符号加权邻接矩阵；
- \(C_b^{(m)}\)：当前窗口的社区编号；
- \(I_b^{(m)}\)：存在时用于相邻窗口节点对齐的稳定节点标识；
- \(M_b\)：样本相关的时间窗口数；
- \(N_b^{(m)}\)：窗口相关的节点数。

必须满足：

- 支持不同样本具有不同 \(M_b\) 和 \(N_b^{(m)}\)；
- 使用 list-based batching，不截断原图；
- padding（若存在）必须由 mask 排除；
- 正边与负边均为有效边；
- 不使用空间坐标；
- 不使用 ROI 名称 embedding；
- 不使用原始 community ID embedding；
- 不把站点或数据子集标识输入模型；
- 标签只来自冻结的数据协议；
- validation 和 test 不参与 scaler 拟合。

边存在条件统一为：

\[
\operatorname{edge\_mask}_{ij}
=
\mathbf 1(|A_{ij}|>\theta_{\mathrm{edge}}).
\]

所有阶段必须读取同一冻结协议中的
\(\theta_{\mathrm{edge}}\)。

---

## 5. 学习型硬关键子图选择器

选择器沿用当前 D3 实现，不改变总体流程：

```text
图序列
→ 节点/边特征
→ 节点重要性
→ 边重要性
→ 社区覆盖硬选择
→ STE反向传播
→ 硬关键子图序列
```

### 5.1 节点与边特征

节点仍使用现有 15 维无坐标特征，包括：

- 绝对、正、负连接强度；
- 连接强度时间变化及有效标记；
- 邻接变化幅度与有效比例；
- 社区规模及社区内外正负结构；
- 局部聚类系数。

边仍使用 6 维特征：

\[
e_{ij}
=
\left[
A_{ij},
|A_{ij}|,
\Delta A_{ij},
|\Delta A_{ij}|,
m_{ij}^{\Delta},
\mathbf 1(c_i=c_j)
\right].
\]

硬图中的边必须保留原始符号。

### 5.2 硬选择

- 节点目标比例默认保持 0.50；
- 边目标比例默认保持 0.30；
- 每个非空社区至少保留一个最高分节点；
- 只从入选节点诱导出的原始有效边中选边；
- 前向使用 0/1 hard mask；
- 反向使用直通估计器；
- 第一轮实验不加入新的选择间隔损失。

第一轮保持选择预算和结构正则不变，避免同时改变选择规模、特征表示和
分类头。

---

## 6. 静态结构通道

静态通道描述关键子图序列的平均结构状态，不计算相邻窗口差分。对每个
有效硬关键子图窗口 \(H_b^{(m)}\) 先构造窗口级特征，再只对有效窗口
取平均。

### 6.1 16维静态谱状态

对每个窗口构造正则化归一化带符号拉普拉斯：

\[
D_{ii}=\sum_j |A_{ij}|,
\]

\[
L_{\mathrm{signed}}
=
(D+\eta I)^{-1/2}
(D-A+\eta I)
(D+\eta I)^{-1/2}.
\]

在固定的 16 个分位点提取特征值分位数：

\[
q_{m,k}=Q_{\alpha_k}
\left(
\operatorname{eig}(L_{\mathrm{signed}}^{(m)})
\right),
\qquad
k=1,\ldots,16.
\]

样本级静态谱状态为：

\[
\bar q_{b,k}
=
\frac{1}{|\mathcal V_b|}
\sum_{m\in\mathcal V_b}q_{m,k},
\]

其中 \(\mathcal V_b\) 是有效硬子图窗口集合。归一化拉普拉斯和谱分位数
使该表示能够适配不同节点数。

### 6.2 12维带符号与社区结构

对每个窗口只在无向上三角有效节点对上计算。记：

\[
P_m=\binom{N_m}{2},
\]

\[
W_m^+
=
\sum_{i<j}\max(A_{ij}^{(m)},0),
\qquad
W_m^-
=
\sum_{i<j}|\min(A_{ij}^{(m)},0)|.
\]

窗口级 12 维结构特征定义如下：

| 维度 | 特征 | 定义 |
|---:|---|---|
| 1 | 正边密度 | \(|E_m^+|/(P_m+\epsilon)\) |
| 2 | 负边密度 | \(|E_m^-|/(P_m+\epsilon)\) |
| 3 | 正连接强度 | \(W_m^+/(P_m+\epsilon)\) |
| 4 | 负连接幅值 | \(W_m^-/(P_m+\epsilon)\) |
| 5 | 社区内正强度 | 社区内正权和/社区内可连接节点对数 |
| 6 | 社区内负幅值 | 社区内负幅值和/社区内可连接节点对数 |
| 7 | 社区间正强度 | 社区间正权和/社区间可连接节点对数 |
| 8 | 社区间负幅值 | 社区间负幅值和/社区间可连接节点对数 |
| 9 | 社区内绝对强度占比 | 社区内绝对权和/全部绝对权和 |
| 10 | 正强度占比 | \(W_m^+/(W_m^++W_m^-+\epsilon)\) |
| 11 | 归一化社区数量 | \(K_m/(N_m+\epsilon)\) |
| 12 | 社区规模熵 | \(-\sum_c\pi_c\log\pi_c/\log K_m\) |

其中：

- \(K_m\) 为硬子图中的非空社区数；
- \(\pi_c=|c|/N_m\)；
- 当 \(K_m\le 1\) 时，归一化社区规模熵定义为 0；
- 分母为 0 时相应特征安全置 0；
- community ID 只用于同社区判断和分组统计；
- 对 community ID 做任意一一重编号，输出必须不变。

所有强度均分别保留正连接和负连接幅值，禁止用带符号简单平均造成正负
抵消。

样本级 12 维结构向量为所有有效窗口的逐维平均。不得把原始节点数、
窗口数或社区编号直接加入特征。

### 6.3 静态通道输出

\[
S_b
=
\left[
\bar q_b^{16},
\bar r_b^{12}
\right]
\in\mathbb R^{28}.
\]

---

## 7. 动态 Variation 通道

动态通道沿用当前已验证的 16 维谱变化幅度。对于相邻且均有效的窗口：

\[
\Delta q_{m,k}=q_{m+1,k}-q_{m,k}.
\]

样本级 Variation 为：

\[
V_{b,k}
=
\frac{1}{|\mathcal T_b|}
\sum_{m\in\mathcal T_b}
|\Delta q_{m,k}|,
\qquad
k=1,\ldots,16.
\]

\(\mathcal T_b\) 是有效相邻窗口集合。无效 transition 必须由 mask 排除。
若不存在有效 transition，则 \(V_b\) 安全置 0，并在产物中记录有效
transition 数。

第一轮分类表示明确排除：

- 16 维有方向 spectral delta；
- spectral speed；
- GW speed。

这些特征仍可保留在缓存和理论诊断中，但不进入 SV2 分类头。GW 结构
信息继续通过选择器的结构保持正则发挥作用。

---

## 8. Train-only 双通道标准化

静态通道和动态通道分别拟合 scaler：

\[
z_b^S=(S_b-\mu_S)/\sigma_S,
\qquad
z_b^V=(V_b-\mu_V)/\sigma_V.
\]

\(\mu_S,\sigma_S,\mu_V,\sigma_V\) 只能由训练集计算。需要：

- 对零方差或近零方差维使用固定下限；
- 在 scaler 中记录训练 manifest、selector checkpoint、协议和特征
  schema 的 SHA256；
- validation/test 只能加载冻结 scaler；
- 两个通道不得共用一个全局标量均值和标准差。

---

## 9. 双通道分类器

### 9.1 等宽通道投影

为避免 28 维静态通道仅因输入维度更高而占据优势，两个通道都投影到
16 维：

```text
Static:
Linear(28,16) → GELU → LayerNorm(16)

Variation:
Linear(16,16) → GELU → LayerNorm(16)
```

### 9.2 固定拼接融合

\[
h_b=[h_b^S,h_b^V]\in\mathbb R^{32}.
\]

分类头为：

```text
Linear(32,16)
→ GELU
→ Dropout(0.10)
→ Linear(16,2)
```

第一轮不使用 attention、门控、prototype 或额外残差融合。这样可以直接
判断静态通道是否提供独立增益，并降低小样本过拟合风险。

---

## 10. 训练阶段

### 10.1 阶段一：联合训练选择器与 Proxy 分类头

硬选择器输出可微 STE 硬图。静态谱状态、可微连续强度统计和 Variation
共同形成 Proxy 表示，训练选择器和 SV 分类头。

总损失保持现有 D3 权重作为第一轮默认值：

\[
\begin{aligned}
L=\;&
0.50L_{\mathrm{cls}}
+0.05L_{\mathrm{node\ budget}}\\
&+0.05L_{\mathrm{edge\ budget}}
+0.05L_{\mathrm{Laplacian}}
+0.02L_{\mathrm{GW\ fidelity}}.
\end{aligned}
\]

本轮不得根据 validation 结果同时调整上述正则权重。

### 10.2 类别加权修复

禁止在每个微批次内部使用：

\[
\frac{\sum_i w_{y_i}CE_i}{\sum_iw_{y_i}},
\]

因为物理 batch size 为 1 时类别权重会完全抵消。

预先根据训练集计算类别权重 \(w_c\)，并归一化为训练分布下期望为 1：

\[
\widetilde w_c
=
\frac{w_c}{\sum_{c'}\pi_{c'}w_{c'}}.
\]

分类风险使用：

\[
L_{\mathrm{cls}}
=
\frac{1}{B_{\mathrm{eff}}}
\sum_i
\widetilde w_{y_i}CE_i.
\]

不得再除以当前微批次的权重和。该定义使物理 batch size 为 1 时类别
权重仍然有效。

### 10.3 有效 batch

默认：

```text
物理 batch size = 1
梯度累积步数 = 8
有效 batch size = 8
```

有效 batch 8 是降低单样本、单类别和单站点梯度噪声的工程默认值，不是
数学硬约束。资源受限时允许使用累积步数 4，但必须记录在实验产物中。

梯度累积不要求同时把 8 个图放入显存。每个 epoch 仍只对每个样本执行
一次主要 forward/backward，显存接近 batch 1，总图计算量不会增加 8 倍。

在一次累积周期内应尽量混合类别和站点。数据加载顺序必须由固定 seed
生成并可复现。

### 10.4 阶段二：冻结选择器并导出特征

选择器冻结后，对 train、validation、test 导出：

- 硬关键子图序列；
- 28 维静态特征；
- 16 维 Variation；
- 有效窗口和 transition mask；
- 协议、checkpoint、选择 seed 和特征 schema 哈希。

### 10.5 阶段三：训练最终 SV 分类器

- 只用 train 拟合两个 scaler；
- 只用 train 更新分类器参数；
- 用 validation 选择 checkpoint；
- 用 validation 确定分类阈值；
- test 只运行一次冻结评估；
- 不允许在 test 上调阈值、选择特征或选择模型。

---

## 11. 站点处理与模型选择

站点不进入模型特征。站点只用于：

1. 构造类别与站点分布更可比的冻结划分；
2. 训练采样/梯度累积周期的多样性控制；
3. 计算分站点和站点内分层指标；
4. 检查模型是否只利用跨站点基线差异。

validation 同时报告：

- pooled AUROC；
- pooled balanced accuracy；
- eligible-site stratified AUROC；
- 各站点 AUROC 和类别数量。

eligible site 必须同时包含两个类别。小样本站点的单独 AUROC 只作描述，
不得主导模型选择。

默认 checkpoint 分数为：

\[
\operatorname{score}_{\mathrm{val}}
=
\frac{1}{2}
\left(
\mathrm{AUC}_{\mathrm{pooled}}
+
\mathrm{AUC}_{\mathrm{site\ stratified}}
\right).
\]

若 stratified AUROC 因类别缺失不可计算，则该次实验应报告协议问题，不得
静默退回 pooled AUROC。

---

## 12. 最小消融实验

在同一冻结划分、相同选择预算、相同训练 seed 和相同训练日程下只比较：

| 编号 | 输入 | 目的 |
|---|---|---|
| SV0 | 16维 Variation | 复现动态基线 |
| SV1 | 28维静态结构 | 判断静态结构是否独立可判别 |
| SV2 | 静态结构＋Variation | 判断二者是否互补 |

执行顺序：

1. seed 42 完成 SV0、SV1、SV2；
2. 仅当 SV2 在 validation 同时改善 pooled 和 site-stratified AUROC，
   再补 seed 43；
3. 不进行大规模超参数搜索；
4. 不把 test 用于决定是否保留静态特征。

第一轮暂不增加其他模型分支。

---

## 13. 验收标准

### 13.1 数据与数值正确性

- 所有输出有限；
- 正负边均被保留；
- 节点和窗口置换不改变样本级输出；
- community ID 一一重编号不改变结构特征；
- 重复同一静态窗口不会改变静态均值；
- 常量序列的 Variation 为 0；
- padding 和无效 transition 不参与任何平均；
- 不同 \(M\)、\(N\) 的样本能在同一 list batch 中运行；
- scaler 只使用 train；
- test 阈值来自 validation。

### 13.2 训练正确性

- batch 1 下两个类别的权重不会被归一化抵消；
- 梯度累积与等价有效 batch 的分类梯度方向一致；
- 节点和边评分器均获得有限的非零梯度；
- 评分分布不退化为全 0、全 1 或完全常数；
- 硬图既不是空图，也不是完整图；
- Laplacian 和 GW fidelity 均被记录。

### 13.3 第一轮性能门槛

性能门槛用于决定是否继续验证，不是对理论成立的证明：

- SV2 validation pooled AUROC 不低于 0.60；
- SV2 validation site-stratified AUROC 不低于 0.58；
- SV2 相对 SV0 的两个 validation AUROC 均改善至少 0.03；
- train–validation AUROC 差距不超过 0.10；
- validation 概率具有非零方差且分类头无表示坍缩。

只有通过 validation 预注册门槛后，才读取一次 test。test 需要同时报告
pooled 与 site-stratified 指标以及 bootstrap 置信区间，不以单一
Accuracy 判定成功。

---

## 14. 必要单元测试

至少新增：

1. 16 维静态谱状态维数与有限性测试；
2. 12 维带符号结构公式测试；
3. 正负强度不发生抵消测试；
4. 社区编号重标记不变性测试；
5. 节点一致置换不变性测试；
6. variable-\(M\)/variable-\(N\) 与 mask 测试；
7. 常量序列 Variation 为 0 测试；
8. train-only 双 scaler 来源校验测试；
9. batch 1 类别权重不抵消测试；
10. 梯度累积与等价 batch 测试；
11. SV0/SV1/SV2 输入维度与分支隔离测试；
12. validation 阈值冻结到 test 测试。

---

## 15. 与现有 D3-B 的关系

| 项目 | D3-B | SV-HardSGW |
|---|---|---|
| 硬选择器 | 保留 | 保留 |
| 带符号图处理 | 保留 | 保留 |
| Variable-length | 保留 | 保留 |
| Variation | 16维 | 16维 |
| 真正静态结构 | 无 | 28维 |
| spectral/GW speed分类输入 | B路径屏蔽 | 第一轮不使用 |
| Laplacian/GW结构保持 | 保留 | 保留 |
| 分类融合 | 冻结旧Exact头 | 新的等宽双通道小分类器 |
| 原路径代码 | 正式基线 | 不删除、不覆盖 |

SV-HardSGW 不是对 D3-B 结果的重新命名，而是针对 Variation 跨划分不稳定
问题提出的新候选架构。

---

## 16. 理论对应

该表示把关键子图序列的信息分为：

\[
\Phi(\mathcal H_b)
=
\left[
\underbrace{\bar S(\mathcal H_b)}_{\text{平均结构状态}},
\underbrace{\bar V(\mathcal H_b)}_{\text{演化幅度}}
\right].
\]

- 静态通道描述关键子图在观测区间内通常呈现的带符号拓扑和谱状态；
- Variation 描述该状态在相邻窗口之间的变化幅度；
- Laplacian fidelity 约束关键图保持原图谱结构；
- GW fidelity 约束关键图保持原图扩散几何；
- 分类损失要求被保留结构与标签相关；
- Budget 防止选择器退化为完整图或空图。

因此，该架构仍然服务于“从动态原图中提取紧凑且具有判别性的关键结构”
这一目标，同时避免把信息传递仅等同于时间变化速度。

---

## 17. 一句话定义

> SV-HardSGW 使用无坐标、带符号、社区覆盖约束的学习型硬选择器提取
> 可变长度关键子图序列，以 28 维平均静态谱/社区结构和 16 维动态谱
> Variation 构成双通道表示，经各自的 train-only 标准化与等宽小型分类
> 头完成二分类，并使用 validation 冻结 checkpoint 和分类阈值。
