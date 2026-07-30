# Selector 完整图–软图–硬图类别间隔传递改进方案

## 0. 文档状态

- 文档性质：架构改进与实验设计
- 当前状态：待实现、待验证
- 理论依据：`关键子图谱_GW演化理论_核心推导版_附录ABC完整推导版.docx`
- 适用对象：Dual-STSE-HardSGW / D3 系列学习型关键子图 selector
- 目标：在固定节点和边预算下，使学习型 selector 比同预算随机选择更稳定地保留具有判别力的谱–GW演化结构

本文档只规定 selector 的改进，不改变后续“冻结 selector、导出硬关键子图、计算 Exact-SGW、训练或复用分类头”的总体流程。

---

## 1. 改进动机

### 1.1 理论给出的传递链

理论中的类别间隔并不是直接从完整图跳到硬图，而是经过：

```text
完整图 G
→ 同节点软关键图 Ḡ
→ 冻结后硬关键图 S
→ 硬关键图谱–GW演化表示
→ 类别间隔保持
```

训练阶段的带符号软邻接矩阵为：

\[
\bar A_{ij}
=
A_{ij}p_i p_j p_{ij}.
\]

因此：

- 正边在软图中仍为正；
- 负边在软图中仍为负；
- 不允许使用 \(|A_{ij}|\) 代替 \(A_{ij}\) 构造软图；
- 不改变完整图的节点空间，只连续衰减节点和边的贡献。

软图到硬图会进一步引入谱量化误差和 GW 量化误差：

\[
q_{\lambda,c},
\qquad
q_{\mathrm{GW},c}.
\]

类别 \(c\) 的最终误差半径为：

\[
\eta_c
=
L_{\mathcal A}
\left(
4\bar\varepsilon_{\lambda,c}
+
2\bar\varepsilon_{\mathrm{GW},c}
\right),
\]

其中：

\[
\bar\varepsilon_{\lambda,c}
=
\frac{B_n}{\pi_c\lambda_L}
+
q_{\lambda,c},
\]

\[
\bar\varepsilon_{\mathrm{GW},c}
=
\frac{B_n}{\pi_c\lambda_{\mathrm{GW}}}
+
q_{\mathrm{GW},c}
+
\varepsilon_{\mathrm{solver},c}.
\]

最终类别间隔满足：

\[
W_1(P_a^S,P_b^S)
\ge
\Delta_{ab}-\eta_a-\eta_b.
\]

因此，selector 的理论目标应当是：

> 在满足稀疏预算的同时，分别控制完整图到软图的结构逼近误差和软图到硬图的量化误差，使两类总误差半径尽可能小，并使硬图仍具有判别能力。

### 1.2 当前诊断提供的工程动机

近期同预算上游信息保留诊断得到：

| 数据与划分 | Full | Learned | Random | Learned − Random |
|---|---:|---:|---:|---:|
| WMRC 单划分 | 0.6742 | 0.4754 | 0.5076 | -0.0322 |
| ADHD 现成 fold 0 | 0.4597 | 0.5327 | 0.5650 | -0.0323 |

以上数值是固定表示上的廉价线性探针 AUROC，不是最终模型结论，但揭示了一个跨数据集重复出现的问题：

> 当前学习型 selector 尚未表现出比同预算随机压缩更强的信息保留能力。

两个数据集上 Full 与压缩图的相对关系并不相同：

- WMRC 中 Full 明显优于 Learned，说明压缩过程损失了较多可用信息；
- ADHD fold 0 中压缩优于 Full，说明压缩可能具有去噪作用；
- 但两者都出现 Learned 低于 Random，说明当前学习目标尚未稳定地把分类监督转化为优于随机的结构选择。

因此，改进不能简单等同于“尽量重建完整图”，而应是：

\[
\text{判别性约束}
+
\text{完整图到软图保真}
+
\text{软图到硬图量化控制}
+
\text{固定预算}.
\]

---

## 2. 当前 selector 与理论之间的差距

### 2.1 当前实现

当前 D3 selector：

1. 使用节点和边评分器得到 \(p_i\) 与 \(p_{ij}\)；
2. 按社区覆盖、节点预算和边预算执行离散 Top-k；
3. 前向使用 0/1 硬 mask；
4. 反向使用 straight-through estimator：

\[
\widetilde m
=
p+(m_{\mathrm{hard}}-p)_{\mathrm{detach}};
\]

5. 在 STE 硬邻接矩阵上计算 Proxy 分类、Laplacian 和 GW-aligned fidelity。

当前损失为：

\[
\begin{aligned}
L_{\mathrm{current}}
=\;&
0.50L_{\mathrm{proxy\_CE}}
+0.05L_{\mathrm{node\_budget}}\\
&+0.05L_{\mathrm{edge\_budget}}
+0.05L_{\mathrm{Laplacian}}\\
&+0.02L_{\mathrm{GW\ proxy}}.
\end{aligned}
\]

### 2.2 主要差距

当前实现把理论中的软、硬两个阶段压缩到了一次 STE 前向中：

- 没有显式产生并监督 \(\bar A_{ij}=A_{ij}p_ip_jp_{ij}\)；
- Laplacian 和 GW proxy 主要比较完整图与 STE 硬图；
- 没有单独测量完整图到软图的误差；
- 没有单独测量软图到硬图的 \(q_\lambda\) 与 \(q_{\mathrm{GW}}\)；
- 无法定位信息是在完整图→软图还是软图→硬图阶段丢失；
- checkpoint 主要由分类 AUROC 决定，没有对理论传递误差设置约束。

因此，当前实现与理论相关，但并未完整实现理论中的两阶段传递路径。

---

## 3. 改进后的总体架构

```text
完整带符号图序列
        │
        ▼
节点/边特征构造
        │
        ▼
节点评分 p_i、边评分 p_ij
        │
        ├─────────────────────────────┐
        ▼                             ▼
同节点带符号软图 Ḡ              社区覆盖 + 固定预算 Top-k
Ā_ij=A_ij p_i p_j p_ij               │
        │                             ▼
        │                         带符号硬图 S
        │                             │
        ├──── 完整图→软图保真 ────────┤
        │                             │
        └──── 软图→硬图量化控制 ──────┘
                                      │
                                      ▼
                           冻结、导出、Exact-SGW
```

需要同时保留三个图对象：

1. 完整图 \(G\)：原始带符号邻接矩阵；
2. 软图 \(\bar G\)：与完整图具有相同节点空间；
3. 硬图 \(S\)：固定预算 Top-k 后的真实可导出关键图。

---

## 4. 不变约束

本次改进不得破坏以下既有要求：

### 4.1 带符号图

\[
\operatorname{edge\_mask}_{ij}
=
\mathbf 1(|A_{ij}|>\tau_{\mathrm{edge}}).
\]

- \(\tau_{\mathrm{edge}}\) 必须来自冻结协议；
- 负边是有效边；
- 完整图、软图和硬图均必须保留边符号；
- soft adjacency 必须使用 \(A_{ij}\)，不得使用 \(|A_{ij}|\)。

### 4.2 可变长度

- 不假设不同样本具有相同时间窗口数 \(M\)；
- 不假设不同窗口具有相同节点数 \(N\)；
- 不截断原始图；
- 所有聚合、损失和统计必须忽略无效窗口或 padding。

### 4.3 社区标签

社区编号只用于：

- 同社区判断；
- 社区覆盖；
- 每社区最高分种子；
- 社区结构统计。

禁止使用：

```python
nn.Embedding(community_id)
```

### 4.4 预算和导出

第一轮比较保持现有预算不变：

\[
r_n=0.50,\qquad r_e=0.30.
\]

硬图仍由社区覆盖节点候选和候选边 Top-k 产生，不改变导出格式和 Exact-SGW 下游接口。

---

## 5. 完整图到软图

### 5.1 软图构造

对有效边：

\[
\bar A_{ij}
=
A_{ij}p_i p_j p_{ij}.
\]

对无效边：

\[
\bar A_{ij}=0.
\]

实现时应保证：

- 邻接矩阵对称；
- 对角线策略与完整图一致；
- \(A_{ij}<0\Rightarrow\bar A_{ij}<0\)；
- 梯度能够到达节点评分器和边评分器；
- 软图仍保留所有原节点，低概率节点只被连续衰减。

### 5.2 软图分类监督

在软图的可微谱–GW Proxy 表示上使用类别加权交叉熵：

\[
L_{\mathrm{cls}}^{\mathrm{soft}}
=
\operatorname{CE}
\left(
f_{\mathrm{proxy}}(H_{\mathrm{proxy}}(\bar{\mathcal G})),
Y
\right).
\]

该损失确保软图不是单纯重建完整图，而是保留与分类有关的结构。

### 5.3 拉普拉斯保真

对每个有效窗口计算：

\[
L_{G\to\bar G}^{L}
=
\operatorname{mean}
\frac{
\|\mathcal L_\eta(G)-\mathcal L_\eta(\bar G)\|_F^2
}{
\|\mathcal L_\eta(G)\|_F^2+\epsilon
}.
\]

该项对应理论中完整图到同节点软图的谱逼近。

### 5.4 GW/扩散几何保真

使用相同节点测度下的恒等耦合上界：

\[
L_{G\to\bar G}^{GW}
=
\operatorname{mean}
\operatorname{GW}_{\mathrm{id}}^2
\left(
\mathbb X_G,
\mathbb X_{\bar G}
\right).
\]

训练期允许使用可微的恒等耦合或扩散几何上界，不把它声明为不同节点集合上的 Exact GW。Exact GW 只用于冻结后的审计和下游计算。

---

## 6. 软图到硬图

### 6.1 硬选择

硬选择继续使用：

1. 每社区至少保留一个最高节点分数候选；
2. 按全局节点分数补足节点预算；
3. 只在原始有效边中建立候选边；
4. 使用：

\[
s_{ij}=p_{ij}\sqrt{p_ip_j}
\]

排序；
5. 按边预算选择硬边；
6. 最终硬节点为入选硬边端点；
7. 前向为真实 0/1 硬图，反向保留 STE。

### 6.2 谱量化误差

对软图和硬图的固定谱分位向量计算：

\[
L_{\bar G\to S}^{\lambda}
=
\operatorname{mean}
\left\|
Q_\lambda(\bar G)-Q_\lambda(S)
\right\|_1.
\]

该项是训练期对 \(q_{\lambda,c}\) 的样本级代理。

需要额外记录类别条件均值：

\[
\hat q_{\lambda,c}
=
\operatorname{mean}_{Y=c}
L_{\bar G\to S}^{\lambda}.
\]

### 6.3 GW量化误差

由于硬图节点数可以变化，训练期不得把同节点恒等耦合直接冒充 Exact GW。

第一版采用可微且计算可控的代理：

\[
L_{\bar G\to S}^{GW}
=
\operatorname{mean}
\left\|
Q_D(\bar G)-Q_D(S)
\right\|_1,
\]

其中 \(Q_D\) 是扩散距离分布的固定分位向量。

冻结后再使用真实不同节点集合 Exact GW 审计：

\[
\hat q_{\mathrm{GW},c}^{\mathrm{exact}}
=
\operatorname{mean}_{Y=c}
d_{\mathrm{GW}}(\mathbb X_{\bar G},\mathbb X_S).
\]

训练代理与冻结后 Exact GW 必须使用不同字段命名，禁止混称。

### 6.4 软硬分类一致性

软图和硬图共享同一个 Proxy 分类头，并增加：

\[
L_{\mathrm{soft\text{-}hard\_KD}}
=
T^2
\operatorname{KL}
\left(
\operatorname{softmax}(z_{\mathrm{soft}}/T)
\;\|\;
\operatorname{softmax}(z_{\mathrm{hard}}/T)
\right).
\]

默认由软图作为教师，硬图作为学生，教师 logits 在该项中停止梯度。

该项用于降低硬化导致的判别排序变化，但不能替代谱和 GW 量化误差。

---

## 7. 改进后的总损失

建议使用下式作为正式目标：

\[
\begin{aligned}
L_{\mathrm{selector}}
=\;&
\lambda_{\mathrm{soft\_cls}}
L_{\mathrm{cls}}^{\mathrm{soft}}
+
\lambda_{\mathrm{hard\_cls}}
L_{\mathrm{cls}}^{\mathrm{hard}}\\
&+
\lambda_L L_{G\to\bar G}^{L}
+
\lambda_{\mathrm{GW}} L_{G\to\bar G}^{GW}\\
&+
\lambda_{q_\lambda}
L_{\bar G\to S}^{\lambda}
+
\lambda_{q_{\mathrm{GW}}}
L_{\bar G\to S}^{GW}\\
&+
\lambda_{\mathrm{KD}}
L_{\mathrm{soft\text{-}hard\_KD}}\\
&+
\lambda_{B_n}L_{\mathrm{node\_budget}}
+
\lambda_{B_e}L_{\mathrm{edge\_budget}}.
\end{aligned}
\]

第一版不直接写死全部新权重。建议：

1. 保留当前分类与预算损失的量级作为起点；
2. 先在少量训练 batch 上记录每个未加权损失的均值和梯度范数；
3. 使各结构项初始加权贡献不压倒分类项；
4. 在正式比较前冻结权重，不根据 test 调整。

不建议一开始加入大量额外正则或复杂自动权重算法，以免无法判断两阶段传递本身是否有效。

---

## 8. 训练阶段

### 阶段 A：软图预热

- 只启用软图路径；
- 优化：

\[
L_{\mathrm{cls}}^{\mathrm{soft}}
+
L_{G\to\bar G}^{L}
+
L_{G\to\bar G}^{GW}
+
L_{\mathrm{budget}};
\]

- 不让硬图量化误差主导早期训练；
- 目标是先学习连续、稳定、非空且非完整的软选择概率。

### 阶段 B：软硬联合训练

- 启用社区覆盖 Top-k 和 STE 硬图；
- 加入硬图分类、谱量化、GW量化和软硬一致性；
- 继续保留完整图到软图保真；
- 节点和边预算保持不变。

### 阶段 C：冻结与审计

- 冻结 selector；
- 关闭随机性和 Dropout；
- 按训练时相同预算导出真实硬图；
- 在 train、validation 上计算：
  - 完整图→软图 Laplacian 误差；
  - 完整图→软图 GW 上界；
  - 软图→硬图谱量化误差；
  - 软图→硬图 Exact GW 量化误差；
  - Soft/Hard Proxy AUROC；
  - Learned/Random 同预算差异；
- 架构选择结束前不查看 test。

### 阶段 D：下游保持不变

- 使用冻结硬图计算 Exact-SGW；
- 沿用既定 train-only scaler；
- 沿用既定分类头与 validation 阈值冻结原则；
- test 仅做一次最终评估。

---

## 9. Checkpoint 选择

不建议只按单个训练损失或单个硬图 AUROC 保存。

第一版采用“主指标加约束”的方式：

- 主指标：validation Hard-Proxy AUROC；
- 必须同时满足：
  - 实际节点/边预算有效；
  - 无空硬图异常；
  - 完整图→软图误差有限；
  - 软图→硬图量化误差有限；
  - Soft 与 Hard 概率排序没有严重崩溃。

同时保存但暂不直接作为唯一 checkpoint 指标：

\[
\hat\Delta_{ab}^{G},
\qquad
\hat\Delta_{ab}^{\bar G},
\qquad
\hat\Delta_{ab}^{S},
\]

以及：

\[
\hat\eta_a,\qquad \hat\eta_b.
\]

高维经验 Wasserstein 间隔可能受 validation 样本量影响，因此第一版将其作为理论审计指标，而不是单独据此选择 checkpoint。待其估计稳定性得到验证后，再考虑使用：

\[
\hat{\mathcal M}
=
\hat\Delta_{ab}^{G}
-\hat\eta_a
-\hat\eta_b
\]

作为候选选择指标。

---

## 10. 必须新增的日志

每个 epoch 至少记录：

### 10.1 分类

- train/validation Soft-Proxy loss、AUROC；
- train/validation Hard-Proxy loss、AUROC；
- Soft–Hard 概率 Pearson、Spearman；
- Soft–Hard 冻结阈值预测不一致率。

### 10.2 选择分布

- 节点概率 mean/std/min/max；
- 边概率 mean/std/min/max；
- 实际节点比例与边比例；
- 空硬图窗口数；
- 每社区覆盖率；
- 被选正边、负边数量及比例。

### 10.3 两阶段误差

- \(L_{G\to\bar G}^{L}\)；
- \(L_{G\to\bar G}^{GW}\)；
- \(L_{\bar G\to S}^{\lambda}\)；
- \(L_{\bar G\to S}^{GW}\)；
- 按类别分别汇总上述误差；
- 冻结后 Exact \(q_{\mathrm{GW},c}\)。

### 10.4 梯度

- 节点评分器梯度范数；
- 边评分器梯度范数；
- 每个损失项对两个评分器的梯度贡献；
- NaN/Inf 检查。

---

## 11. 最小验证实验

为控制计算量，第一轮只比较四个条件：

| 条件 | 完整图→软图 | 软图→硬图 | 分类监督 | 用途 |
|---|---|---|---|---|
| E0 Current Learned | 未显式拆分 | STE隐式 | Hard Proxy | 当前基线 |
| E1 Random | 不训练 | 同预算随机 | 下游同流程 | 随机压缩基线 |
| E2 Full→Soft | 显式 | 不加量化损失 | Soft + Hard | 验证软图路径 |
| E3 Full→Soft→Hard | 显式 | 显式谱/GW量化 | Soft + Hard + KD | 完整改进 |

第一轮只使用一个 seed 和固定划分：

1. 先在 WMRC 上运行；
2. 再在 ADHD 的现成 fold 上复用；
3. 所有条件使用完全相同的预算、协议、数据顺序和训练样本；
4. 不使用 test 做选择；
5. 若 E3 未在 validation 上稳定优于 E0 和 E1，则停止扩展，不进入多 seed 或 OOF；
6. 若 E3 在两个数据集或多个现成折上均表现出方向一致的增益，再进行配对 OOF。

---

## 12. 验收标准

### 12.1 工程验收

- 正负边符号全部保留；
- 不存在 \(A_{ij}>0\) 的错误边判断；
- 可变 \(M\)、可变 \(N\) 正常；
- padding 不影响损失和输出；
- Soft 与 Hard 路径梯度均可到达节点和边评分器；
- train-only scaler 和 validation 阈值规则不变；
- 旧 selector checkpoint 和旧实验入口仍可复现；
- 新 checkpoint 带有明确的模型版本和损失配置。

### 12.2 结构验收

- 软图既不退化为空图，也不退化为完整图；
- 实际硬图预算满足预设比例；
- \(G\to\bar G\) 与 \(\bar G\to S\) 误差均可独立报告；
- 冻结后 Proxy GW 与 Exact GW 的字段和语义明确区分；
- Learned 选择的正负边构成可审计。

### 12.3 实验验收

最低要求：

1. E3 的 Learned selector 在 validation 上不低于同预算 Random；
2. E3 相对 E0 的增益方向在 WMRC 和 ADHD 现成折中一致；
3. Soft→Hard 后 AUROC 和排序相关性没有显著坍缩；
4. 新增结构项没有让分类梯度消失；
5. 配对 OOF 后才允许形成最终性能结论。

如果 E3 只能降低结构误差，却不能提升或保持分类表现，则只能说明结构保真提高，不能声称 selector 的判别能力得到改善。

---

## 13. 必要单元测试

### 13.1 软图符号

给定：

\[
A_{ij}=-0.8,\quad p_i,p_j,p_{ij}>0,
\]

必须满足：

\[
\bar A_{ij}<0.
\]

### 13.2 软图同节点空间

- 软图节点数必须等于完整图节点数；
- 低概率节点不得在软阶段被物理删除。

### 13.3 软硬图差异

- Soft adjacency 为连续值；
- Hard adjacency 前向为真实 0/1 mask 后的原始带符号边；
- 两者不得引用同一个未区分语义的张量。

### 13.4 梯度

- Soft 分类损失可更新节点和边评分器；
- 谱量化损失可更新节点和边评分器；
- GW量化代理可更新节点和边评分器；
- 单独关闭任一项时其梯度贡献必须为零。

### 13.5 可变长度与mask

- 不同样本的 \(M\) 和 \(N\) 可不同；
- 无效窗口不参与平均；
- padding 不改变输出和损失。

### 13.6 随机基线

- 相同 seed 产生相同硬图；
- Random 与 Learned 使用相同节点和边预算；
- Random 不接受 selector 梯度。

### 13.7 导出兼容性

- 新 selector 导出的硬图满足现有 Exact-SGW manifest；
- 原 D3 下游无需修改即可读取；
- checkpoint 配置不匹配时必须显式报错。

---

## 14. 风险与边界

### 14.1 完整图间隔是前提，不是自动事实

理论要求完整图谱–GW表示具有类别间隔。若所选完整图表示本身没有稳定类别信号，则降低提取误差也不保证分类性能提高。

### 14.2 保真不等于复制所有信息

完整图可能同时包含：

- 判别结构；
- 与类别无关的结构；
- 数据来源或采集条件造成的差异；
- 随机噪声。

因此不能只最大化完整图重建精度。分类损失必须与结构保真共同存在。

### 14.3 理论条件是充分条件

若：

\[
\Delta_{ab}
\le
\eta_a+\eta_b,
\]

理论只是不再保证硬图可分，不能反推硬图必然不可分。

### 14.4 Proxy 与 Exact GW 必须区分

训练期的可微 GW-aligned proxy 用于优化；冻结后的 Exact GW 用于审计与正式特征计算。两者不能在报告中混写。

---

## 15. 推荐实施顺序

1. 新增显式 signed soft adjacency；
2. 增加 Full→Soft Laplacian 与 GW proxy；
3. 保留现有硬选择和导出；
4. 增加 Soft→Hard 谱量化代理；
5. 增加 Soft→Hard GW量化代理；
6. 增加 Soft/Hard 共享 Proxy 分类头和一致性损失；
7. 增加两阶段、类别条件和梯度日志；
8. 完成单元测试与本地 dummy-data 流程；
9. 服务器运行 E0–E3 单 seed；
10. 仅在方向一致且优于 Random 后进入配对 OOF。

---

## 16. 最终定位

改进后的 selector 不是单纯的“高分类分数节点/边筛选器”，而是：

> 一个在固定压缩预算下，以分类监督为目标、以完整图到软图的谱–GW保真和软图到硬图的量化误差为约束的带符号关键子图提取器。

它直接对应理论中的：

\[
G\rightarrow\bar G\rightarrow S,
\]

并通过分别控制两个阶段的误差，尝试使：

\[
W_1(P_a^S,P_b^S)
\ge
\Delta_{ab}-\eta_a-\eta_b
\]

在经验数据上具有可观察、可审计的正裕量。
