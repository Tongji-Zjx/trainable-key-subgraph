# 理论引导多视图关键子图网络：修订校验版架构与实施规范

## 0. 文档状态

- 文档用途：指导后续程序实现、单元测试、服务器训练和 OOF 实验验证。
- 修订基础：`Theory_Guided_Multi_view_Critical_Subgraph_Network_Codex_Spec.md`。
- 理论依据：以当前“关键子图谱–GW演化理论”核心推导版为准。
- 兼容要求：Python 3.7、PyTorch 1.13.1。
- 当前状态：架构定义冻结；实际张量维度、损失权重和计算预算需在 Stage 0 审计后写入配置，不得散落为代码常量。

本版已经采纳以下关键修正：

1. 静态 S 分支使用 **Signed Spectral GCNII**，并保留显式谱状态解码器。
2. 原有稳定 Static-spectral 统计表示保留为锚点，神经表示以零/小门控残差加入。
3. 不直接使用未经处理的 signed-Laplacian 特征向量，改用基底不变的谱初始化。
4. 每个窗口允许包含数量可变的关键结构对象，不再固定为三个子图。
5. signed diffusion-FGW 生成跨对象代价，非平衡/部分最优传输负责求对应关系，不执行“GW 套 GW”。
6. G 分支仅作为完整图补充编码器，不增加独立解码器。
7. 所有预处理、scaler、缓存和模型选择均遵守外折训练边界。

---

## 1. 目标与非目标

### 1.1 总体目标

模型同时学习三类信息：

- **S：静态关键谱结构**——回答每个窗口的关键图具有怎样的谱状态；
- **V：多关键结构的跨窗口演化**——回答独立关键结构如何对应、变化、出现或消失；
- **G：完整图拓扑补充**——补偿硬选择可能遗漏的局部有符号拓扑信息。

整体数据流为：

\[
\text{完整带符号图序列}
\rightarrow
\text{软关键图}
\rightarrow
\text{硬关键图}
\rightarrow
\begin{cases}
\text{S：Signed Spectral GCNII}\\
\text{V：多对象 signed diffusion-FGW 演化}\\
\text{G：完整图有符号编码补充}
\end{cases}
\rightarrow
\text{受控残差融合}
\rightarrow
\text{分类}
\]

### 1.2 非目标

本阶段不做以下事项：

- 不使用空间坐标、ROI 身份、站点标签作为模型输入；
- 不使用原始 `community_id` embedding；
- 不假设所有样本具有相同的窗口数、节点数或关键结构数；
- 不通过截断改变原始图结构；
- 不把负边视为无边，也不把邻接矩阵整体替换为绝对值；
- 不给 G 分支增加独立理论解码器；
- 不在外折 test 上选择阈值、超参数、特征或模型变体。

---

## 2. 数据与带符号图约束

对样本 \(b\) 的第 \(m\) 个窗口，记完整图为：

\[
G_b^{(m)}=(V_b^{(m)},A_b^{(m)},X_b^{(m)})
\]

其中窗口数为 \(M_b\)，节点数为 \(N_b^{(m)}\)。

统一边存在判断：

\[
M_{ij}^{(m)}=\mathbf 1\left(|A_{ij}^{(m)}|>\tau_{edge}\right)
\]

阈值 \(\tau_{edge}\) 必须来自冻结的数据协议或统一配置，并由训练、硬图导出、理论特征计算和统计分析共同使用。

正负邻接分解：

\[
A_{ij}^{+}=M_{ij}\max(A_{ij},0),\qquad
A_{ij}^{-}=M_{ij}\max(-A_{ij},0)
\]

绝对度：

\[
D_{ii}=\sum_j(A_{ij}^{+}+A_{ij}^{-})
      =\sum_j|A_{ij}|M_{ij}
\]

所有模块必须保留：

- 正连接和负连接的独立语义；
- 原始边符号；
- 变化方向与变化幅度；
- `time_mask`、`node_mask`、`subgraph_mask` 和 `transition_mask`。

批处理优先采用 list-based batching。任何 padding 都只能用于张量化计算，并且不得参与池化、损失或统计。

---

## 3. 软硬关键图提取与对象分解

### 3.1 软图

当前可训练提取器产生节点概率 \(p_i\) 和边概率 \(p_{ij}\)，软邻接保持符号：

\[
A_{ij}^{soft}=A_{ij}p_ip_jp_{ij}
\]

软图训练负责让选择器学习紧凑且具有判别力的关键区域；预算、谱保持和其他选择器损失沿用独立冻结的选择器训练规范。

### 3.2 硬图

选择器冻结后，逐窗口导出一个合并硬关键图：

\[
H_b^{(m)}=\operatorname{HardExport}
\left(G_b^{(m)},p^{(m)},p_E^{(m)}\right)
\]

重要约束：

- S 和 V 分支共用同一份硬图导出结果；
- 硬图保留原始边权和符号；
- 下游训练的梯度不穿过硬导出操作；
- 每个外折只能使用该折训练数据训练的 selector；
- selector checkpoint、协议和缓存必须记录 SHA256 provenance。

### 3.3 窗口内关键结构对象

V 分支进一步将 \(H_b^{(m)}\) 分解为数量可变的对象集合：

\[
\mathcal S_b^{(m)}=
\left\{S_{b,1}^{(m)},\ldots,S_{b,K_{b,m}}^{(m)}\right\}
\]

其中 \(K_{b,m}\) 不固定。默认候选分解规则：

1. 使用社区标签形成社区诱导子图；
2. 对社区诱导子图按边存在 mask 继续划分连通分量；
3. 保留对象内部原始有符号边；
4. 将对象间连接汇总为窗口级 coupling 特征，而不是静默删除其信息。

不得在未经 Stage 0 审计的情况下硬编码 `K=3`、最小对象大小或 Top-K 截断。若存在大量单节点/微小对象，应先报告分布，再确定可复现的合并策略。

---

## 4. S 分支：静态关键谱结构

### 4.1 输入

S 分支直接输入每个窗口的合并硬关键图 \(H_b^{(m)}\)，不使用旧式提取器，也不先拆成多个对象。

节点结构特征使用当前经过验证的、跨样本具有一致语义的坐标无关特征，包括：

- 绝对连接强度；
- 正连接强度与负连接幅值；
- 正负连接比例；
- 连接强度变化；
- 社区规模；
- 社区内/外正连接强度；
- 社区内/外负连接幅值；
- 其他已验证的社区结构统计。

禁止输入原始社区编号 embedding。

### 4.2 signed-Laplacian 与稳定谱初始化

采用绝对度 signed Laplacian：

\[
L_s=D_{|A|}-A
\]

它满足：

\[
x^\top L_sx=
\frac{1}{2}\sum_{ij}|A_{ij}|
\left(x_i-\operatorname{sgn}(A_{ij})x_j\right)^2
\]

不直接把 \(u_1(i),\ldots,u_K(i)\) 作为跨样本节点身份特征，因为特征向量存在符号翻转和重根子空间旋转的不唯一性。

默认谱初始化使用：

- 多尺度 signed heat-kernel signature：

\[
\operatorname{HKS}_i(t_r)=
\sum_k e^{-t_r\lambda_k}u_k(i)^2
\]

- 分谱带投影对角量；
- signed diffusion 对角响应；
- 可选的 signed Chebyshev 滤波响应。

最终初始特征：

\[
X_i^S=
\left[x_i^{struct};x_i^{spectral\_invariant}\right]
\]

谱尺度、谱带边界和目标标准化参数只允许由外折 train 拟合。

### 4.3 Signed Spectral GCNII 编码器

正负传播矩阵使用共同的绝对度归一化：

\[
P=D^{-1/2}A^+D^{-1/2},\qquad
N=D^{-1/2}A^-D^{-1/2}
\]

孤立节点的传播消息置零，自身信息由 GCNII initial connection 保留。单位阵不作为“负自环”加入 \(A^-\)。

初始化两个可学习符号状态：

\[
H_+^0=\phi_+(X^S),\qquad
H_-^0=\phi_-(X^S)
\]

第 \(l\) 层先执行符号状态传播：

\[
M_+^l=PH_+^l+NH_-^l
\]

\[
M_-^l=PH_-^l+NH_+^l
\]

正边保持符号状态，负边交换符号状态。随后分别应用 GCNII 更新：

\[
Z_s^l=(1-\alpha_l)M_s^l+\alpha_lH_s^0
\]

\[
H_s^{l+1}=\sigma\left(
Z_s^l\left[(1-\beta_l)I+\beta_lW_s^l\right]
\right),\qquad s\in\{+,-\}
\]

正负状态贯穿全部网络层，只在最终节点表示处融合：

\[
H_{signed}^{(m)}=
\operatorname{LayerNorm}\left(
\operatorname{MLP}
\left[H_+^L;H_-^L\right]
\right)
\]

初始建议而非冻结超参数：

- 层数：2–4；
- hidden dimension：64；
- dropout：0.1；
- 每层保留 node mask；
- 必须记录各层方差、有效秩和平均余弦，检查低秩化。

### 4.4 图级与时间级编码

每个窗口使用掩码池化：

\[
h_S^{(m)}=
\left[
\operatorname{MaskedMean}(H_{signed}^{(m)});
\operatorname{MaskedStd}(H_{signed}^{(m)});
\operatorname{GatedAttentionPool}(H_{signed}^{(m)})
\right]
\]

attention 必须报告归一化熵和最大节点权重；若持续接近均匀，不得宣称其学到了关键节点选择。

样本级神经静态表示：

\[
z_S^{neural}=
\operatorname{MaskedTemporalPool}
\left(\{h_S^{(m)}\}_{m=1}^{M_b}\right)
\]

### 4.5 谱状态解码器

对每个有效窗口显式解码理论谱状态：

\[
\widehat Q_b^{(m)}=D_S(h_{S,b}^{(m)})
\]

\[
\mathcal L_S=
\frac{
\sum_{b,m}t_{b,m}
\operatorname{Huber}
\left(\widehat Q_b^{(m)},Q_b^{(m)}\right)
}{
\sum_{b,m}t_{b,m}+\epsilon
}
\]

其中 \(Q\) 的具体字段和顺序必须来自理论特征 schema，不得在模型内部重新定义；目标 scaler 仅由外折 train 拟合。

decoder 用于训练期的理论约束。推理时分类路径只需要编码器表示，不依赖 decoder 输出。

### 4.6 稳定统计锚点

原 Static-spectral 统计分支不删除。记其样本表示为 \(s_S^{stable}\)，则：

\[
z_S^{base}=P_{base}(s_S^{stable})
\]

\[
z_S=z_S^{base}+\tanh(g_S)P_{neural}(z_S^{neural})
\]

其中 \(g_S\) 初始化为 0 或极小值。这样模型初始状态等价或近似等价于稳定基线，神经编码器只有在训练数据支持时才增加残差。

### 4.7 完整 S 配置与分阶段开关

正式实现默认包含完整 S 分支，而不是只实现最小可运行子集：稳定统计、三尺度 HKS、谱带投影、Chebyshev 响应、双流 Signed Spectral GCNII、masked mean/std、gated attention 和 \(Q\) decoder 均进入真实前向。分阶段开关仅用于消融与定位，不代表相应模块可以缺省实现。

实现必须提供互斥模式：

- \`static_mode=stable\`：只使用稳定统计锚点；
- \`static_mode=neural\`：只使用 Signed Spectral GCNII 和 \(Q\) decoder；
- \`static_mode=residual\`：稳定锚点 + 门控神经残差，作为最终候选。

附加模块采用独立开关：

- \`static_attention=false\` 只作为 attention 消融；正式完整配置为 `true`；
- 额外谱带投影和 Chebyshev 响应属于正式 9 维谱不变量 schema，不在正式配置中关闭；
- \`enable_v=false\` 用于单独验收 S；
- \`enable_g=false\` 用于单独验收 S/V。

\(Q\) decoder 属于理论监督闭环，在 neural 和 residual 模式下不得关闭。工程验收可分支逐项进行；最终完整候选必须同时具备已实现的 S、V、G，仅通过门控控制实际贡献。

---

## 5. V 分支：多关键结构的 GW 演化

### 5.1 对象编码

对每个关键结构对象共享编码器参数：

\[
e_{b,i}^{(m)}=E_V(S_{b,i}^{(m)})
\]

默认使用浅层、极性敏感的 signed object encoder，并包含：

- 对象内部正负连接传播；
- 对象级 mean/std pooling；
- 对象规模、正负密度和社区覆盖等上下文；
- 低秩和过平滑诊断。

该编码器不得依赖对象编号、社区编号或固定对象槽位。

### 5.2 signed diffusion-FGW 跨窗口代价

对相邻有效窗口 \(m,m+1\)，逐对象对计算：

\[
C_{ij}^{(m)}=
d_{SDFGW}
\left(S_i^{(m)},S_j^{(m+1)}\right)
\]

距离同时比较：

- 节点/对象属性；
- signed diffusion geometry；
- 多尺度谱结构；
- 正负连接模式。

这里的 SDFGW 只生成跨对象代价矩阵 \(C^{(m)}\)，不再对对象集合执行第二次 GW。

### 5.3 非平衡/部分最优传输对应

为容纳对象出现、消失和质量不守恒，默认求解：

\[
\pi^{(m)}=
\arg\min_{\pi\ge 0}
\langle\pi,C^{(m)}\rangle
+\varepsilon_{ot}\operatorname{KL}(\pi\|ab^\top)
+\tau_a\operatorname{KL}(\pi\mathbf1\|a)
+\tau_b\operatorname{KL}(\pi^\top\mathbf1\|b)
\]

若采用 partial OT，必须在配置和 artifact schema 中明确匹配质量预算。平衡 OT 只能作为对照，不作为默认方案。

### 5.4 对应后的演化表示

对当前对象 \(i\)，匹配质量：

\[
r_i^{(m)}=\sum_j\pi_{ij}^{(m)}
\]

归一化未来表示：

\[
\widetilde e_i^{(m+1)}=
\frac{
\sum_j\pi_{ij}^{(m)}e_j^{(m+1)}
}{r_i^{(m)}+\epsilon}
\]

不得直接使用未归一化加权和，否则对象表示尺度会随匹配质量变化。

转移 token 至少包含：

\[
t_i^{(m)}=
\left[
e_i^{(m)};
\widetilde e_i^{(m+1)};
\frac{\widetilde e_i^{(m+1)}-e_i^{(m)}}{\Delta t_m};
\frac{|\widetilde e_i^{(m+1)}-e_i^{(m)}|}{\Delta t_m};
r_i^{(m)};
\bar C_i^{(m)};
c_{coupling}^{(m)}
\right]
\]

其中 \(\bar C_i\) 是匹配代价摘要，\(c_{coupling}\) 是对象间连接上下文。

对象级 token 先经过 `subgraph_mask` 聚合，形成第 \(m\rightarrow m+1\) 个有效转移的输入：

\[
u^{(m)}=
\operatorname{MaskedObjectPool}\{t_i^{(m)}\}
\]

时序编码器正式确定为**单层单向门控残差 GRU**：

\[
h_{GRU}^{(m)}
=\operatorname{GRU}
\left(u^{(m)},h_{GRU}^{(m-1)}\right)
\]

\[
h_V^{(m)}
=u^{(m)}
+\tanh(g_T)P_T\left(h_{GRU}^{(m)}\right)
\]

其中：

- \(g_T\) 初始化为 0 或 0.01；
- GRU hidden dimension 初始设为 64；
- GRU 层数为 1，不使用双向 GRU；
- dropout 0.1 放在 GRU 输入和输出处；
- 使用 packed sequence 或严格的 `transition_mask`；
- 不直接使用最后一个 hidden state 表示整个样本；
- 不使用 Transformer。

单向 GRU 保持 \(m\rightarrow m+1\) 的演化方向。残差门控使模型初始状态近似 independent-transition 表示，避免递归传播无效时破坏原始 GW token。

样本级动态表示使用全部有效转移的 masked mean/std：

\[
z_V=
\left[
\operatorname{MaskedMean}
\{h_V^{(m)}\}_{m=1}^{M_b-1};
\operatorname{MaskedStd}
\{h_V^{(m)}\}_{m=1}^{M_b-1}
\right]
\]

### 5.5 演化解码器

V 分支逐有效转移预测理论谱变化率：

\[
\widehat{\Delta Q}_b^{(m)}/\Delta t_m
=D_V(h_{V,b}^{(m)})
\]

\[
\mathcal L_V=
\frac{
\sum_{b,m}r_{b,m}
\operatorname{Huber}\left(
D_V(h_{V,b}^{(m)}),
\Delta Q_b^{(m)}/\Delta t_m
\right)
}{
\sum_{b,m}r_{b,m}+\epsilon
}
\]

这里 \(r_{b,m}\) 表示有效 transition mask，不是对象匹配质量。

---

## 6. G 分支：完整图有符号拓扑补充

### 6.1 定位

G 分支编码原始完整图序列，补充硬选择可能遗漏的局部拓扑信息。它不是理论主干，也不得无条件主导融合。

### 6.2 输入与编码

使用当前冻结 schema 中经过验证的坐标无关节点和有符号边特征。当前实现若为 15 维节点、6 维边特征，应从 manifest/schema 读取并断言，禁止把维度作为无来源常量写死。

每个窗口执行轻量 signed graph encoder，再通过 masked temporal mean/std 得到：

\[
z_G=E_G\left(\{G^{(m)}\}_{m=1}^{M_b}\right)
\]

必须监控：

- 窗口表示方差；
- 有效秩；
- 平均样本余弦；
- 通道屏蔽后的 AUROC 变化；
- 正负消息抵消比例。

### 6.3 无独立解码器

按照当前已确认设计，G 分支**不增加独立 decoder**。其参数只接受分类损失和融合路径的梯度。

若 G 分支造成负迁移，应通过门控回到零贡献，而不是依赖附加重建任务强行保留该分支。

---

## 7. 关键子图通道、作者短期分支与最终融合

### 7.1 关键子图通道内部融合

S、V、G 先组成独立的关键子图通道：

\[
z_C=z_S+\tanh(g_V)P_V(z_V)+\tanh(g_G)P_G(z_G)
\]

默认初始化：

- \(g_S\)：0 或 0.01；
- \(g_V\)：0.01，使动态分支可逐步进入；
- \(g_G\)：0，使完整图补充分支必须用数据证明增益。

关键子图通道保留独立轻量分类头，用于模块筛选和消融：

\[
\widehat y_C=f_C(z_C)
\]

### 7.2 作者短期分支

最终架构还包含独立的作者短期分支。记其样本表示为：

\[
z_{ST}=E_{ST}\left(\{G^{(m)}\}_{m=1}^{M_b}\right)
\]

短期分支以作者程序的坐标无关复现为准，与关键子图通道使用同一份样本索引、划分和标签。两个通道不得使用不同的 train/validation/test 成员或不同的样本顺序。

短期分支保留独立分类头：

\[
\widehat y_{ST}=f_{ST}(z_{ST})
\]

以便分别报告 Short-Term only、Critical only 和 Fusion，避免只报告融合结果而无法判断增益来源。

### 7.3 最终表示级门控残差融合

最终默认采用以短期分支为锚点的表示级残差融合：

\[
z_F=P_{ST}(z_{ST})
+\tanh(g_C)P_C(z_C)
\]

\[
\widehat y_F=f_F(z_F)
\]

其中 \(g_C\) 从 0 或 0.01 初始化。这样最终模型初始状态回退到短期分支，关键子图通道只有在 validation 支持时才增加残差。

首轮融合训练遵循：

1. 分别训练并冻结短期分支和关键子图通道；
2. 只训练 \(P_{ST}\)、\(P_C\)、\(g_C\) 和轻量融合分类头；
3. 若融合形成稳定 validation 增益，才允许进行低学习率的末层联合微调；
4. 不允许使用 test 选择融合权重、门控、阈值或微调轮数。

决策级等权融合可以作为廉价对照，但不作为最终默认架构。

### 7.4 损失和模型选择

关键子图通道训练损失：

\[
\mathcal L_C=
\mathcal L_{cls,C}
+\lambda_S\mathcal L_S
+\lambda_V\mathcal L_V
\]

冻结双通道编码器后的融合训练只使用：

\[
\mathcal L_F=\mathcal L_{cls,F}
\]

selector 冻结时，不把 selector 损失加入下游训练。若未来进行软图联合训练，必须另立实验协议，并明确硬图路径不可微。

类别加权只能由当前 train 的类别数计算。模型选择默认使用 validation AUROC；分类阈值只在 validation 上确定并冻结到 test。

“seed43”必须拆分记录为：

- split seed：决定有利划分；
- model seed：决定初始化与数据加载顺序。

如果某个划分已经因既往 test 表现而被选为“有利划分”，其结果只能标记为探索性性能结果，不能当作无偏泛化估计。

---

## 8. 缓存、数据边界与 artifact 规范

### 8.1 必须离线缓存的内容

以下高成本内容按外折、split 和 selector provenance 缓存：

- 硬关键图；
- 关键结构对象及对象间 coupling；
- signed-Laplacian 谱量与 HKS；
- 理论目标 \(Q\) 与 \(\Delta Q/\Delta t\)；
- 对象对 signed diffusion-FGW 代价；
- UOT/partial-OT 对应矩阵及质量摘要。

Exact FGW 不得在每个训练 epoch 重复计算。

### 8.2 数据泄漏约束

每个外折必须独立生成或拟合：

- selector；
- stable S scaler；
- 谱特征 scaler；
- \(Q\) 与 \(\Delta Q\) target scaler；
- OT/FGW 中任何由队列统计估计的尺度；
- 分类阈值。

validation 和 test 只能应用训练折冻结的参数。

### 8.3 provenance

每个 manifest 至少记录：

- 数据协议 SHA256；
- sample index 与 split SHA256；
- selector checkpoint SHA256；
- edge threshold；
- 特征 schema 版本；
- 对象分解策略及版本；
- 谱参数；
- FGW/OT 参数；
- scaler SHA256；
- 生成代码的 git commit。

默认不覆盖已有 artifact；只有显式 `--overwrite` 才允许重建。

---

## 9. 实施阶段

### Stage 0：只读可行性审计

不训练模型，统计：

- 每窗口 \(K_{b,m}\) 分布；
- 对象节点数、边数和正负边比例；
- 单节点/微小对象比例；
- 相邻窗口对象数量差；
- FGW 单对象对耗时和显存；
- UOT 匹配质量、未匹配质量和代价分布；
- 预计单折缓存大小与总耗时。

只有审计表明对象分解和对应具有可计算性，才进入 Stage 1。

### Stage 1：S 分支神经化

固定 selector 和数据划分，比较：

1. `S_stable`：原稳定 Static-spectral；
2. `S_neural`：Signed Spectral GCNII；
3. `S_residual`：稳定锚点 + 神经残差（默认候选）。

必须同时报告分类 AUROC、\(Q\) 解码误差和表示秩，不能只看分类结果。

### Stage 2：V 分支

在最佳 S 上加入：

1. 无 V；
2. 原 Variation；
3. 多对象 SDFGW + UOT 演化；
4. 对应打乱负对照。

Stage 2 的核心验收是：真实对应优于打乱对应，并且 V 在至少一个数据集的 paired OOF 中形成正增益。

### Stage 3：G 分支

比较：

1. `S+V`；
2. `S+V+G(gated)`；
3. G 通道屏蔽诊断。

如果 G 没有形成可复现增益，则正式架构中保持 \(g_G=0\) 或移除推理计算，但保留代码开关。

### Stage 4：作者短期分支融合与有利划分实验

在固定的有利划分（例如 split seed 43）上，使用完全一致的 train/validation/test 成员训练：

1. Author Short-Term only；
2. Critical S/V/G only；
3. Short-Term + Critical 表示级门控残差融合。

融合结构、checkpoint epoch 和分类阈值只能由 validation 确定。test 只执行一次最终评估。该阶段的目标是获得较好的探索性分类性能，并验证关键子图通道相对于短期分支是否产生增量。

### Stage 5：可选的完整 OOF 泛化验证

若需要报告无偏泛化证据，只对 Stage 4 冻结的一个最终配置执行完整 OOF。不得根据 outer-test 或 OOF 结果返回修改模型。OOF 与有利划分实验必须分开报告，不能混为同一证据。

---

## 10. 必要单元测试与诊断

### 10.1 Signed Spectral GCNII

- `A_ij < 0` 必须进入负通道；
- 节点置换后图级输出保持一致；
- 一致置换节点、邻接和 mask 后结果不变；
- isolated node 不产生 NaN；
- padding 不改变有效图输出；
- 梯度能到达 \(\phi_+\)、\(\phi_-\)、GCNII 层、融合层和 \(D_S\)；
- 谱特征向量符号翻转不改变 HKS/投影特征；
- \(Q\) loss 忽略无效窗口。

### 10.2 对象与 GW 演化

- 对象编号置换不改变样本输出；
- 不同 \(K_m\) 可正常 batch；
- 相同对象的最小代价匹配可被恢复；
- 对象出现/消失时 UOT 不强制错误的一一匹配；
- \(\pi\) 归一化后的表示不随整体质量缩放；
- transition padding 不影响输出；
- 对应打乱诊断可复现。

### 10.3 G 分支和融合

- G 分支保留负边符号；
- \(g_G=0\) 时输出严格等于无 G 路径；
- \(g_S=0\) 时 S 输出严格回退到稳定锚点；
- \(g_C=0\) 时最终融合严格回退到短期分支锚点；
- 短期分支与关键子图通道的 sample ID、标签和序列顺序必须逐项一致；
- batch 一致置换后两个通道及融合输出同步置换；
- 通道屏蔽不修改冻结权重；
- 不存在 G 分支 decoder 或对应 loss。

### 10.4 artifact 与泄漏

- train/validation/test manifest provenance 一致；
- scaler 只声明 train 来源；
- outer-test 标签不参与阈值拟合；
- 缓存不可被默认覆盖；
- checkpoint 加载严格校验 schema 和输入维度。

---

## 11. 验收指标

### 11.1 工程验收

- CPU dummy forward、loss、backward、save/load 全部通过；
- 16 样本过拟合测试能够明显降低分类损失和理论解码损失；
- 真实少量数据 CUDA smoke 通过；
- 不出现 NaN、符号丢失、mask 泄漏或 artifact hash 不一致；
- Python 3.7 与 PyTorch 1.13.1 兼容。

### 11.2 表示验收

每折记录：

- 各层方差、有效秩、归一化有效秩和平均余弦；
- 正负消息范数及抵消率；
- attention 熵和有效节点数；
- S/V/G 通道屏蔽 AUROC；
- \(Q\) 与 \(\Delta Q\) 解码误差；
- UOT 匹配质量、未匹配质量和对应稳定性。

### 11.3 分类验收

有利划分的探索性主要指标：

- Fusion AUROC；
- Fusion accuracy 和 balanced accuracy；
- Fusion 相对 Author Short-Term only 的 AUROC 增量；
- Fusion 相对 Critical only 的 AUROC 增量。

若执行 OOF，泛化主要指标为：

- Mean-fold outer-test AUROC；
- paired fold AUROC difference。

共同辅助指标：

- 各折 AUROC；
- pooled OOF AUROC；
- site-stratified OOF AUROC；
- balanced accuracy、accuracy、F1、sensitivity、specificity；
- paired bootstrap 置信区间和折内配对差异。

有利划分结果用于获得和展示较好的探索性分类性能；OOF 结果用于估计泛化能力。两种目标和证据等级必须分别表述。

---

## 12. 最终冻结架构摘要

最终计划架构为：

\[
G^{(m)}
\rightarrow
G_{soft}^{(m)}
\rightarrow
H^{(m)}
\]

\[
H^{(m)}
\xrightarrow[
\text{stable spectral anchor}
]{
\text{Signed Spectral GCNII Encoder}
}
h_S^{(m)}
\xrightarrow{D_S}
\widehat Q^{(m)}
\]

\[
H^{(m)}
\rightarrow
\{S_i^{(m)}\}_{i=1}^{K_m}
\rightarrow
\text{SDFGW cost}
\rightarrow
\text{UOT correspondence}
\rightarrow
h_V^{(m)}
\xrightarrow{D_V}
\widehat{\Delta Q}^{(m)}/\Delta t
\]

\[
G^{(m)}
\rightarrow
E_G
\rightarrow
z_G
\qquad\text{（无独立 decoder）}
\]

\[
z_C=z_S+\tanh(g_V)P_V(z_V)+\tanh(g_G)P_G(z_G)
\]

\[
z_{ST}=E_{ST}\left(\{G^{(m)}\}_{m=1}^{M_b}\right)
\]

\[
z_F=P_{ST}(z_{ST})+\tanh(g_C)P_C(z_C)
\rightarrow
f_F
\rightarrow
\widehat y_F
\]

该设计在保留已验证稳定谱统计的同时，引入满足理论要求的神经编码—解码闭环、多关键结构的显式跨窗口对应和可自动退回零贡献的完整图补充分支；最终再将关键子图通道作为门控残差融合到作者短期分支。
