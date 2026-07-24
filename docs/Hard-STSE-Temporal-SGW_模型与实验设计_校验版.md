# Hard-STSE-Temporal-SGW：模型架构与实验设计（校验版）

## 1. 文档目的

本文档给出一个以分类性能为第一目标、同时保留硬关键图提取与谱–GW演化理论路径的单模型方案。

模型名称：

\[
\boxed{\text{Hard-STSE-Temporal-SGW}}
\]

核心要求：

- 取消教师–学生与知识蒸馏；
- 不使用 Transformer；
- 主模型不使用 ROI 空间坐标；
- 节点名称只用于相邻窗口对齐，不作为默认判别特征；
- 社区编号只用于同社区判断与社区结构统计，不直接使用原始编号 embedding；
- 前向分类只使用真正减少节点和边的硬关键图；
- 所有神经分类特征、谱特征和 GW 特征都从硬关键图重新计算；
- 保留与理论直接对应的 18 维谱–GW核心表示；
- 为提高分类性能，允许使用节点/边残差 MLP、注意力池化、一层 BiGRU、SGW序列编码和多头监督。

---

# 2. 当前问题与设计动机

此前模型包含软教师、硬导出、硬学生、蒸馏和双分支融合，存在：

1. 训练阶段多，调试困难；
2. 软教师分类能力限制提取器；
3. 软图节点数与原图一致，不能直接满足冗余压缩要求；
4. 软图到硬图再到学生的误差链过长；
5. 当前应优先建立有效分类能力。

新模型改为：

```text
完整动态图
→ 硬关键图选择
→ Hard-STSE神经统计编码
→ 谱–GW演化编码
→ 直接分类
```

参考原论文 STSE 的原因：STSE 使用显式结构特征、共享残差 MLP 和轻量图级聚合，不依赖多层 GNN，因此能避免稠密图上的过平滑。但原 STSE 使用坐标、主要依赖节点统计且时间聚合定义不完整，所以必须改造。

---

# 3. 总体架构

```text
完整动态图序列
        │
        ▼
节点名称对齐、时间差分、社区结构特征
        │
        ▼
STSE式节点评分器 + 边评分器
        │
        ▼
社区感知硬节点/硬边选择（STE）
        │
        ▼
硬关键并图序列 U(1), ..., U(M)
        │
        ├──────────────────────────┐
        ▼                          ▼
增强型 Hard-STSE 分支          谱–GW演化分支
        │                          │
节点/边/图统计窗口表示       18维理论核心
        │                     +16维绝对谱变化
        ▼                     +32维SGW序列表示
一层BiGRU+时间注意力               │
        │                          │
        └─────────────┬────────────┘
                      ▼
              神经/理论/融合多头分类
                      ▼
                  二分类预测
```

数学形式：

\[
\mathcal G
\longrightarrow
\mathcal U
\longrightarrow
\left[
H_{\mathrm{neural}}(\mathcal U);
H_{\mathrm{theory}}(\mathcal U)
\right]
\longrightarrow
\widehat y.
\]

其中：

\[
\mathcal G=\{G^{(1)},\ldots,G^{(M)}\},
\qquad
\mathcal U=\{U^{(1)},\ldots,U^{(M)}\}.
\]

---

# 4. 输入与语义约束

每个窗口：

\[
G_b^{(m)}=(A_b^{(m)},C_b^{(m)},I_b^{(m)}),
\]

其中：

- \(A_b^{(m)}\)：带符号邻接矩阵；
- \(C_b^{(m)}\)：社区编号；
- \(I_b^{(m)}\)：稳定节点 ID 或节点名称；
- 不同样本和窗口允许具有不同 \(M_b\) 与 \(N_b^{(m)}\)。

边存在统一定义：

\[
|A_{ij}^{(m)}|>\tau_{\mathrm{edge}}.
\]

阈值必须在特征工程、选择器、拉普拉斯、GW和统计模块中保持一致。

## 4.1 节点名称

节点名称只用于对齐：

\[
\pi_m(i)=\operatorname{Match}(I_i^{(m)},I^{(m-1)}).
\]

默认：

```yaml
use_node_identity_embedding: false
```

任务专用实验可以单独开启身份 embedding，但不能把该结果作为通用模型的主要结论。

## 4.2 社区编号

社区编号用于：

- \(\mathbf 1(c_i=c_j)\)；
- 社区大小；
- 社区内外正负连接强度；
- 社区内外密度；
- 社区覆盖的硬选择。

默认：

```yaml
use_raw_community_embedding: false
```

---

# 5. 相邻窗口对齐与差分

绝对度：

\[
d_i^{(m)}=\sum_j|A_{ij}^{(m)}|.
\]

正负强度：

\[
d_{i,+}^{(m)}=\sum_j\max(A_{ij}^{(m)},0),
\]

\[
d_{i,-}^{(m)}=\sum_j\max(-A_{ij}^{(m)},0).
\]

节点在相邻窗口均存在时：

\[
\Delta d_i^{(m)}
=d_i^{(m)}-d_{\pi_m(i)}^{(m-1)}.
\]

两个端点都可对齐时：

\[
\Delta A_{ij}^{(m)}
=A_{ij}^{(m)}-A_{\pi_m(i)\pi_m(j)}^{(m-1)}.
\]

无法匹配时，差分 mask 设为 0，不得用数值 0 冒充真实差分。第一个窗口的差分均无效。

---

# 6. 关键图提取器

## 6.1 节点特征

\[
x_{\mathrm{extract},i}^{(m)}=
\left[
\begin{array}{c}
d_i,d_{i,+},d_{i,-},\Delta d_i,
\overline{|\Delta A|}_i,s_{c,i},\\
w_{\mathrm{intra},+,i},w_{\mathrm{intra},-,i},
 w_{\mathrm{inter},+,i},w_{\mathrm{inter},-,i},\\
\rho_{c,+,i},\rho_{c,-,i},\operatorname{cc}_i
\end{array}
\right].
\]

其中：

\[
\overline{|\Delta A|}_i=
\frac{\sum_jM_{\Delta A,ij}|\Delta A_{ij}|}
{\sum_jM_{\Delta A,ij}+\varepsilon}.
\]

## 6.2 STSE式节点评分

\[
h_i^{(0)}=W_v\operatorname{LN}(x_i)+b_v,
\]

\[
h_i^{(1)}=\operatorname{LN}\left(h_i^{(0)}+FFN_v(h_i^{(0)})\right),
\]

\[
p_i=\sigma(w_v^\top h_i^{(1)}+b_{v,s}).
\]

推荐隐藏维度 64，一层残差块。

## 6.3 边特征与评分

\[
e_{ij}=
\left[
A_{ij},|A_{ij}|,\Delta A_{ij},|\Delta A_{ij}|,
\mathbf 1(c_i=c_j),h_i^{(1)}+h_j^{(1)},|h_i^{(1)}-h_j^{(1)}|
\right].
\]

\[
u_{ij}^{(0)}=W_e\operatorname{LN}(e_{ij})+b_e,
\]

\[
u_{ij}^{(1)}=\operatorname{LN}\left(u_{ij}^{(0)}+FFN_e(u_{ij}^{(0)})\right),
\]

\[
p_{ij}=\sigma(w_e^\top u_{ij}^{(1)}+b_{e,s}).
\]

推荐隐藏维度 32，一层残差块。

---

# 7. 社区感知硬选择

目标节点数：

\[
K_v^{(m)}=\max\left(K_{v,\min},\left\lceil\rho_vN_m\right\rceil\right).
\]

每个非空社区优先选择一个高分节点，再按全局节点分数补足预算。

在已选节点之间定义联合边分数：

\[
q_{ij}=p_{ij}\sqrt{p_ip_j}.
\]

目标边数：

\[
K_e^{(m)}=\max\left(K_{e,\min},
\left\lceil\rho_e|E_{\mathrm{candidate}}^{(m)}|\right\rceil\right).
\]

选择 \(q_{ij}\) 最大的候选边。最终有效节点为所有保留边的端点：

\[
V_U^{(m)}=\left\{i:\sum_jm_{e,ij}^{H,(m)}>0\right\}.
\]

孤立节点删除。训练中不强制裁剪最大连通分量，硬关键并图允许具有多个连通分量。

第一版目标预算：

\[
\rho_v^\star=0.50,
\qquad
\rho_e^\star=0.30.
\]

---

# 8. STE硬选择

\[
m_v^{ST}=p_v+\operatorname{stopgrad}(m_v^H-p_v),
\]

\[
m_e^{ST}=p_e+\operatorname{stopgrad}(m_e^H-p_e).
\]

前向为二值硬 mask，反向使用代理梯度。硬邻接矩阵：

\[
\boxed{
A_{U,ij}^{(m)}
=A_{ij}^{(m)}m_{v,i}^{ST,(m)}m_{v,j}^{ST,(m)}m_{e,ij}^{ST,(m)}
}
\]

保留边必须使用原始带符号权重；不创建新边，不产生半保留边。

STE 是代理优化，不能声称其必然找到离散最优硬关键图。

---

# 9. 硬图特征重计算

所有分类特征必须在硬图上重新计算：

\[
X_{\mathrm{class},U}^{(m)}
=\Phi_{\mathrm{class}}(U^{(m-1)},U^{(m)}).
\]

禁止：

- 从完整图特征直接切片；
- 使用完整图度或社区统计；
- 把软分数作为最终分类输入；
- 自动补入未选择的诱导边。

---

# 10. 增强型 Hard-STSE 节点分支

硬图节点特征：

\[
f_{U,i}^{(m)}=
\left[
\begin{array}{c}
d_{U,i},d_{U,+,i},d_{U,-,i},\Delta d_{U,i},M_{\Delta d,U,i},\\
\overline{|\Delta A_U|}_i,s_{U,c,i},
 w_{U,\mathrm{intra},+,i},w_{U,\mathrm{intra},-,i},\\
w_{U,\mathrm{inter},+,i},w_{U,\mathrm{inter},-,i},
\operatorname{cc}_{U,i},b_{U,i}
\end{array}
\right].
\]

节点出生指标：

\[
b_{U,i}^{(m)}=\mathbf1(i\in V_U^{(m)},i\notin V_U^{(m-1)}).
\]

采用两层残差 FFN：

\[
h_{U,i}^{(0)}=W_0\operatorname{LN}(f_{U,i})+b_0,
\]

\[
h_{U,i}^{(1)}=\operatorname{LN}(h_{U,i}^{(0)}+FFN_1(h_{U,i}^{(0)})),
\]

\[
h_{U,i}^{(2)}=\operatorname{LN}(h_{U,i}^{(1)}+FFN_2(h_{U,i}^{(1)})).
\]

建议隐藏维度 96。

节点池化：

\[
g_{\mathrm{node}}^{(m)}=
[\operatorname{Mean}(h),\operatorname{Std}(h),\operatorname{Max}(h),\operatorname{AttnPool}(h)].
\]

注意力：

\[
\alpha_i=\operatorname{softmax}_i(w_a^\top\tanh(W_ah_i)),
\qquad
g_{\mathrm{attn}}=\sum_i\alpha_ih_i.
\]

---

# 11. 硬图边级分支

边特征：

\[
r_{U,ij}^{(m)}=
[A_{U,ij},|A_{U,ij}|,\Delta A_{U,ij},|\Delta A_{U,ij}|,
M_{\Delta A,U,ij},\mathbf1(c_i=c_j),b_{U,ij}].
\]

边出生：

\[
b_{U,ij}^{(m)}=\mathbf1((i,j)\in E_U^{(m)},(i,j)\notin E_U^{(m-1)}).
\]

边残差 MLP：

\[
v_{ij}^{(0)}=W_r\operatorname{LN}(r_{U,ij})+b_r,
\]

\[
v_{ij}^{(1)}=\operatorname{LN}(v_{ij}^{(0)}+FFN_r(v_{ij}^{(0)})).
\]

建议隐藏维度 64。

边池化：

\[
g_{\mathrm{edge}}^{(m)}=
[\operatorname{Mean}(v),\operatorname{Std}(v),\operatorname{Max}(v),\operatorname{AttnPool}(v)].
\]

---

# 12. 图级统计与窗口融合

图级统计：

\[
g_{\mathrm{stat}}^{(m)}=
[|V_U|,|E_U|,\rho_U,C_U,\mathcal Q_U,K_U,
\mu_{A,+},\mu_{A,-},\sigma_{|A|},\mu_{|\Delta A|},
 r_{v,b},r_{v,d},r_{e,b},r_{e,d}].
\]

其中包括节点和边的出生、消失比例。

窗口原始表示：

\[
g_{\mathrm{raw}}^{(m)}=
[g_{\mathrm{node}}^{(m)};g_{\mathrm{edge}}^{(m)};g_{\mathrm{stat}}^{(m)}].
\]

窗口编码：

\[
z_U^{(m)}=MLP_{\mathrm{window}}(g_{\mathrm{raw}}^{(m)})\in\mathbb R^{128}.
\]

推荐 MLP：输入 \(\rightarrow256\rightarrow128\)，dropout 0.20。

---

# 13. 硬关键图时间编码

显式加入窗口变化：

\[
\Delta z_U^{(m)}=z_U^{(m)}-z_U^{(m-1)},
\]

\[
\widetilde z_U^{(m)}=[z_U^{(m)};\Delta z_U^{(m)};|\Delta z_U^{(m)}|].
\]

线性投影到 128 维后输入一层 BiGRU：

\[
H_U^{(1:M)}=\operatorname{BiGRU}(\widetilde z_U^{(1:M)}).
\]

推荐双向每方向 64 维。

时间池化：

\[
\beta_m=\operatorname{softmax}_m(w_t^\top\tanh(W_tH_U^{(m)})),
\]

\[
h_{\mathrm{attn}}=\sum_m\beta_mH_U^{(m)}.
\]

最终神经表示：

\[
\boxed{
H_{\mathrm{neural}}
=MLP_N[h_{\mathrm{attn}};h_{\mathrm{mean}};h_{\mathrm{max}}]
\in\mathbb R^{192}
}
\]

padding 窗口不得参与 BiGRU 和池化。

---

# 14. 硬关键图谱–GW分支

## 14.1 带符号正则化拉普拉斯

\[
D_{U,ii}^{(m)}=\sum_j|A_{U,ij}^{(m)}|,
\]

\[
\mathcal L_{\eta,U}^{(m)}
=I-(D_U^{(m)}+\eta I)^{-1/2}A_U^{(m)}(D_U^{(m)}+\eta I)^{-1/2}.
\]

## 14.2 谱状态

\[
\nu_U^{(m)}=\frac1{N_U^{(m)}}\sum_i\delta_{\lambda_i^{(m)}}.
\]

16 维谱分位：

\[
Q_U^{(m)}=[F_{\nu_U^{(m)}}^{-1}(q_1),\ldots,F_{\nu_U^{(m)}}^{-1}(q_{16})].
\]

## 14.3 热核与扩散距离

\[
K_{t,U}^{(m)}=\exp(-t\mathcal L_{\eta,U}^{(m)}),
\]

\[
d_{t,U}^{(m)}(i,j)=\|K_{t,U}^{(m)}(i,:)-K_{t,U}^{(m)}(j,:)\|_2.
\]

第一版采用统一节点测度：

\[
\mu_{U,i}^{(m)}=\frac1{N_U^{(m)}}.
\]

## 14.4 相邻窗口演化

\[
\Delta Q_U^{(m)}=Q_U^{(m+1)}-Q_U^{(m)},
\]

\[
\delta_{\mathrm{spec},U}^{(m)}=W_1(\nu_U^{(m)},\nu_U^{(m+1)}),
\]

\[
\delta_{\mathrm{GW},U}^{(m)}
=d_{\mathrm{GW}}(\mathsf X_U^{(m)},\mathsf X_U^{(m+1)}).
\]

GW耦合：

\[
\pi_U^{(m)}\in\Pi(\mu_U^{(m)},\mu_U^{(m+1)})
\]

提供不同节点数硬图之间的结构角色软对应。

## 14.5 理论核心与增强表示

每个转移：

\[
\Gamma_{\mathrm{SGW},U}^{(m)}
=[\Delta Q_U^{(m)};\delta_{\mathrm{spec},U}^{(m)};\delta_{\mathrm{GW},U}^{(m)}]
\in\mathbb R^{18}.
\]

理论核心：

\[
H_{\mathrm{core}}=\operatorname{MeanMask}_m\Gamma_{\mathrm{SGW},U}^{(m)}\in\mathbb R^{18}.
\]

绝对谱变化：

\[
H_{\mathrm{variation}}=\operatorname{MeanMask}_m|\Delta Q_U^{(m)}|\in\mathbb R^{16}.
\]

固定分类表示：

\[
H_{\mathrm{SGW,fixed}}=[H_{\mathrm{core}};H_{\mathrm{variation}}]\in\mathbb R^{34}.
\]

将转移序列输入小型单向 GRU：

\[
h_{\mathrm{SGW,seq}}=GRU_{\mathrm{SGW}}(\Gamma^{(1:M-1)})\in\mathbb R^{32}.
\]

最终理论表示：

\[
H_{\mathrm{theory}}=[H_{\mathrm{SGW,fixed}};h_{\mathrm{SGW,seq}}]\in\mathbb R^{66}.
\]

理论直接保证对象仅为 18 维 \(H_{\mathrm{core}}\)；其余维度是经验增强。

---

# 15. SGW梯度策略

精确特征分解和 GW 在硬选择下可能产生不稳定梯度。默认模式：

1. 精确硬图 SGW 特征参与最终分类；
2. 在硬选择器边界处对精确 SGW 特征执行 `detach`；
3. SGW-GRU、理论辅助头和融合分类头正常训练；
4. 提取器通过 Hard-STSE 神经分支的分类损失接收 STE 梯度；
5. 提取器通过可微拉普拉斯保真和 GW 恒等耦合代理获得理论约束。

训练理论代理时使用原节点支撑上的 padded 硬邻接矩阵：

\[
A_{U,\mathrm{pad}}^{(m)}\in\mathbb R^{N_m\times N_m}.
\]

拉普拉斯代理：

\[
\mathcal L_{\mathrm{Lap}}
=\frac1M\sum_m
\frac{\|\mathcal L_\eta(A^{(m)})-\mathcal L_\eta(A_{U,\mathrm{pad}}^{(m)})\|_F^2}{N_m^2}.
\]

精确完整图到裁剪硬图的 GW 距离主要用于验证和理论诊断。

---

# 16. 多头分类

神经头：

\[
z_N=f_N(H_{\mathrm{neural}}).
\]

理论头：

\[
z_T=f_T(\widetilde H_{\mathrm{SGW,fixed}}).
\]

融合：

\[
H_{\mathrm{final}}=[H_{\mathrm{neural}};\widetilde H_{\mathrm{theory}}]\in\mathbb R^{258}.
\]

\[
z_F=f_F(H_{\mathrm{final}}).
\]

融合分类器：

```text
258 → 128 → 64 → 2
```

分类损失：

\[
\boxed{
\mathcal L_{\mathrm{cls}}
=CE(z_F,y)+\alpha CE(z_N,y)+\beta CE(z_T,y)
}
\]

第一版：

\[
\alpha=0.3,\qquad\beta=0.3.
\]

最终预测使用融合头 \(z_F\)。标准化参数只能在训练集拟合。

---

# 17. 总损失

\[
\boxed{
\mathcal L
=\mathcal L_{\mathrm{cls}}
+\lambda_v\mathcal L_v
+\lambda_e\mathcal L_e
+\lambda_{\mathrm{Lap}}\mathcal L_{\mathrm{Lap}}
+\lambda_{\mathrm{GW}}\mathcal L_{\mathrm{GW,proxy}}
}
\]

硬 Top-K 已固定实际保留数量，因此预算损失用于软分数校准：

\[
\mathcal L_v=|\operatorname{Mean}(p_v)-\rho_v|,
\]

\[
\mathcal L_e=|\operatorname{Mean}(p_e)-\rho_e|.
\]

---

# 18. 课程式单模型训练

整个过程始终是同一个模型，不引入教师或学生。

## 阶段一：高保留率启动（建议 epoch 1–10）

\[
\rho_v=0.90,\qquad\rho_e=0.80.
\]

主要优化分类损失，理论和预算权重设为 0 或极小值。

## 阶段二：逐步压缩（建议 epoch 11–30）

线性退火到：

\[
\rho_v^\star=0.50,\qquad\rho_e^\star=0.30.
\]

同时逐步提高预算校准权重。

## 阶段三：联合稳定（epoch 31以后）

固定预算，逐步加入理论代理：

\[
\lambda_{\mathrm{Lap}}\in[0.02,0.10],
\qquad
\lambda_{\mathrm{GW}}\in[0.01,0.05].
\]

这些是验证集搜索范围，不是理论常数。

---

# 19. 推荐初始配置

```yaml
model:
  name: hard_stse_temporal_sgw
  num_classes: 2
  use_teacher: false
  use_student: false
  use_gnn: false
  use_transformer: false

data:
  batching: list
  use_coordinates: false
  use_node_name_for_alignment: true
  use_node_identity_embedding: false
  use_raw_community_embedding: false
  use_community_structure_features: true
  edge_presence_threshold_source: sample_metadata
  edge_presence_threshold_key: global_threshold

extractor:
  node_hidden_dim: 64
  node_residual_blocks: 1
  edge_hidden_dim: 32
  edge_residual_blocks: 1
  hard_selection: ste_topk
  community_coverage: true
  start_node_ratio: 0.90
  start_edge_ratio: 0.80
  target_node_ratio: 0.50
  target_edge_ratio: 0.30
  remove_isolated_nodes: true
  force_largest_connected_component: false

hard_stse:
  node_hidden_dim: 96
  node_residual_blocks: 2
  node_pooling: [mean, std, max, attention]
  edge_hidden_dim: 64
  edge_residual_blocks: 1
  edge_pooling: [mean, std, max, attention]
  use_graph_statistics: true
  window_hidden_dim: 256
  window_output_dim: 128
  dropout: 0.20

temporal:
  type: bigru
  hidden_dim_per_direction: 64
  num_layers: 1
  bidirectional: true
  pooling: [attention, mean, max]
  output_dim: 192

signed_laplacian:
  eta: 1.0e-3
  epsilon: 1.0e-12

heat_kernel:
  diffusion_time: 1.0

sgw:
  num_spectral_quantiles: 16
  quantile_min: 0.05
  quantile_max: 0.95
  core_dim: 18
  fixed_classification_dim: 34
  sequence_encoder: gru
  sequence_hidden_dim: 32
  enhanced_dim: 66
  exact_feature_detach_from_selector: true
  node_measure: uniform
  gw_solver: entropic_gw
  warm_start_coupling: true

classifier:
  neural_dim: 192
  theory_dim: 66
  fusion_input_dim: 258
  fusion_hidden_dims: [128, 64]
  dropout: 0.20
  neural_auxiliary_head: true
  theory_auxiliary_head: true

loss:
  fusion_ce_weight: 1.0
  neural_aux_ce_weight: 0.3
  theory_aux_ce_weight: 0.3
  laplacian_weight_max: 0.05
  gw_proxy_weight_max: 0.02
```

---

# 20. 实验设计

## 20.1 目标

实验回答：

1. 增强型 STSE 是否能在完整图上建立有效分类能力？
2. 学习硬图是否优于同预算随机硬图？
3. 硬图压缩后是否保持或提升分类性能？
4. 谱–GW分支是否提供独立增益？

---

# 21. 实现正确性检查

## 21.1 硬图

```python
assert hard_node_mask.dtype == torch.bool
assert hard_edge_mask.dtype == torch.bool
assert selected_edges_use_original_signed_weights
assert no_new_edges_are_created
assert isolated_nodes_are_removed
assert hard_graph_num_nodes <= original_num_nodes
assert hard_graph_num_edges <= original_num_edges
```

## 21.2 梯度

```python
assert node_scorer_grad_norm > 0
assert edge_scorer_grad_norm > 0
assert hard_stse_grad_norm > 0
assert temporal_encoder_grad_norm > 0
assert fusion_classifier_grad_norm > 0
```

默认模式不要求精确 GW 对选择器产生梯度。

## 21.3 Mask与变长

```python
assert padding_does_not_change_output
assert node_reordering_with_same_ids_does_not_change_deltas
assert invalid_delta_is_never_replaced_by_real_zero
assert padded_windows_do_not_affect_bigru
assert empty_hard_window_is_masked
```

## 21.4 谱–GW

相同图应满足：

\[
\delta_{\mathrm{spec}}(U,U)\approx0,
\qquad
\delta_{\mathrm{GW}}(U,U)\approx0.
\]

相同窗口序列应满足：

\[
\Delta Q=0,
\qquad
H_{\mathrm{variation}}=0.
\]

---

# 22. 最小过拟合实验

固定 16 个训练样本，两类均包含。

## 22.1 完整图模式

\[
\rho_v=1,\qquad\rho_e=1.
\]

```yaml
dropout: 0.0
weight_decay: 0.0
laplacian_weight: 0.0
gw_proxy_weight: 0.0
early_stopping: false
max_epochs: 200
```

目标：

\[
\text{Train Accuracy}\ge95\%.
\]

## 22.2 硬选择模式

\[
\rho_v=0.8,\qquad\rho_e=0.7.
\]

若完整图可过拟合而硬选择不能，检查 STE、硬图空窗口、预算、社区覆盖和硬图特征重计算。

---

# 23. 主开发实验

## 23.1 划分

严格读取：

- `sample_index.csv`；
- `splits.csv`。

开发阶段只用固定训练/验证划分，每个模型运行 3 个随机种子，不查看测试集。

```yaml
seeds: [42, 43, 44]
```

## 23.2 训练

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

checkpoint 主指标为验证集 Balanced Accuracy，次级指标为 AUROC。

---

# 24. 最小必要对照

## M0：Full-STSE-Temporal

```text
完整图 → 增强型STSE → BiGRU → 神经分类
```

用于验证基础分类编码器。

## M1：Random-Hard-STSE-Temporal

```text
随机同预算硬图 → Hard-STSE → BiGRU → 分类
```

用于控制任意压缩带来的影响。

## M2：Learned-Hard-STSE-Temporal

```text
学习硬选择 → Hard-STSE → BiGRU → 分类
```

用于验证学习提取器是否优于随机硬图。

## M3：Learned-Hard-STSE-Temporal-SGW

```text
学习硬选择
→ Hard-STSE + BiGRU
→ 34维固定SGW + SGW-GRU
→ 多头融合分类
```

用于验证谱–GW分支的增量价值。

---

# 25. 可选消融

时间允许时：

- SGW-only；
- 去掉 SGW-GRU；
- 仅34维固定SGW；
- 启用节点身份适配器（任务专用，不作通用主结论）；
- 特征拼接融合与决策级融合对比。

---

# 26. 评价指标

主指标：Balanced Accuracy。

次要指标：

- AUROC；
- Accuracy；
- F1；
- Sensitivity；
- Specificity。

同时记录：

- 参数量；
- 单 epoch 时间；
- 峰值显存；
- 节点/边保留率；
- 空硬图窗口比例；
- 连通分量数；
- 完整图到硬图谱误差；
- 完整图到硬图 GW 误差。

主结果表：

| 模型 | Train BA | Val BA | Val AUC | 节点保留率 | 边保留率 | 时间/epoch |
|---|---:|---:|---:|---:|---:|---:|
| M0 Full-STSE-Temporal |  |  |  | 100% | 100% |  |
| M1 Random-Hard-STSE-Temporal |  |  |  |  |  |  |
| M2 Learned-Hard-STSE-Temporal |  |  |  |  |  |  |
| M3 Full Model |  |  |  |  |  |  |

---

# 27. 判定规则

## 27.1 基础分类器

若 M0 在完整训练集上仍接近随机，暂停提取器训练，检查：

- 去坐标后的结构特征是否具有足够样本差异；
- 特征标准化；
- 时间编码；
- 分类头和损失；
- 当前实现与原 STSE 的差异。

## 27.2 提取器

应满足：

\[
\operatorname{Val}(M2)>\operatorname{Val}(M1).
\]

否则说明提取器未学到优于随机预算的判别结构。

## 27.3 SGW

理想情况：

\[
\operatorname{Val}(M3)>\operatorname{Val}(M2).
\]

若 SGW-only 高于随机但融合降低性能，可降低理论分支权重或改用决策级融合。若 SGW-only 也接近随机，应检查谱分位、扩散时间、节点测度、GW求解精度及完整图类别间隔假设。

性能优先时，不强制要求 SGW 提升最终分类，但必须报告其独立贡献。

---

# 28. 最终测试

只有在以下内容冻结后才允许查看测试集：

- 模型结构；
- 节点/边预算；
- 损失权重；
- 谱分位点；
- 扩散时间；
- GW参数；
- checkpoint规则。

若存在预定义多折划分，在结构冻结后完成全部折并报告均值和标准差。

---

# 29. 理论边界

当前理论可以支持：若完整图存在谱–GW类别演化间隔，且完整图到硬关键图的类别内提取误差足够小，则：

\[
W_1(P_a^{U,\mathrm{core}},P_b^{U,\mathrm{core}})
\ge
\Delta_{ab}-\eta_a-\eta_b.
\]

理论直接作用于 18 维 \(H_{\mathrm{core}}\)。不能声称：

- STE 必然找到最优硬图；
- 34维和66维增强表示全部继承同一下界；
- BiGRU、融合MLP或输出概率继承同一下界；
- 理论保证高准确率；
- 关键图必然对应真实疾病机制。

---

# 30. 校验结果：已修正的错漏

## 30.1 最终分类对象

所有神经和谱–GW特征均来自硬关键图。软分数只用于选择器优化。

## 30.2 真正减少节点和边

硬边选择后删除孤立节点。训练张量可以 padding，但谱、GW和分类的数学对象必须使用有效硬节点裁剪结果。

## 30.3 泛化性

主模型不使用 ROI 坐标、邻居坐标、节点名称 embedding 或原始社区编号 embedding。

## 30.4 跨窗口对应

- 节点名称用于可识别节点的显式对齐；
- GW耦合用于不同节点数硬图的结构角色软对应。

模型每窗口输出一张硬关键并图，不显式维护多个子图轨迹。

## 30.5 子图演化分类

神经分支使用节点/边出生消失、硬图差分、窗口表示差分和 BiGRU；理论分支直接使用 \(\Delta Q\)、谱Wasserstein和GW演化。

## 30.6 精确GW梯度

精确 SGW 参与最终分类，但默认不直接向选择器反传；提取器由神经STE梯度和理论代理损失训练。

## 30.7 预算损失

硬 Top-K 已固定数量，预算损失改为校准软评分均值，硬预算通过课程式调度控制。

## 30.8 STSE跨窗口聚合

原论文“仅STSE”的样本级窗口聚合没有明确说明；本模型明确使用窗口差分、一层BiGRU和时间注意力/均值/最大值池化。

---

# 31. 仍需验证集确定的参数

- 目标节点与边保留率；
- \(\eta\)；
- 扩散时间 \(t\)；
- 谱分位点数量；
- GW正则与迭代次数；
- 辅助头权重；
- 谱/GW代理权重；
- 是否启用 SGW-GRU；
- 特征级或决策级融合。

---

# 32. 最终总结

\[
\boxed{\text{Hard-STSE-Temporal-SGW}}
\]

职责划分：

- **硬提取器**：真正删除冗余节点和边；
- **增强型 Hard-STSE**：读取硬图节点、边与社区结构；
- **BiGRU**：学习硬图随窗口变化的判别时序；
- **谱–GW分支**：提供与理论对齐的全局结构演化表示；
- **多头融合**：以分类性能为优先，同时保证神经和理论分支都获得监督。

最终数据流：

```text
完整动态图
→ 通用结构与差分特征
→ STE硬节点/边选择
→ 硬关键图序列
→ Hard-STSE节点/边/图统计编码
→ BiGRU时间建模
→ 硬图谱–GW演化编码
→ 神经/理论/融合多头分类
→ 最终预测
```
