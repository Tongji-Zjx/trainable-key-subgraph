# Dual-STSE-HardSGW：双通道完整图 STSE 与硬关键图谱–GW 融合模型设计

## 文档状态

- **版本**：V1.0
- **模型名称**：Dual-STSE-HardSGW
- **任务类型**：动态图序列二分类
- **设计目标**：
  1. 以分类性能为第一目标；
  2. 保留已经验证有效的完整图 STSE 分类能力；
  3. 引入硬关键图提取与谱–GW 演化表示作为补充通道；
  4. 不使用教师–学生结构；
  5. 不使用学习型时间依赖编码器；
  6. 通过独立辅助头和严格对照实验验证两个通道是否真正互补。

---

# 1. 核心设计结论

本模型采用双通道结构：

\[
\boxed{
\begin{aligned}
\text{通道一：完整图 STSE}
&\rightarrow
\text{提供稳定的基础分类能力};\\
\text{通道二：硬关键图谱–GW}
&\rightarrow
\text{提供压缩后的结构演化信息}.
\end{aligned}
}
\]

最终分类器融合两个通道的表示：

\[
\boxed{
\mathcal G
\longrightarrow
\begin{cases}
H_{\mathrm{STSE}}(\mathcal G),\\[2mm]
H_{\mathrm{SGW}}(\mathcal U(\mathcal G))
\end{cases}
\longrightarrow
\widehat y
}
\]

其中：

- \(\mathcal G\) 为完整动态图序列；
- \(\mathcal U(\mathcal G)\) 为从完整图中提取的硬关键图序列；
- \(H_{\mathrm{STSE}}\) 为完整图 STSE 表示；
- \(H_{\mathrm{SGW}}\) 为硬关键图谱–GW 演化表示。

---

# 2. 最重要的实现原则：通道一直接复用已实现的无空间坐标 STSE

## 2.1 必须直接复用现有实现

通道一应尽可能直接复用当前已经实现、经过实验验证的无空间坐标 STSE。

已知当前实现：

- 不使用节点空间坐标；
- 不使用邻居空间坐标；
- 不使用学习型时间编码器；
- 已在实验中达到约 \(62\%\) 的准确率；
- 已证明去掉空间坐标后，分类性能未出现明显依赖性下降。

因此，通道一不应重新设计，也不应为了与通道二“统一风格”而修改。

\[
\boxed{
\text{通道一的首要原则是保持现有实现、参数和训练行为尽可能不变。}
}
\]

## 2.2 不应随意修改的内容

除非独立实验已经证明必要，否则不得修改：

- 当前节点特征定义；
- 社区 embedding 的实现；
- LayerNorm 的位置；
- 线性投影维度；
- 残差 FFN 结构；
- 节点均值池化；
- 非学习型窗口聚合方式；
- 分类头结构；
- 优化器；
- 学习率；
- Dropout；
- 类别权重；
- checkpoint 选择规则；
- 训练和验证划分；
- 随机种子设置。

特别禁止在通道一中新增：

- BiGRU；
- Transformer；
- TCN；
- GNN；
- 注意力时间编码器；
- 边级神经编码器；
- 原型记忆；
- 关键图选择；
- 谱–GW 特征。

这些模块属于新模型的其他部分，不能污染已经验证过的 STSE 基线。

## 2.3 推荐的代码复用方式

建议保留现有模块：

```python
class ExistingNoCoordSTSE(nn.Module):
    ...
```

双通道模型中直接实例化：

```python
self.full_graph_stse = ExistingNoCoordSTSE(
    **validated_stse_config
)
```

而不是复制一份代码后再修改。

推荐配置：

```yaml
full_stse:
  implementation: reuse_existing_no_coord_stse
  freeze_architecture: true
  use_coordinates: false
  use_neighbor_coordinates: false
  use_learned_temporal_encoder: false
  use_validated_hyperparameters: true
```

---

# 3. 整体架构

```text
                         完整动态图序列
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
       通道一：完整图 STSE            通道二：硬关键图谱–GW
                  │                           │
     已实现 NoCoord-STSE              节点/边重要性评分
                  │                           │
       非学习型时间聚合                STE 硬选择
                  │                           │
          H_STSE                        硬关键图序列
                  │                           │
                  │                 拉普拉斯谱 + GW 演化
                  │                           │
                  ▼                           ▼
            STSE 辅助头                 SGW 辅助头
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                         特征投影与拼接
                                ▼
                           融合分类头
                                ▼
                            二分类结果
```

---

# 4. 输入定义

对第 \(b\) 个样本：

\[
\mathcal G_b
=
\left\{
G_b^{(1)},
\ldots,
G_b^{(M_b)}
\right\}.
\]

每个窗口：

\[
G_b^{(m)}
=
\left(
A_b^{(m)},
C_b^{(m)},
I_b^{(m)}
\right).
\]

其中：

- \(A_b^{(m)}\)：带符号加权邻接矩阵；
- \(C_b^{(m)}\)：社区编号；
- \(I_b^{(m)}\)：稳定节点 ID 或节点名称；
- \(M_b\)：有效窗口数；
- \(N_b^{(m)}\)：有效节点数。

模型必须支持：

- 不同样本窗口数不同；
- 不同窗口节点数不同；
- 相邻窗口节点顺序不同；
- 图中存在正边和负边；
- 使用 mask 忽略 padding。

---

# 5. 通道一：完整图无空间坐标 STSE

## 5.1 节点输入特征

通道一采用当前已实现的无坐标版本：

\[
f_i^{(m)}
=
\left[
\deg_i^{(m)};
\operatorname{emb}(c_i^{(m)});
\Delta\deg_i^{(m)}
\right].
\]

其中：

\[
\deg_i^{(m)}
=
\sum_j
|A_{ij}^{(m)}|.
\]

相邻窗口节点对齐后：

\[
\Delta\deg_i^{(m)}
=
\deg_i^{(m)}
-
\deg_{\pi_m(i)}^{(m-1)}.
\]

若当前已实现代码采用其他经过验证的差分处理，则应以现有实现为准，不在双通道模型中重新解释或更改。

## 5.2 原文式节点编码

\[
h_i^{(m)}
=
W_1
\operatorname{LayerNorm}
\left(
f_i^{(m)}
\right)
+b_1.
\]

残差 FFN：

\[
r_i^{(m)}
=
W_3
\operatorname{GELU}
\left(
W_2h_i^{(m)}+b_2
\right)
+b_3.
\]

节点输出：

\[
x_i^{(m)}
=
\operatorname{LayerNorm}
\left(
h_i^{(m)}+r_i^{(m)}
\right).
\]

窗口表示：

\[
z_{\mathrm{STSE}}^{(m)}
=
\operatorname{MeanMask}_i
x_i^{(m)}.
\]

## 5.3 非学习型时间聚合

该通道不使用任何学习型时间依赖编码器。

样本表示直接沿用现有 STSE 已验证的窗口聚合方式。

若当前实现为窗口均值，则：

\[
H_{\mathrm{STSE}}
=
\operatorname{MeanMask}_m
z_{\mathrm{STSE}}^{(m)}.
\]

若当前实现还包含固定统计量，则也应直接复用，不在此处新增 BiGRU、Transformer 或注意力时序模块。

## 5.4 STSE 独立分类头

\[
z_S
=
Classifier_S
\left(
H_{\mathrm{STSE}}
\right).
\]

该辅助头的目标是：

- 保证通道一保持原有分类能力；
- 在联合训练时避免其表示被融合梯度破坏；
- 检查双通道训练后是否仍能复现约 \(62\%\) 的单通道准确率。

---

# 6. 通道二：硬关键图提取器

## 6.1 节点结构特征

节点提取特征建议为：

\[
x_{\mathrm{extract},i}^{(m)}
=
\left[
d_i;
d_{i,+};
d_{i,-};
\Delta d_i;
\overline{|\Delta A|}_i;
s_{c,i};
w_{\mathrm{intra},+,i};
w_{\mathrm{intra},-,i};
w_{\mathrm{inter},+,i};
w_{\mathrm{inter},-,i};
\rho_{c,+,i};
\rho_{c,-,i}
\right].
\]

其中：

\[
d_{i,+}
=
\sum_j\max(A_{ij},0),
\]

\[
d_{i,-}
=
\sum_j\max(-A_{ij},0).
\]

## 6.2 节点评分

\[
p_i^{(m)}
=
\sigma
\left(
MLP_v
\left(
x_{\mathrm{extract},i}^{(m)}
\right)
\right).
\]

建议：

```yaml
node_scorer:
  hidden_dim: 64
  layers: 2
  dropout: 0.10
```

## 6.3 边特征

\[
e_{ij}^{(m)}
=
\left[
A_{ij}^{(m)};
|A_{ij}^{(m)}|;
\Delta A_{ij}^{(m)};
|\Delta A_{ij}^{(m)}|;
\mathbf 1(c_i^{(m)}=c_j^{(m)})
\right].
\]

## 6.4 边评分

\[
p_{ij}^{(m)}
=
\sigma
\left(
MLP_e
\left(
e_{ij}^{(m)}
\right)
\right).
\]

无向图中必须对称化：

\[
P_e^{(m)}
\leftarrow
\frac{
P_e^{(m)}
+
(P_e^{(m)})^\top
}{2}.
\]

---

# 7. 硬关键图选择

## 7.1 节点预算

\[
K_v^{(m)}
=
\max
\left(
K_{v,\min},
\left\lceil
\rho_vN_m
\right\rceil
\right).
\]

第一版建议：

\[
\rho_v=0.50.
\]

## 7.2 社区覆盖

每个非空社区优先选择至少一个高分节点：

\[
v_r^\star
=
\arg\max_{i\in C_r}
p_i.
\]

剩余名额由全局高分节点补足。

## 7.3 边预算

在选中节点之间定义联合分数：

\[
q_{ij}
=
p_{ij}
\sqrt{p_ip_j}.
\]

目标边数：

\[
K_e^{(m)}
=
\max
\left(
K_{e,\min},
\left\lceil
\rho_e
|E_{\mathrm{candidate}}^{(m)}|
\right\rceil
\right).
\]

第一版建议：

\[
\rho_e=0.30.
\]

## 7.4 STE 硬选择

节点 mask：

\[
m_v^{ST}
=
p_v+
\operatorname{stopgrad}
\left(
m_v^H-p_v
\right).
\]

边 mask：

\[
m_e^{ST}
=
p_e+
\operatorname{stopgrad}
\left(
m_e^H-p_e
\right).
\]

硬邻接矩阵：

\[
\boxed{
A_{U,ij}^{(m)}
=
A_{ij}^{(m)}
m_{v,i}^{ST,(m)}
m_{v,j}^{ST,(m)}
m_{e,ij}^{ST,(m)}
}
\]

前向过程中：

- 保留边使用原始符号和边权；
- 未选边严格为零；
- 不创建新边；
- 删除没有保留边的孤立节点。

---

# 8. 硬关键图谱–GW演化分支

## 8.1 带符号正则化拉普拉斯

\[
D_{U,ii}^{(m)}
=
\sum_j
|A_{U,ij}^{(m)}|.
\]

\[
\mathcal L_{\eta,U}^{(m)}
=
I
-
\left(
D_U^{(m)}+\eta I
\right)^{-1/2}
A_U^{(m)}
\left(
D_U^{(m)}+\eta I
\right)^{-1/2}.
\]

## 8.2 谱状态

定义谱经验测度：

\[
\nu_U^{(m)}
=
\frac{1}{N_U^{(m)}}
\sum_i
\delta_{\lambda_i^{(m)}}.
\]

在固定 16 个分位点上得到：

\[
Q_U^{(m)}
\in
\mathbb R^{16}.
\]

## 8.3 扩散距离

热核：

\[
K_{t,U}^{(m)}
=
\exp
\left(
-t\mathcal L_{\eta,U}^{(m)}
\right).
\]

扩散距离：

\[
d_{t,U}^{(m)}(i,j)
=
\left\|
K_{t,U}^{(m)}(i,:)
-
K_{t,U}^{(m)}(j,:)
\right\|_2.
\]

节点测度第一版使用均匀分布：

\[
\mu_{U,i}^{(m)}
=
\frac{1}{N_U^{(m)}}.
\]

## 8.4 相邻窗口演化

谱方向变化：

\[
\Delta Q_U^{(m)}
=
Q_U^{(m+1)}
-
Q_U^{(m)}.
\]

谱演化距离：

\[
\delta_{\mathrm{spec},U}^{(m)}
=
W_1
\left(
\nu_U^{(m)},
\nu_U^{(m+1)}
\right).
\]

GW 演化距离：

\[
\delta_{\mathrm{GW},U}^{(m)}
=
d_{\mathrm{GW}}
\left(
\mathsf X_U^{(m)},
\mathsf X_U^{(m+1)}
\right).
\]

转移表示：

\[
\Gamma_U^{(m)}
=
\left[
\Delta Q_U^{(m)};
\delta_{\mathrm{spec},U}^{(m)};
\delta_{\mathrm{GW},U}^{(m)}
\right]
\in\mathbb R^{18}.
\]

## 8.5 非学习型时间聚合

核心表示：

\[
H_{\mathrm{core}}
=
\operatorname{MeanMask}_m
\Gamma_U^{(m)}
\in\mathbb R^{18}.
\]

绝对谱变化：

\[
H_{\mathrm{variation}}
=
\operatorname{MeanMask}_m
|\Delta Q_U^{(m)}|
\in\mathbb R^{16}.
\]

最终 SGW 表示：

\[
\boxed{
H_{\mathrm{SGW}}
=
\left[
H_{\mathrm{core}};
H_{\mathrm{variation}}
\right]
\in\mathbb R^{34}.
}
\]

本通道不使用：

- SGW-GRU；
- Transformer；
- BiGRU；
- TCN；
- 学习型时间注意力。

---

# 9. 双通道融合

## 9.1 标准化

STSE 表示：

\[
\widetilde H_{\mathrm{STSE}}
=
\operatorname{LayerNorm}
\left(
H_{\mathrm{STSE}}
\right).
\]

SGW 表示使用训练集拟合的标准化器：

\[
\widetilde H_{\mathrm{SGW}}
=
\frac{
H_{\mathrm{SGW}}
-
\widehat\mu_{\mathrm{train}}
}{
\widehat\sigma_{\mathrm{train}}
+
\varepsilon
}.
\]

验证集和测试集不得重新拟合标准化参数。

## 9.2 分支投影

\[
h_S
=
MLP_S
\left(
\widetilde H_{\mathrm{STSE}}
\right)
\in\mathbb R^{64}.
\]

\[
h_G
=
MLP_G
\left(
\widetilde H_{\mathrm{SGW}}
\right)
\in\mathbb R^{64}.
\]

两个通道投影到相同维度，避免 STSE 因维度更大而天然支配融合。

## 9.3 拼接融合

\[
H_{\mathrm{fusion}}
=
\left[
h_S;
h_G
\right]
\in\mathbb R^{128}.
\]

融合分类头：

```text
128
→ Linear(64)
→ GELU
→ Dropout(0.20)
→ Linear(2)
```

第一版使用直接拼接，不使用动态门控。

---

# 10. 三个分类头

## 10.1 STSE 辅助头

\[
z_S
=
Classifier_S
\left(
H_{\mathrm{STSE}}
\right).
\]

目标：

- 保持已验证的 STSE 性能；
- 检查联合训练是否破坏基线。

## 10.2 SGW 辅助头

\[
z_G
=
Classifier_G
\left(
\widetilde H_{\mathrm{SGW}}
\right).
\]

目标：

- 强制关键图–SGW 通道具备独立分类能力；
- 防止融合模型完全忽略 SGW；
- 为提取器提供分类监督。

## 10.3 融合头

\[
z_F
=
Classifier_F
\left(
H_{\mathrm{fusion}}
\right).
\]

最终预测使用融合头。

---

# 11. 损失函数

分类损失：

\[
\boxed{
\mathcal L_{\mathrm{cls}}
=
CE(z_F,y)
+
\alpha CE(z_S,y)
+
\beta CE(z_G,y)
}
\]

第一版建议：

\[
\alpha=0.3,
\qquad
\beta=0.5.
\]

总损失：

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{cls}}
+
\lambda_v\mathcal L_v
+
\lambda_e\mathcal L_e
+
\lambda_{\mathrm{Lap}}\mathcal L_{\mathrm{Lap}}
+
\lambda_{\mathrm{GW}}\mathcal L_{\mathrm{GW,proxy}}
}
\]

其中：

- \(\mathcal L_v\)：节点预算校准；
- \(\mathcal L_e\)：边预算校准；
- \(\mathcal L_{\mathrm{Lap}}\)：完整图到硬图的谱保真代理；
- \(\mathcal L_{\mathrm{GW,proxy}}\)：完整图到硬图的 GW 几何保真代理。

分类性能始终是主目标，理论约束权重不应过大。

---

# 12. 推荐训练策略

## 阶段一：复用并确认 STSE 基线

单独运行现有无坐标 STSE。

要求：

- 使用原有代码和原有配置；
- 重新确认约 \(62\%\) 的准确率；
- 保存最佳 checkpoint；
- 记录每个随机种子的结果。

这一阶段不能修改 STSE 结构。

## 阶段二：冻结 STSE，训练 Hard-SGW 通道

冻结：

- STSE 编码器；
- STSE 辅助头。

训练：

- 节点评分器；
- 边评分器；
- SGW 辅助头；
- SGW 投影层；
- 理论代理损失。

目标是让 Hard-SGW 通道具备独立分类能力。

## 阶段三：联合融合微调

载入两边的 checkpoint。

学习率建议：

```yaml
optimizer_groups:
  stse:
    learning_rate: 0.0001
  hard_sgw:
    learning_rate: 0.001
  fusion:
    learning_rate: 0.001
```

联合优化总损失。

STSE 使用较小学习率，避免已经验证有效的表示被破坏。

---

# 13. 可选防忽略策略

如果融合模型完全忽略 SGW 通道，可以按顺序尝试：

1. 提高 \(\beta\)；
2. 延长阶段二独立训练；
3. 冻结 STSE 更长时间；
4. 对 STSE 投影使用小概率分支 Dropout；
5. 使用决策级平均或验证集学习的固定权重融合。

可选分支 Dropout：

```yaml
fusion:
  stse_branch_dropout_probability: 0.10
```

第一版不默认启用。

---

# 14. 实验设计

## 14.1 实验目标

需要回答：

1. 现有无坐标 STSE 是否能稳定复现约 \(62\%\) 的准确率？
2. 完整图 SGW 是否存在可用的分类信号？
3. 学习型硬关键图是否优于同预算随机硬图？
4. Hard-SGW 通道是否具有独立分类能力？
5. 双通道融合是否优于 STSE 单通道？
6. 融合性能提升是否来自关键图学习，而不是额外参数量？

---

# 15. 必做模型对照

## D0：Existing-NoCoord-STSE

```text
完整图
→ 已实现无坐标 STSE
→ 非学习型时间聚合
→ STSE 分类头
```

目的：

- 复现已知基线；
- 作为所有后续实验的主要比较对象。

## D1：Full-Graph-SGW

```text
完整图
→ 谱–GW 演化表示
→ SGW 分类头
```

不进行关键图提取。

目的：

- 判断完整图谱–GW 本身是否有分类信号；
- 估计压缩前的理论表示上限。

## D2：Random-Hard-SGW

```text
随机选择同等节点和边预算
→ 硬图谱–GW
→ SGW 分类头
```

目的：

- 建立同预算随机压缩基线；
- 判断任意压缩是否已经足够。

## D3：Learned-Hard-SGW

```text
学习型硬节点/边选择
→ 硬图谱–GW
→ SGW 分类头
```

目的：

- 验证提取器是否学到判别结构。

关键判断：

\[
\boxed{
D3>D2
}
\]

## D4：Dual-STSE-HardSGW

```text
现有无坐标 STSE
+
学习型硬关键图谱–GW
→ 双通道融合
```

关键判断：

\[
\boxed{
D4>D0
}
\]

若 D4 不优于 D0，则不能声称 SGW 通道带来有效增益。

---

# 16. 可选消融

## A1：无 SGW 辅助头

删除：

\[
CE(z_G,y).
\]

用于验证 SGW 辅助监督是否必要。

## A2：不冻结 STSE 直接联合训练

用于验证分阶段训练是否必要。

## A3：只使用18维理论核心

比较：

\[
H_{\mathrm{core}}\in\mathbb R^{18}
\]

与：

\[
H_{\mathrm{SGW}}\in\mathbb R^{34}.
\]

## A4：不同节点和边预算

建议搜索：

\[
\rho_v\in\{0.3,0.5,0.7\},
\]

\[
\rho_e\in\{0.2,0.3,0.5\}.
\]

---

# 17. 最小过拟合测试

## 17.1 STSE 通道

应直接使用现有测试结果。

若重新测试：

```yaml
overfit:
  num_samples: 16
  dropout: 0.0
  weight_decay: 0.0
  early_stopping: false
```

要求：

\[
\text{Train Accuracy}\geq95\%.
\]

## 17.2 Hard-SGW 通道

使用高保留率：

\[
\rho_v=0.8,
\qquad
\rho_e=0.7.
\]

检查：

```python
assert node_scorer_grad_norm > 0
assert edge_scorer_grad_norm > 0
assert sgw_classifier_grad_norm > 0
```

---

# 18. 数据划分与训练规范

严格读取：

- `sample_index.csv`；
- `splits.csv`。

训练代码不得自行随机重新划分。

开发阶段建议：

```yaml
seeds:
  - 42
  - 43
  - 44
```

统一训练配置：

```yaml
optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

training:
  max_epochs: 80
  early_stopping_patience: 15
  gradient_clip_norm: 1.0

scheduler:
  name: reduce_lr_on_plateau
  factor: 0.5
  patience: 5
  min_learning_rate: 0.00001
```

checkpoint 主指标：

\[
\text{Validation Balanced Accuracy}.
\]

次级指标：

\[
\text{Validation AUROC}.
\]

---

# 19. 评价指标

主要指标：

\[
\boxed{
\text{Balanced Accuracy}
}
\]

同时报告：

- Accuracy；
- AUROC；
- F1；
- Sensitivity；
- Specificity；
- 混淆矩阵；
- 参数量；
- 单 epoch 时间；
- 峰值显存；
- 节点保留率；
- 边保留率；
- 空硬图窗口比例；
- 连通分量数量；
- 谱保真误差；
- GW 保真误差。

---

# 20. 主结果表

| 模型 | Train BA | Val BA | Val AUC | 节点保留率 | 边保留率 | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| D0 Existing-NoCoord-STSE |  |  |  | 100% | 100% |  |
| D1 Full-Graph-SGW |  |  |  | 100% | 100% |  |
| D2 Random-Hard-SGW |  |  |  |  |  |  |
| D3 Learned-Hard-SGW |  |  |  |  |  |  |
| D4 Dual-STSE-HardSGW |  |  |  |  |  |  |

---

# 21. 结果解释

## 21.1 D3 优于 D2

若：

\[
D3>D2,
\]

说明学习型提取器优于随机压缩，关键图选择具有有效性。

## 21.2 D4 优于 D0

若：

\[
D4>D0,
\]

说明硬关键图谱–GW 通道为完整图 STSE 提供了额外判别信息。

## 21.3 D4 与 D0 相近

说明：

- SGW 通道可能被融合模型忽略；
- SGW 表示可能没有额外分类信号；
- 提取器可能没有学到有效结构；
- 两个通道可能高度冗余。

此时必须检查：

- SGW-only 的 D3 性能；
- SGW 辅助头损失；
- 分支梯度；
- 投影层范数；
- 融合头对两个通道的权重敏感性。

## 21.4 D4 低于 D0

说明联合训练破坏了 STSE 或 SGW 引入噪声。

优先采取：

1. 冻结 STSE；
2. 降低 STSE 学习率；
3. 减小理论约束；
4. 改用决策级融合；
5. 只在验证集学习固定融合权重。

---

# 22. 模型解释边界

该模型可以声称：

> 完整图 STSE 通道提供稳定的图状态分类表示，硬关键图谱–GW 通道提供压缩后的全局结构演化信息，二者通过多头监督进行融合。

不能声称：

- 整个模型只依赖关键子图分类；
- 硬关键图是所有预测的唯一原因；
- 最终预测完全可由硬关键图解释；
- 融合 MLP 的输出继承 SGW 核心表示的理论类别间隔；
- 理论保证最终分类准确率。

理论直接作用于：

\[
H_{\mathrm{core}}\in\mathbb R^{18}.
\]

完整图 STSE 通道属于经验分类分支。

---

# 23. 推荐初始配置

```yaml
model:
  name: dual_stse_hard_sgw
  num_classes: 2
  use_teacher: false
  use_student: false
  use_learned_temporal_encoder: false

full_stse:
  implementation: reuse_existing_no_coord_stse
  freeze_architecture: true
  use_coordinates: false
  use_neighbor_coordinates: false
  use_node_identity_embedding: false
  use_raw_community_embedding: true
  use_validated_hyperparameters: true
  temporal_pooling: reuse_validated_nonlearned_pooling

extractor:
  node_hidden_dim: 64
  edge_hidden_dim: 32
  hard_selection: ste_topk
  community_coverage: true
  target_node_ratio: 0.50
  target_edge_ratio: 0.30
  remove_isolated_nodes: true

signed_laplacian:
  eta: 1.0e-3
  epsilon: 1.0e-12

heat_kernel:
  diffusion_time: 1.0

sgw:
  num_spectral_quantiles: 16
  core_dim: 18
  variation_dim: 16
  output_dim: 34
  use_sequence_encoder: false
  node_measure: uniform
  standardize_from_train_only: true

fusion:
  stse_projection_dim: 64
  sgw_projection_dim: 64
  method: concatenate
  hidden_dim: 64
  dropout: 0.20

loss:
  fusion_ce_weight: 1.0
  stse_aux_ce_weight: 0.3
  sgw_aux_ce_weight: 0.5
  laplacian_weight: 0.05
  gw_proxy_weight: 0.02
```

---

# 24. 最终总结

Dual-STSE-HardSGW 的核心思想是：

\[
\boxed{
\text{不要求关键图通道替代已经有效的 STSE，}
}
\]

而是：

\[
\boxed{
\text{让完整图 STSE 保留稳定分类能力，}
\qquad
\text{让硬关键图谱–GW提供额外结构演化信息。}
}
\]

最关键的工程原则是：

\[
\boxed{
\text{通道一必须尽可能直接复用当前已经实现并验证有效的无空间坐标 STSE。}
}
\]

通道一不重新设计、不新增时间编码器、不替换聚合方式、不调整已验证的超参数。所有新设计集中在：

- 通道二的硬关键图提取；
- 谱–GW 演化表示；
- 双通道投影与融合；
- 多头监督；
- 分阶段训练。

只有在保持 STSE 基线稳定的前提下，才能准确判断硬关键图谱–GW 通道是否真正带来了分类增益。
