# MoKSE-Net-BG-Safe 本轮实现与实验指导（修订版）

> 目标：冻结现有关键子图完整分支，在避免 ADHD 融合负迁移的前提下，提高 WMRC 对完整静态背景信息的利用率。  
> 适用划分：当前固定 20% test、其余 development 样本进行四次 validation 轮换的 60/20/20 协议。  
> 正式约束：固定 test 不参与静态模型、融合权重、checkpoint 或超参数选择；分类阈值固定为 0.5。

---

## 1. 本轮冻结与更新范围

冻结：

```text
M1 Selector
M2 跨窗口轨迹组织
M3 图论演化编码
M4 状态聚合与Rank训练
现有子图XGB残差头
```

每次 rotation 直接读取冻结子图完整分支的最终 margin：

\[
\ell_{KS}.
\]

本轮只更新：

```text
完整静态图节点特征
静态 Signed Weighted Residual GCNII
静态图级读出
静态线性分类头
冻结分支之后的安全标量融合
```

不允许融合损失反向更新子图分支。

---

## 2. 与原方案相比的关键修正

1. 不在每个 rotation 内重新训练三份子图模型；
2. 四次 rotation 的 validation 集互不相交，并覆盖完整 development cohort；
3. 将四次 validation 预测拼成 development OOF 元数据；
4. 仅在 development OOF 上选择一个共享融合权重；
5. 四个 rotation 使用同一融合权重和同一规则；
6. 固定 test cohort 始终不参与选择；
7. 不使用带截距的正斜率校准，避免变相移动固定0.5决策边界；
8. 采用保持logit零点的尺度对齐，而不是均值中心化z-score；
9. ADHD回退由显式程序分支直接返回原始 \(\ell_{KS}\)，不是返回标准化后的logit；
10. 本轮不增加第二个双分支XGB。

---

## 3. 静态节点输入

基础输入保持：

\[
12D\ static+8D\ signed\ spectral=20D.
\]

新增每个节点的正负连接权重分布轮廓：

```text
positive q25 / q50 / q75 / q90
negative-magnitude q25 / q50 / q75 / q90
has-positive-edge
has-negative-edge
```

记为：

\[
F_i^{profile}\in\mathbb R^{10}.
\]

最终输入：

\[
X_i^{BG}
=
[F_i^{static}\Vert U_i^{spec}\Vert F_i^{profile}]
\in\mathbb R^{30}.
\]

使用q90而不是最大值，以减少节点数和极端边权对尾部统计的影响。validity flag只用于处理不存在某一符号边的节点；若训练集中为常量，其标准化结果自然为零。

所有30维输入只在当前rotation的train样本真实节点上拟合mean/std；validation和test只复用冻结scaler。

---

## 4. 静态GNN与社区残差

主干保持两层 Signed Weighted Residual GCNII：

```text
30D input
→ LayerNorm + Linear(30,64)
→ 2-layer signed positive/negative GCNII
→ 原有逐层 node mean+population-std residual readout
```

新增社区残差只读取第二层节点状态。对样本内每个社区：

\[
u_c=\frac{1}{|V_c|}\sum_{i\in V_c}H_i^{(2)}.
\]

然后：

\[
g_{comm}
=[mean_c(u_c)\Vert std_c(u_c)].
\]

最终：

\[
g_{BG}
=g_{existing}
+\gamma_{comm}W_{comm}LN(g_{comm}),
\]

\[
\gamma_{comm}
=0.25\sigma(a_{comm}),
\qquad
\gamma_{comm}^{init}=0.05.
\]

社区编号只在样本内部使用，不作为embedding；聚合对社区编号排列不敏感。

静态输出仍为：

\[
z_{BG}\in\mathbb R^{24},
\qquad
\ell_{BG}\in\mathbb R.
\]

---

## 5. Signed-balanced DropEdge

训练时构造两个视图：

```text
view 0：完整原始静态图
view 1：signed-balanced DropEdge图
```

DropEdge规则：

1. 仅对原始正负边权执行；
2. 在矩阵上三角分别选取正边和负边；
3. 每个符号通道删除相同比例的真实边；
4. 镜像回下三角以保持无向对称；
5. 不产生新边；
6. 保留未删除边的原始符号和权重；
7. 删除后重新计算正负度归一化；
8. validation/test完全关闭DropEdge。

第一轮：

\[
p_{drop}=0.05.
\]

30维节点特征与谱位置编码始终由完整原图计算，不随随机view重新计算。DropEdge只扰动GNN消息传播结构，避免随机谱分解和高额CPU开销。

每个掩码由：

```text
global seed + epoch + sample_key + view_id
```

确定，保证严格重训可复现。

---

## 6. 静态分支训练目标

两个视图分别产生：

\[
(z_{BG}^{(0)},\ell_{BG}^{(0)}),
\qquad
(z_{BG}^{(1)},\ell_{BG}^{(1)}).
\]

平均logit：

\[
\bar\ell_{BG}
=\frac{\ell_{BG}^{(0)}+\ell_{BG}^{(1)}}{2}.
\]

一致性损失：

\[
L_{cons}
=mean_s\left[1-cos(z_s^{(0)},z_s^{(1)})
+0.1(\ell_s^{(0)}-\ell_s^{(1)})^2\right].
\]

总损失：

\[
L_{BG}
=\frac12[BCE(\ell^{(0)},y)+BCE(\ell^{(1)},y)]
+0.05L_{rank}(\bar\ell,y)
+0.05L_{cons}.
\]

保持：

```text
无类别加权BCE
rank temperature = 1.0
decision threshold = 0.5
严格确定性训练
```

---

## 7. Top-3 checkpoint输出集成

每个rotation按预先冻结的规则保存validation最优三个epoch：

```text
Validation AUROC
→ ACC@0.5
→ 较低train loss
```

推理时分别前向三个checkpoint，并平均：

\[
\ell_{BG}=\frac13\sum_{r=1}^{3}\ell_{BG}^{(r)},
\qquad
z_{BG}=\frac13\sum_{r=1}^{3}z_{BG}^{(r)}.
\]

这属于同一训练轨迹的输出集成，不增加训练次数。Top-1仍单独保留，便于判断增益来自新特征/正则还是checkpoint集成。

---

## 8. 静态分支筛选流程

为避免五项修改捆绑后无法归因，静态分支按以下条件递进：

| 条件 | Profile | Community residual | DropEdge/Consistency | Top-3 |
|---|---:|---:|---:|---:|
| S0 | × | × | × | × |
| S1 | ✓ | × | × | × |
| S2 | ✓ | ✓ | × | × |
| S3 | ✓ | ✓ | ✓ | ✓ |

筛选只使用train和validation，不查看固定test。

选择顺序：

1. 首先比较WMRC development-OOF mean rotation AUROC；
2. 再比较最差rotation和rotation标准差；
3. 检查validation相对train的gap；
4. 检查site-stratified disease AUROC；
5. 若性能近似，选择结构更简单的条件。

本轮按实验要求为 S0–S3 每个条件都导出固定 test 指标，以完整观察每项改动的外部表现；但 test 结果不得参与阶段选择、checkpoint 选择、融合权重选择或任何重训决策。正式阶段仍只由 development validation rotations 决定。

---

## 9. 四次validation组成development OOF

当前60/20/20协议先形成五个互斥block：

```text
block 0：固定test，永不进入开发集
block 1–4：development
```

四次rotation中，development的每个样本：

```text
恰好一次作为validation
恰好三次作为train
```

因此，取每个样本作为validation时对应模型的预测，可得到完整development OOF：

\[
\mathcal D_{meta}^{OOF}
=\{y_s,site_s,\ell_{KS,s}^{OOF},\ell_{BG,s}^{OOF}\}.
\]

这套元数据用于选择融合权重，不需要重新训练子图分支，也不接触固定test。

---

## 10. 保持零点的尺度对齐

在development OOF上估计两个分支logit的population standard deviation：

\[
\sigma_{KS}^{dev},\qquad \sigma_{BG}^{dev}.
\]

只缩放静态logit：

\[
\ell_{BG}^{aligned}
=\ell_{BG}
\frac{\sigma_{KS}^{dev}}
{\sigma_{BG}^{dev}+\epsilon}.
\]

不减均值，因此logit零点和固定0.5阈值语义不被中心化操作改变。

---

## 11. 安全凸融合

\[
\boxed{
\ell_F(a)
=a\ell_{KS}+(1-a)\ell_{BG}^{aligned}
},
\qquad
a\in\{0,0.05,\ldots,1.0\}.
\]

对每个候选权重计算四个validation rotation的AUROC：

\[
J(a)
=mean(AUC_r(a))-0.5std(AUC_r(a)).
\]

在距离最优目标不超过预设容忍值的候选中：

- ADHD选择最接近 \(a=1\) 的候选；
- WMRC选择目标最高的候选。

最终四个rotation共享同一个 \(a^*\)。

---

## 12. ADHD No-Harm与精确回退

只有同时满足以下development OOF条件才启用静态融合：

```text
mean rotation AUROC增益 >= 0.005
至少3/4 rotation AUROC不下降
最差rotation下降不超过0.01
site-stratified disease AUROC下降不超过0.01
```

否则执行显式回退：

```python
final_logit = original_subgraph_logit
```

回退路径不经过缩放、融合或校准，必须逐样本精确复现冻结E0输出。

该规则降低负迁移风险，但不声称能够数学保证未知test绝不下降；最终仍需报告固定test结果。

---

## 13. WMRC启用与回退规则

优先要求：

```text
相对冻结子图分支mean rotation AUROC增益 >= 0.005
至少3/4 rotation不下降
corrected ranking pairs > corrupted ranking pairs
site-stratified disease AUROC不明显下降
```

若融合不通过，比较development OOF上的两个单分支稳定性目标，回退到更稳定的单分支。允许静态分支权重大于子图分支，但必须明确报告该情况。

---

## 14. 固定test评估

四个rotation共享同一固定test cohort，但产生四组模型预测。因此不能把它称为四个独立outer-test折或pooled OOF。

必须同时报告：

1. 四个rotation在同一固定test上的指标均值和标准差；
2. 四个rotation fused logits等权平均后的ensemble test指标；
3. 每个rotation是否触发回退；
4. 融合相对各自冻结E0的配对差值。

主要结构筛选依据是development OOF；固定test只作最终一次验收。

本轮会同时报告 S0–S3 的固定 test 结果；这些结果属于并列的一次性审计，不允许用于反向选择阶段或调整训练配置。

---

## 15. 本轮XGB边界

冻结的 \(\ell_{KS}\) 已包含原子图分支XGB。本轮安全融合后不再增加第二个XGB，先明确静态信息和融合机制本身是否有效。

只有Safe Fusion在WMRC稳定成立后，下一轮才评估低维双分支XGB：

```text
PCA(z_KS) 4–8D
+ PCA(z_BG) 4–8D
+ l_KS
+ l_BG
+ logit difference/product
→ shallow residual XGB
```

PCA、XGB和残差系数必须只由development OOF/validation选择，固定test不得参与。

---

## 16. 产物与来源审计

每个实验必须保存：

```text
split assignment SHA256
子图checkpoint SHA256
子图XGB模型与参数SHA256
静态checkpoint SHA256
静态缓存schema和feature names
train-only scaler
sample_key/site/label/split
l_KS/l_BG/l_BG_aligned/l_fused
fusion weight/fallback flag
```

正式启动前必须解决历史报告中WMRC静态分支 `0.581867` 与 `0.591960` 的产物来源差异，不能混用不同训练模式或checkpoint。

---

## 17. 实现模块

修改：

```text
src/keysubgraph/background/data.py
src/keysubgraph/background/model.py
src/keysubgraph/background/training.py
src/keysubgraph/background/__init__.py
```

新增：

```text
src/keysubgraph/background/safe_fusion.py
scripts/run_mokse_background_safe_fold.py
scripts/select_mokse_background_safe_stage.py
scripts/fit_mokse_background_safe_fusion.py
tests/test_mokse_background_safe.py
```

说明：四次validation拼接得到的是“样本未进入对应模型参数训练”的development轮换预测，可用于本轮固定test协议下的模型开发；但已有基础模型的epoch或XGB轮数可能曾使用同一validation选择，因此它不等价于额外嵌套一层、完全独立的inner-OOF估计。固定test仍保持完全隔离。如需估计严格嵌套泛化误差，必须重新执行内层基础模型训练，本轮冻结方案不承担该额外开销。

旧入口和旧缓存不得被静默解释为新版本产物。

---

## 18. 验收标准

本地代码验收：

```text
Profile维度、符号分离和节点置换等变
社区池化忽略padding和社区编号置换
DropEdge对称、无新边、保留权重、严格可复现
validation/test不启用DropEdge
Top-3输出平均正确
四个validation集合互斥并覆盖development
固定test集合四次完全一致
a=1逐样本精确复现原l_KS
ADHD回退逐样本精确复现原l_KS
固定test不参与融合选择
```

实验晋级：

- ADHD正式要求不低于冻结E0；开发容忍值只用于排错，不用于宣称增益；
- WMRC优先看development-OOF与固定test是否同方向；
- WMRC目标为mean rotation AUROC提高至少0.01，0.005–0.01视为弱增益；
- 同时报告ACC@0.5、AUPRC、BA、Sensitivity、Specificity和site-stratified disease AUROC。

---

## 19. 一句话原则

\[
\boxed{
\text{先独立改善完整静态图表示，
再用development OOF学习一个可精确回退的低容量融合；
固定test只负责最终验收。}
}
\]
