# SV-HardSGW + Signed GIN Key Subgraph Encoder 模型设计与实验方案

## 1. 设计目标

当前 SV-HardSGW 已完成关键子图选择、静态结构统计、Variation动态特征和分类，但主要学习模块集中在评分器和分类头，缺少关键子图表示学习模块。

因此加入：

\[
\boxed{\text{Signed GIN Key Subgraph Encoder}}
\]

目标：

\[
\text{关键子图选择 + 人工统计}
\rightarrow
\text{关键子图选择 + 神经表示学习}
\]

---

# 2. 总体架构

```text
动态图序列
      |
      v
Hard Key Subgraph Selector
      |
      v
关键子图序列 U(1)...U(M)
      |
      +----------------+
      |                |
      v                v
 Signed GIN        Variation计算
 Encoder              |
      |                |
      v                v
Graph Embedding   16维Variation
      |                |
      +-------+--------+
              |
              v
          Feature Fusion
              |
              v
             MLP
              |
              v
          Classification
```

---

# 3. Signed GIN关键子图编码器

## 3.1 输入

输入为Hard关键子图：

\[
U^{(m)}
\]

节点特征保持当前15维设计，包括连接强度、degree、动态变化和社区结构信息。

边特征：

\[
e_{ij}=[A_{ij},|A_{ij}|]
\]

保留正负边信息。

---

## 3.2 Signed GIN

普通GIN：

\[
h_v^{k+1}
=
MLP
\left(
(1+\epsilon)h_v^k
+
\sum_{u\in N(v)}h_u^k
\right)
\]

针对signed graph，将邻居拆分：

\[
N_v^+,
\quad
N_v^-
\]

并进行：

\[
h_v^{k+1}
=
MLP
\left(
(1+\epsilon)h_v^k
+
\sum_{u\in N_v^+}h_u^k
-
\sum_{u\in N_v^-}h_u^k
\right)
\]

---

# 4. 子图表示生成

每个窗口：

\[
z^{(m)}
=
GIN(U^{(m)})
\]

节点表示通过Attention Pooling得到图级表示：

\[
z^{(m)}
=
\sum_i \alpha_i h_i^{(m)}
\]

推荐：

\[
z^{(m)}
\in
\mathbb R^{64}
\]

时间窗口采用Mean Pooling：

\[
z_{GIN}
=
\operatorname{Mean}_m
\left(
z^{(m)}
\right)
\]

---

# 5. Variation分支

保持当前有效动态特征：

\[
v^{(m)}
=
\left|
Q^{(m+1)}-Q^{(m)}
\right|
\]

聚合：

\[
V
=
\operatorname{Mean}_m
\left(
v^{(m)}
\right)
\]

其中：

\[
V
\in
\mathbb R^{16}
\]

---

# 6. 特征融合

## 方案A：GIN替代静态统计

\[
[z_{GIN};V]
\]

## 方案B：GIN + 原静态统计

\[
[z_{GIN};S_{\mathrm{static}};V]
\]

用于验证神经表示和人工统计的互补性。

---

# 7. 训练策略

## 阶段一

冻结：

- selector；
- 节点评分器；
- 边评分器。

训练：

- Signed GIN；
- pooling层；
- 分类头。

目的：验证关键子图表示学习是否有效。

## 阶段二

若阶段一有效，再联合微调selector和GIN。

---

# 8. 实验设计

## Experiment 1：GIN有效性

比较：

| 模型 | 输入 |
|---|---|
| Baseline | Static + Variation |
| GIN | GIN embedding + Variation |
| GIN + Static | GIN embedding + Static + Variation |

指标：

- AUROC；
- Accuracy；
- Balanced Accuracy；
- F1。

---

## Experiment 2：编码器比较

比较：

- MLP统计特征；
- GCN；
- GIN；
- Signed GIN。

---

## Experiment 3：Pooling消融

比较：

- Mean Pooling；
- Max Pooling；
- Attention Pooling。

---

## Experiment 4：Signed边消融

比较：

- Signed GIN；
- Unsigned GIN。

验证负边信息作用。

---

## Experiment 5：关键子图有效性

比较：

- 完整图GIN；
- Hard关键图GIN。

验证关键子图是否减少冗余。

---

## Experiment 6：随机子图实验

比较：

\[
GIN(U_{\mathrm{learned}})
\]

与：

\[
GIN(U_{\mathrm{random}})
\]

验证选择器有效性。

---

# 9. 推荐最终版本

第一版采用：

\[
\boxed{
Hard\ Selector
+
Signed\ GIN
+
Variation
+
MLP
}
\]

优势：

1. 修改最小；
2. 保留Hard-SGW理论；
3. 增加真正神经表示学习；
4. 参数量低；
5. 适合小样本脑网络数据；
6. 为未来Temporal Encoder和Memory扩展提供接口。