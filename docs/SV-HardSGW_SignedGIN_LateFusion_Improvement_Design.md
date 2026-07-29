# SV-HardSGW SignedGIN 后期融合改进设计

## 1. 目标

本版本只处理两个在 ADHD 与 WMRC 上共同出现的架构瓶颈：

1. 静态、Variation、GIN 直接拼接导致融合负迁移；
2. GIN 表示低秩化，并在联合模型中被分类头忽略。

旧 SG0–SG2 的模型结构、checkpoint 和实验入口保持兼容。新增变体为：

```text
signed_gin_multibranch_late_fusion
```

## 2. 架构

```text
冻结的 learned hard subgraph sequence
                 |
      +----------+-----------+
      |                      |
      v                      v
修复后的 Signed GIN       冻结统计特征
      |                 +----+----+
      |                 |         |
      v                 v         v
GIN projection     Static spectral  Variation
      |                 |         |
      v                 v         v
GIN auxiliary head  Static head  Variation head
      |                 |         |
      +-------- nonnegative logit fusion -------+
                             |
                             v
                         final logits
```

### 2.1 静态分支

仅使用 28 维 static 中的前 16 维谱统计。既有诊断显示，后 12 维
static structural 在两个数据集上没有提供稳定增益。

### 2.2 Variation 分支

保留原 16 维 Variation，并使用独立投影与分类头。该分支不再与其他
特征在输入层直接拼接。

### 2.3 GIN 分支

采用以下修复：

- 带符号邻接矩阵按绝对度进行对称归一化；
- 正负边符号不变；
- 每层使用残差连接；
- 使用 Jumping Knowledge 融合初始层和各消息传递层；
- 节点池化改为 `mean + std`，避免近似均匀 attention 只产生均值；
- GIN 分支拥有独立辅助分类头。

### 2.4 后期融合

三个分支分别输出二分类 logits：

\[
\ell_g,\quad \ell_s,\quad \ell_v.
\]

融合权重通过 softmax 约束为非负且和为1：

\[
\alpha_k=\operatorname{softmax}(w)_k,
\qquad
\ell=\alpha_g\ell_g+\alpha_s\ell_s+\alpha_v\ell_v.
\]

这样禁止通过负权重反向破坏某个分支的排序。

## 3. 损失

\[
\mathcal L=
\mathcal L_{\mathrm{fusion}}
+
\lambda_{\mathrm{aux}}
\frac{
\mathcal L_g+\mathcal L_s+\mathcal L_v
}{3}.
\]

默认：

```text
lambda_aux = 0.25
```

主损失和三个辅助损失均使用相同的类别加权交叉熵。辅助损失只防止
分支被忽略，不改变 selector 和硬图。

## 4. 冻结范围

本实验继续冻结：

- learned selector；
- 硬图；
- train-only scaler。

只训练：

- 改进后的 Signed GIN；
- 三个分支投影与辅助头；
- 三个非负融合权重。

不使用坐标、站点标签或原始社区编号 embedding。

## 5. 自动实验闸门

### 闸门A：程序正确性

- 单元测试、forward、loss、backward、保存、加载全部通过；
- 旧 SG0–SG2 checkpoint 仍可加载；
- fusion 权重非负且和为1；
- 分类梯度到达三个分支。

### 闸门B：最小记忆能力

16个平衡训练样本上，关闭 dropout 和 weight decay 后：

```text
train replay AUROC >= 0.90
```

若失败，停止正式实验并排查实现。

### 闸门C：WMRC 单划分

旧 SG2 与改进版必须使用完全相同的 selector、硬图、scaler 和划分。
进入交叉拟合至少要求：

1. 改进版 validation composite AUC 比旧 SG2 高至少 0.01；
2. 改进版 final AUROC 不低于最佳独立分支超过 0.01；
3. validation GIN representation 归一化有效秩不低于 0.10；
4. GIN projection 平均样本间余弦相似度低于 0.995。

### 闸门D：3-fold OOF

只有闸门C通过后才运行 WMRC 3-fold 完整交叉拟合。每折均重新训练
selector，并只使用 inner-validation 选 checkpoint 和阈值。最终报告：

- mean fold AUROC ± SD；
- pooled OOF AUROC；
- site-stratified OOF AUROC；
- OOF Accuracy、BA、F1；
- 每折 GIN 秩和融合权重。

## 6. 结论边界

单次划分只能用于筛选架构。只有3-fold完整交叉拟合通过后，才能认为
改进对 WMRC 的未见样本具有较稳定的分类增益。
