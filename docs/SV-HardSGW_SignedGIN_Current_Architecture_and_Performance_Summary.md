# SV-HardSGW SignedGIN 当前架构与性能总结

更新日期：2026-07-29  
当前模型变体：`signed_gin_multibranch_late_fusion`  
当前状态：**研究默认架构（SVG）**

> 2026-07-31 决策更新：项目选择 SVG 作为理论契合优先的研究默认架构，
> 原 `current` selector 保持不变，S 保留为主要精简基线。本文后续早期状态
> 描述按历史记录保留；最新决策及解释边界以
> `docs/SV_HardSGW_SVG_Default_Architecture_Decision.md` 为准。

## 1. 任务与设计目标

当前架构面向带符号、可变长度脑图序列的二分类任务。其目标是：

1. 由学习型 selector 从每个时间窗口提取紧凑硬关键子图；
2. 保留正、负连接及社区结构语义；
3. 分别学习静态谱结构、跨窗口 Variation 和关键子图神经表示；
4. 在不使用空间坐标、站点标签和原始社区编号 embedding 的情况下完成分类；
5. 缓解旧架构中的 GIN 表示低秩化和多通道融合负迁移。

该架构只把类别标签用于训练损失和 validation 模型选择；test/外折样本不用于
checkpoint、阈值或架构选择。

## 2. 总体流程

```text
带符号全图序列
        │
        ▼
学习型 selector
  节点目标比例 0.50
  边目标比例   0.30
  社区覆盖约束
        │
        ▼
冻结的带符号硬关键子图序列
        │
        ├──────────────┬─────────────────┐
        ▼              ▼                 ▼
修复后 Signed GIN   Static-spectral   Variation
        │              │                 │
     16维投影        16维投影          16维投影
        │              │                 │
     独立分类头      独立分类头        独立分类头
        └──────────────┴─────────────────┘
                       │
                非负 logit 后期融合
                       │
                       ▼
                    二分类输出
```

## 3. 数据与硬子图约束

### 3.1 可变长度

- 不假设不同样本具有相同时间窗口数或节点数；
- 使用 list-based batching；
- 不通过截断改变原图或关键子图；
- 无效窗口和无效相邻窗口转换不参与统计。

### 3.2 带符号边

边存在条件为：

\[
\mathbf 1(|A_{ij}|>\tau).
\]

其中阈值 \(\tau\) 来自冻结数据协议。负边是有效连接，不被视为缺边。
Signed GIN 使用保留符号的归一化邻接：

\[
\widetilde A=D_{|A|}^{-1/2}AD_{|A|}^{-1/2}.
\]

### 3.3 社区信息

社区编号只用于同社区判断、社区覆盖和结构统计，不输入
`nn.Embedding(community_id)`。模型使用的是具有跨样本一致语义的社区规模、
社区内外正负连接强度和密度。

### 3.4 不使用的信息

当前架构不使用：

- 空间坐标；
- 站点标签；
- ROI 名称 embedding；
- 原始社区编号 embedding。

## 4. 三条分类分支

### 4.1 Static-spectral 分支

每个有效硬图窗口计算 signed Laplacian 的 16 个固定分位点谱统计，再对窗口取均值，
得到 16 维静态谱表示。

硬图特征构造器还会生成 12 维静态结构统计，但当前后期融合版不把这 12 维送入
static 分支。原因是既有冻结诊断没有观察到其稳定增益。

处理过程为：

```text
16维 static-spectral
→ Linear(16,16)
→ GELU
→ LayerNorm
→ Linear(16,16)
→ GELU
→ Dropout(0.10)
→ Linear(16,2)
```

### 4.2 Variation 分支

Variation 表示相邻有效窗口间 16 维谱分位点变化的平均绝对幅度：

\[
v=\frac{1}{|\mathcal T|}
\sum_{m\in\mathcal T}|q^{(m+1)}-q^{(m)}|.
\]

它只统计真实相邻且均有效的窗口转换。该 16 维向量通过独立的
`16→16→2` 投影和分类头，不再与其他通道在输入层直接拼接。

### 4.3 修复后的 Signed GIN 分支

每个硬图节点使用 15 维、从硬图重新计算的无泄漏特征：

- 绝对连接强度；
- 正连接强度；
- 负连接幅值；
- 连接强度变化及其有效 mask；
- 平均边变化幅度；
- 有效变化比例；
- 七项社区结构特征；
- 局部聚类系数。

GIN 的主要配置为：

| 项目 | 当前值 |
|---|---:|
| 节点输入维数 | 15 |
| 隐藏维数 | 64 |
| GIN 层数 | 2 |
| 消息模式 | `signed_normalized` |
| 残差连接 | 是 |
| Jumping Knowledge | 是 |
| 节点池化 | `mean + std` |
| 每窗口压缩维数 | 32 |
| 跨窗口聚合 | `mean + std` |
| 样本级 GIN 表示 | 64维 |
| 分支投影 | 64→16 |

GIN 表示和投影使用 SafeBatchNorm：仅训练批次更新 running statistics；
validation/test 只使用冻结统计。该设计用于降低样本间近乎共线和表示低秩化。

## 5. 后期融合与训练目标

三个分支分别输出二分类 logits：

\[
\ell_g,\quad \ell_s,\quad \ell_v.
\]

全局可学习权重经过 softmax：

\[
\alpha_k=\operatorname{softmax}(w)_k,\qquad
\alpha_k\geq0,\qquad\sum_k\alpha_k=1.
\]

最终输出为：

\[
\ell=
\alpha_g\ell_g+
\alpha_s\ell_s+
\alpha_v\ell_v.
\]

训练损失为：

\[
\mathcal L=
\mathcal L_{\mathrm{fusion}}
+0.25\,
\frac{\mathcal L_g+\mathcal L_s+\mathcal L_v}{3}.
\]

主损失和三个辅助损失均使用类别加权交叉熵。辅助损失用于防止某个分支在联合训练中
被完全忽略。

正式交叉拟合中的主要训练设置：

| 项目 | 当前值 |
|---|---:|
| 最大 epoch | 80 |
| 物理 batch size | 4 |
| 梯度累积步数 | 2 |
| 有效 batch size | 8 |
| 学习率 | 0.001 |
| weight decay | 0.0001 |
| gradient clip | 1.0 |
| early stopping patience | 15 |
| checkpoint 指标 | Composite AUC |

Composite AUC 定义为 validation 全局 AUROC 与 site-stratified AUROC 的平均值。
分类阈值只在 validation 上按 balanced accuracy 拟合，随后冻结到 test 或外折。

## 6. 程序验收

- 本地全量单元测试：336 项通过，1 项跳过；
- 服务器 CUDA 回归：通过；
- 16 个平衡样本记忆实验：AUROC、BA、Accuracy 均为 `1.0000`；
- 梯度能够到达三个分支及融合参数；
- 融合权重非负且总和为 1；
- 旧 SG0–SG2 模型入口和 checkpoint 保持兼容。

## 7. WMRC 数据集表现

### 7.1 单次冻结划分

单次划分只用于架构筛选，不作为最终泛化结论。

| Split | AUROC | Site-AUC | Composite AUC | BA | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.6584 | 0.6927 | 0.6755 | 0.6371 | 0.6341 | 0.6053 |
| Test | 0.5824 | 0.5696 | 0.5760 | 0.5550 | 0.5610 | 0.5000 |

Validation 最佳 epoch 为 7，冻结 BA 阈值为 `0.4970`。Validation 增益没有完整
传递到 test，提示划分敏感性。

### 7.2 3-fold 完整交叉拟合 OOF

样本数为 546。每折重新训练 selector、重新导出硬图、重新拟合 train-only scaler，
每个样本恰好获得一次外折预测。

| 指标 | 当前架构 |
|---|---:|
| Pooled OOF AUROC | 0.5634 |
| Site-stratified OOF AUROC | 0.5591 |
| Mean fold AUROC | 0.5613 ± 0.0572 |
| Accuracy | 0.5641 |
| Balanced Accuracy | 0.5411 |
| F1 | 0.4306 |
| Sensitivity | 0.3879 |
| Specificity | 0.6943 |

各外折：

| Fold | AUROC | Site-AUC | BA | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5860 | 0.5775 | 0.5657 | 0.5769 | 0.4967 |
| 1 | 0.4823 | 0.4682 | 0.4628 | 0.5000 | 0.2720 |
| 2 | 0.6156 | 0.6081 | 0.5939 | 0.6154 | 0.5000 |

Pooled AUROC 的事后 10,000 次 bootstrap 95% CI 为 `[0.5149, 0.6108]`；
10,000 次标签置换单侧 `p=0.0071`。这表明存在弱但可检测的 OOF 排序信号，
不代表已经达到强分类性能。

### 7.3 与改进前架构比较

| 指标 | 改进前 | 当前架构 | 差值 |
|---|---:|---:|---:|
| Pooled OOF AUROC | 0.5378 | 0.5634 | +0.0255 |
| Site-stratified OOF AUROC | 0.5367 | 0.5591 | +0.0225 |
| Accuracy | 0.4908 | 0.5641 | +0.0733 |
| Balanced Accuracy | 0.5292 | 0.5411 | +0.0119 |
| F1 | 0.5670 | 0.4306 | −0.1364 |
| Fold AUROC SD | 0.0077 | 0.0572 | +0.0495 |

当前架构获得小幅总体 AUROC 增益，但折间稳定性下降。

## 8. 表示修复与融合诊断

### 8.1 GIN 低秩化已明显缓解

旧 WMRC 模型：

| 表示 | 归一化有效秩 | 平均样本间余弦 |
|---|---:|---:|
| GIN representation | 0.059 | 0.9989 |
| GIN projection | 0.115 | 1.0000 |

当前架构三个外折的 validation：

| Fold | 原始GIN秩 | 原始GIN余弦 | GIN投影秩 | GIN投影余弦 |
|---:|---:|---:|---:|---:|
| 0 | 0.1440 | 0.9760 | 0.4506 | 0.4109 |
| 1 | 0.1519 | 0.9785 | 0.3962 | 0.2477 |
| 2 | 0.1570 | 0.9884 | 0.4229 | 0.5427 |

GIN 表示修复在三个外折均复现，因此不是单次划分偶然现象。

### 8.2 融合负迁移尚未解决

冻结外折预测的只读分支比较：

| 路径 | Pooled OOF AUROC | Mean fold AUROC ± SD |
|---|---:|---:|
| 融合 | 0.5634 | 0.5613 ± 0.0572 |
| GIN | 0.5303 | 0.5309 ± 0.0622 |
| Static-spectral | **0.5827** | **0.5792 ± 0.0220** |
| Variation | 0.5290 | 0.5327 ± 0.0296 |

融合权重在三折中始终接近均匀：

| Fold | GIN | Static-spectral | Variation |
|---:|---:|---:|---:|
| 0 | 0.3658 | 0.3169 | 0.3173 |
| 1 | 0.3468 | 0.3310 | 0.3223 |
| 2 | 0.3716 | 0.3302 | 0.2981 |

当前全局 softmax 融合不能在弱分支无增益时将其权重收缩到零，导致融合结果低于较稳定的
static-spectral 分支。

## 9. ADHD 数据集状态

截至本文更新时，`signed_gin_multibranch_late_fusion` 尚未完成 ADHD 938 样本的
正式 3-fold 交叉拟合。因此：

- 不把旧 `signed_gin_static_variation`、D3-B 或其他 ADHD 单划分结果记为当前架构成绩；
- 不对当前架构在 ADHD 上的泛化能力作结论；
- 已冻结的计划使用 `site_subject` 分组，每折重新训练 selector、硬图、scaler 和分类器；
- 待 ADHD OOF 完成后补充 pooled AUROC、site-stratified AUROC、各折波动和分支贡献。

## 10. 当前结论与下一步

当前证据支持：

1. 修复后的 Signed GIN 能学习明显更非退化的关键子图表示；
2. 相较改进前模型，WMRC 总体 OOF AUROC 有小幅提升；
3. 当前融合仍存在折间负迁移，不能视为稳定优于最佳独立分支；
4. 当前模型应保留为实验候选，而不是正式默认模型。

下一步优先事项：

1. 完成 ADHD 3-fold OOF，判断上述结论能否跨数据集复现；
2. 保留已经验证有效的 GIN 修复；
3. 将 static-spectral 设为主干；
4. 将 GIN 和 Variation 改为默认收缩到零的残差专家，避免无增益时破坏主干。

## 11. 代码与结果位置

关键实现：

```text
src/keysubgraph/features/sv_hard_graph_features.py
src/keysubgraph/models/sv_signed_gin.py
src/keysubgraph/training/sv_signed_gin_trainer.py
src/keysubgraph/crossfit/sv_signed_gin_runner.py
```

设计与详细实验报告：

```text
docs/SV-HardSGW_SignedGIN_LateFusion_Improvement_Design.md
docs/experiment_results/sv_signed_gin_late_fusion_wmrc_crossfit_report.md
```

冻结 WMRC 结果：

```text
analysis_artifacts/sv_signed_gin_late_fusion_wmrc_20260729/
```

当前代码分支与本文采用的实现基线：

```text
branch: codex/sv-late-fusion-v1
implementation baseline: 81dff29
```
