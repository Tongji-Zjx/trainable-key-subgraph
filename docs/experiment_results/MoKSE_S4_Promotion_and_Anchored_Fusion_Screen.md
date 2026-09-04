# MoKSE S4 双数据集晋级与锚定融合筛查记录

## 1. 晋级决定

S4 静态背景分支在 ADHD 与 WMRC 上均进入下一阶段。该决定由用户明确指定，属于使用固定 test 指标的架构筛选，不应解释为无偏泛化估计。

| 数据集 | 自动规则 | 最终决定 | 主要依据 |
|---|---:|---:|---|
| ADHD | 未通过 | S4 晋级 | 平均 Test ACC 提高 0.021277；AUROC 近似持平，变化 -0.000722 |
| WMRC | 通过 | S4 晋级 | 平均 Test AUROC 提高 0.034593，ACC 提高 0.043578 |

## 2. 下一阶段筛查设置

- 子图完整分支冻结，不反向更新 selector、轨迹模型、M3/M4、Rank 神经头或既有 XGB 残差头。
- S4 使用同一 rotation 的 seeds 43、44、45，并采用三 seed 标准化分数中位数与 seed 不确定性。
- 融合形式为以子图 logit 为锚点的有界静态互补残差，允许精确回退到 `beta=0`。
- 四个 development rotation 采用 leave-one-fold-out 标准化、正交化和 beta 评估。
- beta 搜索范围为 0.00–0.40，固定分类阈值为 0.5。
- 固定 test 不参与 beta 选择。

现有正式子图产物没有严格 development-OOF 预测；现有 validation 同时参与了 checkpoint 选择。因此本轮只能明确标记为 `checkpoint_selection_validation` 的探索性筛查，不能称为严格 OOF 或无偏确认实验。

## 3. 探索性融合结果

两数据集的 development-validation 筛查均选择 `beta=0.35`。

| 数据集 | 路径 | Mean Test AUROC | Mean Test ACC | Mean Test AUPRC | Mean Site-AUC |
|---|---|---:|---:|---:|---:|
| ADHD | 冻结子图 | 0.619237 | 0.630319 | 0.495971 | 0.559980 |
| ADHD | S4 锚定融合 | 0.620471 | 0.635638 | 0.489125 | 0.558804 |
| WMRC | 冻结子图 | 0.592391 | 0.580275 | 0.520913 | 0.595304 |
| WMRC | S4 锚定融合 | 0.569876 | 0.580275 | 0.508787 | 0.567680 |

相对冻结子图分支：

- ADHD：AUROC `+0.001234`，ACC `+0.005319`，但 AUPRC 与 Site-AUC 略降；只构成很弱的探索性正信号。
- WMRC：AUROC `-0.022516`，ACC 不变，AUPRC `-0.012126`，Site-AUC `-0.027624`；出现明显 validation–test 排序反转。

## 4. 当前结论

1. S4 静态分支按用户决定在两个数据集上保留并晋级。
2. 现有结果不支持把 `beta=0.35` 的融合器直接设为正式融合：ADHD 增益很小，WMRC 明显负迁移。
3. `beta=0` 的精确回退机制仍是必要安全边界；S4 静态分支晋级不等于 S4 融合必然晋级。
4. 若继续做确认性融合，需要补建与 checkpoint 选择、模型拟合均不重叠的子图与 S4 development-OOF 预测；普通 validation 不能改名为 OOF。

