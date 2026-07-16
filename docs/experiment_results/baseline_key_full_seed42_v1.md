# Key-subgraph full-history baseline：seed 42

## 实验身份

- 实验 ID：`baseline_key_full_seed42_v1`
- 证据等级：`exploratory_in_sample`
- 子图来源：`key`
- 历史模式：`full`
- 时间顺序：`ordered`
- 时间差分：关闭
- 结构特征分支与统计先验：关闭
- 最佳 checkpoint 选择指标：validation unweighted log-loss
- 分类阈值：在 validation 选择并原样用于 test

关键子图提取器使用过全部 938 个样本的标签，因此本记录属于监督性样本内探索，不能表述为独立泛化验证。

## 训练结果

训练最多 100 epochs，early-stopping patience 为 15。训练在第 24 个 epoch 正常提前停止，最佳 epoch 为 9。

| 项目 | 数值 |
|---|---:|
| 最佳 epoch 的 train weighted loss | 0.689177 |
| 训练耗时 | 337.22 s |
| CUDA 峰值显存 | 51.51 MiB |
| validation 分类阈值 | 0.460415 |

## 数据划分

| 分区 | 样本数 | 0 类 | 1 类 |
|---|---:|---:|---:|
| train | 657 | 408 | 249 |
| validation | 141 | 87 | 54 |
| test | 140 | 87 | 53 |

## 效果

| 指标 | Validation | Test |
|---|---:|---:|
| Unweighted log-loss | 0.678096 | 0.668603 |
| AUROC | 0.461260 | 0.545218 |
| Balanced accuracy | 0.537676 | 0.543049 |
| Accuracy | 0.602837 | 0.642857 |
| F1 | 0.333333 | 0.218750 |
| Sensitivity | 25.93% | 13.21% |
| Specificity | 81.61% | 95.40% |

Validation confusion matrix：

```text
[[71, 16],
 [40, 14]]
```

Test confusion matrix：

```text
[[83, 4],
 [46, 7]]
```

使用训练集 1 类比例 `249/657` 作为固定预测概率时，validation/test 的常数基线 log-loss 分别为 0.665537 和 0.663361。当前模型分别高 1.89% 和 0.79%，尚未超过该常数概率基线。

## 冻结用途

本结果作为后续 `current_only`、`truncate_history`、`independent_bag`、顺序置换和子图来源实验的原始比较基准。后续比较应复用相同下游划分、节点特征、隐藏维度、优化器、训练预算、checkpoint 规则与评估指标。
