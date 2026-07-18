# 时间顺序探索实验：Ordered、Shuffled 与 Independent-bag

## 验收

- 3 种模式 × 5 个训练 seed，共 15 次训练；Shuffled 固定 permutation seed=101。
- validation/test 样本、标签、subject、site、manifest 和 checkpoint 哈希全部一致。
- 三种模式的图编码器、表示维度、分类头和状态张量元素数一致；仅历史/顺序机制不同。
- 所有结果的 evidence level 为 `exploratory_in_sample`。

## Validation（均值 ± 样本标准差）

| 模式 | Log-loss | AUROC | Balanced accuracy |
|---|---:|---:|---:|
| Ordered-GRU | 0.672959 ± 0.004676 | 0.450170 ± 0.025977 | 0.524521 ± 0.009158 |
| Shuffled-GRU (perm=101) | 0.667259 ± 0.002049 | 0.508557 ± 0.047677 | 0.546871 ± 0.019423 |
| Independent-bag | 0.670389 ± 0.003035 | 0.529864 ± 0.011355 | 0.568391 ± 0.004363 |

## Test（均值 ± 样本标准差）

| 模式 | Log-loss | AUROC | Balanced accuracy |
|---|---:|---:|---:|
| Ordered-GRU | 0.666257 ± 0.005898 | 0.574930 ± 0.027545 | 0.540468 ± 0.007461 |
| Shuffled-GRU (perm=101) | 0.663928 ± 0.006594 | 0.558838 ± 0.067522 | 0.529820 ± 0.034890 |
| Independent-bag | 0.654586 ± 0.011242 | 0.599002 ± 0.056772 | 0.559900 ± 0.032004 |

## Test 同-seed配对比较

正值表示目标模式优于对照；log-loss 使用 `对照 − 目标`。

| 目标 vs 对照 | ΔLog-loss (95% CI) | 胜/5 | ΔAUROC (95% CI) | 胜/5 | ΔBAcc | 胜/5 |
|---|---:|---:|---:|---:|---:|---:|
| Ordered vs Shuffled | -0.002328 [-0.012480, 0.007824] | 2/5 | 0.016092 [-0.076743, 0.108927] | 3/5 | 0.010648 | 4/5 |
| Ordered vs Bag | -0.011670 [-0.030394, 0.007053] | 2/5 | -0.024073 [-0.113034, 0.064888] | 1/5 | -0.019432 | 1/5 |
| Bag vs Shuffled | 0.009342 [-0.004370, 0.023055] | 4/5 | 0.040165 [-0.078048, 0.158378] | 3/5 | 0.030080 | 3/5 |

所有精确双侧符号翻转检验及 BH 校正结果均未达到 0.05；5 个 seed 的最小可能双侧 p 值为 0.0625。

## 各 seed 的 Test AUROC

| 模式 | seed42 | seed43 | seed44 | seed45 | seed46 |
|---|---:|---:|---:|---:|---:|
| Ordered-GRU | 0.545218 | 0.601171 | 0.566471 | 0.606593 | 0.555194 |
| Shuffled-GRU (perm=101) | 0.607027 | 0.552809 | 0.444155 | 0.588159 | 0.602039 |
| Independent-bag | 0.623075 | 0.503687 | 0.642594 | 0.634570 | 0.591087 |

## 结论

1. Ordered-GRU 没有稳定优于 Shuffled-GRU：test log-loss 反而平均高 0.002328，AUROC 仅平均高 0.016092，且方向随 seed 改变。因此当前不支持时间顺序贡献。
2. Independent-bag 的 test 均值最好。相对 Ordered，它降低 log-loss 0.011670、提高 AUROC 0.024073；相对 Shuffled，它降低 log-loss 0.009342、提高 AUROC 0.040165。
3. Bag 相对 Shuffled 的 log-loss 在 4/5 seed 改善，但区间仍跨 0。它是下一阶段最合理的探索骨干，不是已经证实的最优模型。
4. 结果更支持“多个窗口的无序集合聚合可能有价值”，不支持“GRU 将早期子图信息按时间顺序有效传递”的主张。
5. 15 次运行中有 4 次出现至少一个 `validation AUROC=0.5 且 threshold=0.5` 的 epoch：Ordered 和 Shuffled 各 2 次，Bag 为 0 次。Bag 在这一定义下未出现完全塌缩，但其跨 seed 波动仍不可忽略。
6. Validation 上 Shuffled 的平均 log-loss 最低，而 Bag 的 AUROC 和 balanced accuracy 最高；Test 上 Bag 三项均值最高。这种指标与分区不一致进一步要求避免把均值排序解释成确定结论。
7. 本轮只有一个 permutation seed，无法描述所有可能排列的分布；但按简化实验的停止规则，已经没有必要立即扩展为 25 次 shuffled 训练。
8. 上游提取器接触过全样本标签，且既有 CUDA 训练未严格逐次复现；这些结果只能用于监督性样本内探索，不能证明样本外泛化或理论机制成立。

## 下一步

冻结 Independent-bag 作为探索性分类骨干，进入子图来源比较：Key vs matched Low-score、Top-degree 和 Random。先使用 seed42–44；只有 Key 稳定优于匹配控制，才继续结构先验或扰动实验。
