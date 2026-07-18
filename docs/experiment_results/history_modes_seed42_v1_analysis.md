# 历史模式探索实验对比：seed 42

## 1. 分析范围与完整性

本次比较包含六个实验：

- `full`
- `current_only`
- `truncate_history=0.25`
- `truncate_history=0.50`
- `truncate_history=0.75`
- `independent_bag`

六个实验使用相同的父 manifest、下游划分、关键子图、节点特征、模型维度、训练预算和 seed 42。所有 validation/test 文件分别包含相同且完整的 141/140 条逐样本预测；样本键、真实标签、subject 和 site 均严格对齐。test 中有 140 个 subject group，本数据上每组对应一个 test 样本。

证据等级仍为 `exploratory_in_sample`：关键子图提取器使用过全部 938 个样本标签，结果不能表述为独立泛化验证。

各实验均因 validation log-loss 连续 15 个 epoch 未改善而正常 early-stop：

| 历史模式 | 最佳 epoch | 完成 epochs |
|---|---:|---:|
| full | 9 | 24 |
| current-only | 24 | 39 |
| truncate 25% | 8 | 23 |
| truncate 50% | 2 | 17 |
| truncate 75% | 2 | 17 |
| independent Bag | 9 | 24 |

## 2. 聚合效果

### Validation

| 历史模式 | Log-loss ↓ | AUROC ↑ | Balanced accuracy ↑ |
|---|---:|---:|---:|
| full | 0.678096 | 0.461260 | 0.537676 |
| current-only | 0.676314 | 0.492550 | 0.545977 |
| truncate 25% | 0.679729 | 0.466156 | 0.511814 |
| truncate 50% | 0.675658 | 0.420605 | 0.517241 |
| truncate 75% | 0.671982 | 0.438697 | 0.506386 |
| independent Bag | 0.672006 | 0.524053 | 0.560983 |

### Test

| 历史模式 | Log-loss ↓ | AUROC ↑ | Balanced accuracy ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| full | 0.668603 | 0.545218 | 0.543049 | 0.218750 |
| current-only | 0.660399 | 0.557146 | 0.562676 | 0.390805 |
| truncate 25% | 0.664049 | 0.567990 | **0.579050** | 0.460000 |
| truncate 50% | 0.674185 | 0.586424 | 0.492626 | **0.539683** |
| truncate 75% | 0.670138 | 0.576664 | 0.527868 | 0.187500 |
| independent Bag | **0.652848** | **0.623075** | 0.560941 | 0.487395 |

按预先指定的主要指标 unweighted log-loss，independent Bag 最好；其 test log-loss 比 full 低 0.015755，即相对降低约 2.36%。AUROC 也由 0.545218 提高到 0.623075，绝对提高 0.077857。

阈值相关指标存在较明显不稳定性。例如 truncate 50% 在 test 中几乎把所有样本判为 1 类，其 balanced accuracy 只有 0.492626；因此理论判断应继续以不依赖分类阈值的 log-loss 为主，AUROC 为次要指标。

## 3. Test 逐主体配对比较

统一定义：

\[
\Delta_{LL}=LL_{control}-LL_{full}.
\]

正值表示 full 更好，负值表示对照更好。置信区间通过 subject-level bootstrap 100,000 次获得；双侧 p-value 使用 100,000 次配对 sign-flip permutation；五项主要比较使用 Benjamini–Hochberg FDR。

| 比较 | 平均 ΔLL | 95% CI | p | FDR q | 解释 |
|---|---:|---:|---:|---:|---|
| full vs current-only | -0.008204 | [-0.027235, 0.010993] | 0.4057 | 0.5071 | 无 full 优势 |
| full vs truncate 25% | -0.004554 | [-0.011957, 0.002646] | 0.2206 | 0.3677 | 无 full 优势 |
| full vs truncate 50% | +0.005582 | [-0.001278, 0.012303] | 0.1097 | 0.2742 | 不显著 |
| full vs truncate 75% | +0.001535 | [-0.004841, 0.007890] | 0.6392 | 0.6392 | 不显著 |
| full vs independent Bag | **-0.015755** | **[-0.026025, -0.005399]** | **0.00324** | **0.01620** | Bag 优于 full |

只有 full 与 independent Bag 的差异在当前 seed 下通过 FDR。负方向说明不是 full 优于 Bag，而是 Bag 的逐样本 log-loss 更低。full 只在 38.57% 的 test 样本上优于 Bag。

## 4. 历史长度曲线

从“只保留最后窗口”到完整历史，test log-loss 为：

```text
current-only  0.660399
truncate 25%  0.664049
truncate 50%  0.674185
truncate 75%  0.670138
full          0.668603
```

该曲线没有随着历史长度增加而稳定改善：加入历史先变差，随后局部回升，但 full 仍不优于 current-only 和 25% 历史。因此当前结果不支持“保留更多历史产生稳定增益”，也不支持递归状态传递有效。

## 5. 与常数概率基线比较

训练集 1 类比例为 0.378995，据此得到 test 常数概率基线 log-loss 0.663361。

- full：0.668603，比常数基线差 0.005242；
- current-only：0.660399，比常数基线好 0.002962；
- independent Bag：0.652848，比常数基线好 0.010513。

但 independent Bag 相对常数基线的逐样本改善 95% CI 为 `[-0.010066, 0.031689]`，双侧 p=0.3242，尚不显著。因此 Bag 虽显著优于 full，却还不能证明其具有稳定的绝对预测增益。

## 6. 理论判断

基于当前单个训练 seed，可以作出以下探索性判断：

1. **历史信息有用：暂不支持。** full 没有优于 current-only，差异方向反而偏向 current-only。
2. **更长历史带来稳定改善：不支持。** 截断曲线明显非单调。
3. **递归状态传递有效：暂不支持。** 不递归的 independent Bag 显著优于 full GRU。
4. **窗口集合可能比时间递归更有价值：存在候选信号。** Bag 在 validation/test 的 log-loss 和 AUROC 上均优于 full，但仍需多 seed 验证。
5. **不能据此判断时间顺序无用。** 当前尚未运行在窗口内容完全不变条件下的 shuffled GRU；Bag 优势只能否定当前 full GRU 的优势，不能单独分离“顺序”与“递归建模”的作用。

## 7. 下一步

在进入 shuffled 顺序实验前，应先用相同划分和配置补做 seed 43、44、45、46，至少重复 full、current-only 和 independent Bag；若计算预算允许，也重复三个 truncate 比例。多 seed 后需同时纳入样本不确定性和训练随机性，检查 Bag 优势是否稳定。

若 Bag 优势在多 seed 下保持，再实施多个固定 permutation seed 的 shuffled GRU：

- ordered full vs shuffled full：检验顺序贡献；
- ordered full vs independent Bag：检验递归有序建模是否优于无序窗口集合；
- shuffled full vs Bag：判断差异来自递归结构还是顺序本身。
