# 历史模式实验独立性审计

## 结论

现有代码与六份实际产物均不支持“后运行的实验继承了先前模型、优化器、预测或标签信息”的怀疑。可以高置信度排除直接的跨实验状态传递。

test AUROC 与执行先后顺序确实呈较强相关，但该现象只出现在 test AUROC，不出现在 validation AUROC 或主要指标 test log-loss。由于实验顺序同时也是模型条件顺序，且只有六个点和一个训练 seed，这种相关性不能作为信息传递证据，更可能来自模型差异、训练随机性或偶然的 test 排序波动。

仍建议补做“末尾重放 full seed 42”作为直接的顺序效应阴性对照，因为当前 checkpoint 没有记录 Git commit、PyTorch/CUDA 版本和完整 RNG 状态，单靠现有产物不能证明跨时间的软件环境完全相同。

## 1. 训练进程与模型初始化

每次命令均启动新的：

```text
python -u scripts/train_baseline.py ...
```

因此 Python 对象、CUDA 张量、DataLoader worker、优化器和随机数生成器不会跨命令保留。

训练入口在构造模型前调用 `set_baseline_seed(seed)`，重置 Python、NumPy、PyTorch CPU 和全部 CUDA seed。随后直接调用 `SignedSequenceBaseline(BaselineModelConfig(...))` 创建新模型。训练入口没有 `--resume` 参数，也没有读取其他实验 checkpoint 的代码路径。

在相同 seed 42 下分别构建六种历史模式，初始化后的 25 个 state tensors 逐元素完全相同。历史模式只改变 forward 中允许使用哪些窗口，以及是否递归更新状态，不改变初始化权重来源。

## 2. 优化器取证

训练器每次调用都执行：

```python
optimizer = torch.optim.AdamW(model.parameters(), ...)
```

且不会加载 optimizer state。train 分区 657 个样本、batch size 4，因此每个 epoch 有 165 次 optimizer step。实际最佳 checkpoint 中的 optimizer step 为：

| 实验 | 最佳 epoch | 实际 optimizer step | 从零训练的预期 step |
|---|---:|---:|---:|
| full | 9 | 1485 | 1485 |
| current-only | 24 | 3960 | 3960 |
| truncate 25% | 8 | 1320 | 1320 |
| truncate 50% | 2 | 330 | 330 |
| truncate 75% | 2 | 330 | 330 |
| independent Bag | 9 | 1485 | 1485 |

六项均精确相等。如果后一个实验继承前一个 optimizer，step 必然包含此前累计值。本结果直接排除了 optimizer 续训。

## 3. Checkpoint 与输出目录隔离

训练器为每个命令创建新的模型和输出路径，并在目标目录已存在 `history.json`、`best_checkpoint.pt` 或 `last_checkpoint.pt` 时直接抛出 `FileExistsError`，不会覆盖或接续旧实验。

六份最佳 checkpoint SHA-256 前缀分别为：

| 实验 | SHA-256 前缀 |
|---|---|
| full | `c66bc542c4c0` |
| current-only | `a9a25d719f2a` |
| truncate 25% | `a82b8737e3da` |
| truncate 50% | `45d87cd69baa` |
| truncate 75% | `67e707f8c02a` |
| independent Bag | `c34f20ce9489` |

所有 checkpoint 都有相同的 25 个参数/缓冲区名称与 shape，但后五项相对 full 的 25 个 tensors 无一逐元素完全相同。这符合“相同初始化、独立训练后因 forward 条件不同而分化”，不符合直接复制 checkpoint。

## 4. 数据与划分隔离

六份 checkpoint 的以下标识完全相同：

- 父 manifest SHA-256：`cc657fd71a98692e169c6a16d1edb8d26bdeb6fb63a958ec819d6e82861a15e8`
- 下游 split JSON SHA-256：`52ab6327cb29a8420d4b6fdf3c54c11033935b4e23a3ae9ac153ba79a5d7b3c2`
- test manifest SHA-256：`7dfde4913250f1cfd91102488e721bf981e403f514baeb6d27a8a9cf6d7a2852`

六个 test 文件均包含完全相同的 140 个 sample key、label、subject 和 site。模型之间的差异不是由样本集合变化造成的。

Dataset 只读取冻结 manifest、硬子图 JSON 和原始 `.pt`；训练过程没有写回这些文件。训练 DataLoader 为每个新进程创建新的 `torch.Generator` 并设置同一 seed 42。相同的训练样本顺序是受控公平条件，不是跨实验状态延续。

## 5. Validation/test 使用路径

训练期间只使用：

- train：更新模型；
- validation：选择最佳 checkpoint 和分类阈值。

test 不进入 `train_baseline.py`。独立评估脚本读取冻结 checkpoint 后执行：

```python
model.eval()
parameter.requires_grad_(False)
torch.no_grad()
```

评估只把指标与逐样本预测写入当前实验目录，不修改 checkpoint、manifest 或后续模型。每份 validation/test JSON 中记录的 checkpoint SHA-256 均与其所在目录的 `best_checkpoint.pt` 完全匹配。

AUROC 由连续概率直接计算，与 validation 选择的硬分类阈值无关。因此阈值不能解释“后续 test AUROC 上升”。

## 6. 执行顺序现象

按实际时间顺序：

```text
full → current-only → truncate 25% → truncate 50% → truncate 75% → independent Bag
```

Spearman 顺序相关为：

| 指标 | Spearman ρ | 六点精确双侧 p |
|---|---:|---:|
| validation AUROC | 0.0857 | 0.9194 |
| test AUROC | 0.9429 | 0.0167 |
| validation log-loss | -0.7714 | 0.1028 |
| test log-loss | -0.0857 | 0.9194 |

所以用户观察到的 test AUROC 趋势确实存在，不应否认。但它没有在 validation AUROC 上重现，也没有出现在预先指定的主要指标 test log-loss 上。truncate 75% 的 test AUROC 还低于 truncate 50%，并非严格逐次提高。

此外，执行顺序不是随机安排：它同时对应不同历史条件，最后运行的 independent Bag 与此前递归模型在数学结构上不同。因此“顺序相关”无法区分模型条件效应和偶然波动，不能反推存在进程间信息传递。

## 7. 仍存在的复现性边界

当前代码固定了常见随机源并设置 cuDNN deterministic，但没有：

- 调用 `torch.use_deterministic_algorithms(True)`；
- 在 checkpoint 中记录完整 RNG state；
- 记录 Git commit、PyTorch/CUDA/cuDNN 和 GPU 型号；
- 对同一模式执行训练后重放测试。

这意味着不能仅凭现有产物保证同一 seed 在 RTX 5090 上逐 bit 重现。潜在 CUDA 数值非确定性会造成独立训练分化，但它不是“从先前实验传递信息”。

## 8. 建议的直接排除实验

在所有历史实验之后，以相同 seed 42、相同配置和新目录重新运行一次 full：

```text
key_full_seed42_replay_after_all_v1
```

然后比较原 full 与 replay：

- history 每个 epoch 的指标；
- best epoch；
- 140 个 test 概率的最大绝对差；
- log-loss 和 AUROC。

若完全相同或仅有浮点级差异，即可直接排除运行顺序效应。若出现明显差异，应连续再运行两个 replay，并检查软件版本和 CUDA 确定性；这时首先应解释为训练非确定性或代码/环境版本差异，而不是默认存在跨实验信息泄漏。

后续多 seed 实验应随机化或平衡模型运行顺序，避免“模型条件”和“时间顺序”再次混杂。
