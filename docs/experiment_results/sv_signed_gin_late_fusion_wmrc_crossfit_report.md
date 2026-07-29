# SV-HardSGW SignedGIN 后期融合改进与 WMRC 交叉拟合报告

## 1. 结论

本轮改进取得了两个不同层面的结果：

1. **GIN 低秩化问题已被稳定缓解。** 紧凑 `mean+std` 读出、残差与
   Jumping Knowledge、train-only BatchNorm 显著提高了 GIN 表示的有效秩，
   三个外折均通过预设表示闸门。
2. **融合负迁移没有被跨折稳定解决。** 单次划分的 validation 闸门通过，
   但 3-fold OOF 的融合 AUROC 仅为 `0.5634`；只读分支分析显示，
   static-spectral 分支的 pooled OOF AUROC 为 `0.5827`，高于融合模型。

因此，当前改进版证明了 GIN 编码器修复有效，但**不应直接替代原模型成为正式默认架构**。
下一轮应保留修复后的 GIN，并重新设计“静态主干 + 零初始化残差专家”的安全融合方式。

## 2. 实现与验收

新增模型变体：

```text
signed_gin_multibranch_late_fusion
```

主要修改：

- static-spectral、variation、GIN 三分支独立投影和辅助分类；
- 非负 softmax logit 融合；
- Signed GIN 使用带符号归一化邻接、残差和 Jumping Knowledge；
- 节点读出使用 `mean + std`；
- 每窗口压缩到 32 维，跨窗口 `mean + std` 形成 64 维紧凑表示；
- GIN 表示及投影使用只由训练批次更新的 SafeBatchNorm；
- validation/test 只使用冻结 running statistics。

验收结果：

- 本地全量单元测试：336 项通过，1 项跳过；
- 服务器 CUDA 回归通过；
- 16 样本记忆实验：AUROC、BA、Accuracy 均为 `1.0000`；
- 单划分 validation 四项预设闸门全部通过；
- test 未参与 checkpoint、阈值、分支或架构选择。

代码版本：

- 本地分支：`codex/sv-late-fusion-v1`
- 本地提交：`e2b1c25`、`47f383a`
- 服务器等价提交：`2c32890`

## 3. 单次划分筛选结果

旧 SG2 与改进版使用相同 selector、硬图、scaler 和数据划分。

| Split | 模型 | AUROC | Site-AUC | Composite AUC | BA | Accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | 旧 SG2 | 0.5872 | 0.5659 | 0.5765 | 0.6328 | 0.6585 | 0.5333 |
| Validation | 改进版 | **0.6584** | **0.6927** | **0.6755** | **0.6371** | 0.6341 | **0.6053** |
| Test | 旧 SG2 | **0.6036** | **0.6130** | **0.6083** | **0.5863** | **0.6220** | 0.4364 |
| Test | 改进版 | 0.5824 | 0.5696 | 0.5760 | 0.5550 | 0.5610 | **0.5000** |

单划分 validation 的 composite AUC 提升 `+0.0990`，但 test AUROC 下降
`-0.0213`，说明 validation 增益不能直接解释为稳定泛化增益。

## 4. WMRC 3-fold 完整交叉拟合

每个外折均重新训练 selector、重新导出硬图、重新拟合 train-only scaler，
并且只由该折 inner-validation 选择 checkpoint 和阈值。546 个样本均恰好得到
一次外折预测。

### 4.1 正式 OOF 主结果

| 指标 | 结果 |
|---|---:|
| Pooled OOF AUROC | 0.5634 |
| Site-stratified OOF AUROC | 0.5591 |
| Mean fold AUROC | 0.5613 ± 0.0572 |
| Accuracy | 0.5641 |
| Balanced Accuracy | 0.5411 |
| F1 | 0.4306 |
| Sensitivity | 0.3879 |
| Specificity | 0.6943 |

混淆矩阵为 `[[218, 96], [142, 90]]`。模型对 1 类的召回明显不足。

补充的只读分层 bootstrap（10,000 次，随机种子 `20260729`；546 个
`sample_key` 均唯一）给出 pooled AUROC 95% CI：

```text
[0.5149, 0.6108]
```

同一随机种子下 10,000 次标签置换的单侧 `p=0.0071`。这些为事后补充统计；
它们说明 OOF 排序信号可检测且高于随机，但效应较弱，不能据此声称已经达到
强分类性能。

### 4.2 各外折

| Fold | Selector最佳AUC | Inner Val AUC | Outer AUROC | Outer Site-AUC | BA | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6322 | 0.7162 | 0.5860 | 0.5775 | 0.5657 | 0.5769 | 0.4967 |
| 1 | 0.6879 | 0.6693 | 0.4823 | 0.4682 | 0.4628 | 0.5000 | 0.2720 |
| 2 | 0.5933 | 0.5455 | 0.6156 | 0.6081 | 0.5939 | 0.6154 | 0.5000 |

三个外折阈值分别为 `0.4776`、`0.6003`、`0.5982`。Inner-validation 与
outer-test 的表现差异较大，尤其 fold 1 出现明显反转，表明划分敏感性和校准
不稳定仍然存在。

## 5. GIN 表示修复是否成功

旧 WMRC 架构中：

| 表示 | 归一化有效秩 | 平均余弦 |
|---|---:|---:|
| GIN representation | 0.059 | 0.9989 |
| GIN projection | 0.115 | 1.0000 |

修复后的三个外折 validation：

| Fold | 原始GIN秩 | 原始GIN余弦 | GIN投影秩 | GIN投影余弦 |
|---:|---:|---:|---:|---:|
| 0 | 0.1440 | 0.9760 | 0.4506 | 0.4109 |
| 1 | 0.1519 | 0.9785 | 0.3962 | 0.2477 |
| 2 | 0.1570 | 0.9884 | 0.4229 | 0.5427 |

原始 GIN 表示的归一化有效秩在三折中均超过预设下限 `0.10`，投影后的样本间
余弦也远低于 `0.995`。因此，GIN 低秩化修复不是单划分偶然现象。

## 6. 融合负迁移是否解决

没有稳定解决。

### 6.1 融合权重仍接近均匀

| Fold | GIN | Static-spectral | Variation |
|---:|---:|---:|---:|
| 0 | 0.3658 | 0.3169 | 0.3173 |
| 1 | 0.3468 | 0.3310 | 0.3223 |
| 2 | 0.3716 | 0.3302 | 0.2981 |

即使某个分支明显较弱，融合器也没有把其权重压到接近零。例如 fold 1 的
inner-validation GIN AUROC 为 `0.7356`，static 与 variation 仅为
`0.4819/0.5013`，但后二者仍获得合计约 `65%` 的权重。

### 6.2 OOF 冻结分支只读比较

下表只使用已经冻结的外折预测，不重新训练，也不在 test 上拟合阈值：

| 路径 | Pooled OOF AUROC | Mean fold AUROC ± SD |
|---|---:|---:|
| Fusion | 0.5634 | 0.5613 ± 0.0572 |
| GIN | 0.5303 | 0.5309 ± 0.0622 |
| Static-spectral | **0.5827** | **0.5792 ± 0.0220** |
| Variation | 0.5290 | 0.5327 ± 0.0296 |

Static-spectral 不仅总体更高，而且折间波动更小。融合器在 fold 0/2 略优于
static，但在 fold 1 严重退化，导致 pooled OOF 被拉低。

进一步用 inner-validation 选择“最佳冻结路径”后，pooled outer AUROC 仅为
`0.5302`。这说明问题不只是缺少一个简单的分支选择规则；inner-validation
中的分支排名本身也不稳定。

## 7. 最终判断

本轮实验支持以下判断：

- **已解决：** GIN 表示坍缩和低秩化；
- **部分改善：** 单次划分中的多分支互补；
- **未解决：** 跨划分稳定融合、selector 稳定性以及分类器过拟合；
- **不能成立：** “当前后期融合版稳定优于旧 SG2”；
- **可以成立：** “修复后的 GIN 能学习非坍缩表示，但其信号尚未被稳定转化为
  外层分类增益”。

当前版本应保留为实验分支，不建议升为正式默认模型。

## 8. 下一步架构方向

建议保留本轮 GIN 修复，仅重构融合层：

1. 以较稳定的 static-spectral 分支作为主干；
2. GIN 与 variation 改为**零初始化残差专家**，初始模型严格等价于静态主干；
3. 残差门控默认收缩到零，只有在训练证据充分时才增加贡献；
4. 避免三个全局 softmax 权重被约束为近均匀分配；
5. 继续以完整交叉拟合 OOF 而非单次 validation 作为主要验收标准。

这一设计直接针对当前证据：既保留已经修复的 GIN 信息，又保证动态分支无法在
没有增益时破坏静态主干。

## 9. 产物

本地结果根目录：

```text
analysis_artifacts/sv_signed_gin_late_fusion_wmrc_20260729/
```

关键正式产物：

```text
outputs/sv_signed_gin_crossfit/
  wmrc_late_fusion_3fold_seed202607_v1/
    summary_late_fusion_v2/summary.json
    summary_late_fusion_v2/summary.md
    summary_late_fusion_v2/oof_predictions.csv
```

结果包 SHA256：

```text
3c161ab144481b676d4653c2f43b35183f13446fc5505ac01cd74dfe79da9824
```
