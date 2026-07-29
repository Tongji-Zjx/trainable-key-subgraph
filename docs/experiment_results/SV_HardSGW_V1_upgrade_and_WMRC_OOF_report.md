# SV-HardSGW 神经模块 V1 升级与 WMRC OOF 报告

## 1. 结论

V1 已完成程序实现、本地回归、服务器 CUDA 验收、候选筛选和 WMRC
3-fold OOF。

- V1A 的安全残差融合实现有效：无可靠增益时，GIN/Variation 门控保持接近
  零，最终模型不会明显低于静态锚点。
- V1B 的 residual attention 未产生可验证的最终增益，因此未被保留。
- V1A 的 WMRC pooled OOF AUROC 为 `0.554703`，低于对应的三分支
  late-fusion 基线 `0.563379`，差值为 `-0.008676`。
- V1A 的 fold AUROC 标准差由 `0.057178` 降至 `0.033540`，稳定性提高，
  但平均性能没有提高。
- V1A 的最终 OOF 排序几乎完全来自 static-spectral 锚点；残差专家只带来
  `+0.000220` pooled AUROC。

因此，V1 当前解决了“融合负迁移的安全性”，但没有解决“神经分支提供稳定
增益”。按照冻结升级路线，当前不能直接进入 V2；下一项确认工作应是等待
ADHD 基线 OOF 完整结束后，复用其 fold-local 产物运行冻结的 V1A。

## 2. 已实现架构

### V1A：Static Anchor + Zero-output Residual Experts

训练分为两阶段：

1. 仅训练 static-spectral 投影与分类头，并由 inner-validation 选择静态锚点；
2. 冻结静态锚点，训练 SignedGIN 与 Variation 残差专家。

最终 logits：

\[
\ell
=
\ell_s
+g_G\Delta\ell_G
+g_V\Delta\ell_V,
\qquad
g_G,g_V\ge 0.
\]

实现包括：

- GIN/Variation 输出层零权重、零偏置初始化；
- 残差门初始 logit 为 `-6`；
- 门控向零收缩；
- epoch 0 静态锚点作为合法 checkpoint 候选；
- 静态锚点在残差阶段完全冻结；
- V1A/V1B 按组件确定性初始化，保证配对比较；
- static 阶段跳过冻结 GIN 前向，降低计算量。

### V1B：Residual Node Attention

在 V1A 上加入：

\[
z_G
=
z_{\mathrm{mean+std}}
+g_A P_A(z_{\mathrm{att}}).
\]

Attention 投影采用零输出初始化，门控同样从近零开始。V1B 与 V1A 除
attention 残差外保持一致。

## 3. 验收结果

### 软件测试

- 本地全量回归：`344` 项通过，`1` 项跳过；
- 云端 V1 相关测试：`18` 项全部通过；
- CUDA smoke：V1B 静态阶段和残差阶段均完成；
- signed edge、variable-length、排列不变、梯度、保存/加载和 OOF
  覆盖检查均通过。

### 16 样本记忆

| 候选 | AUC | BA | Accuracy | 最佳阶段 |
|---|---:|---:|---:|---|
| V1A | 0.953125 | 0.875000 | 0.875000 | static anchor |
| V1B | 0.953125 | 0.875000 | 0.875000 | static anchor |

结果证明模型可训练且不会破坏锚点，但未达到完美记忆；残差专家也未在该
小样本任务上超过静态锚点。

## 4. Validation-only 候选冻结

筛选仅使用 WMRC fold 0 的 train/inner-validation，未读取 outer-test。

| 候选 | Final AUC | Composite AUC | GIN AUC | GIN归一化有效秩 | Fusion regret |
|---|---:|---:|---:|---:|---:|
| V1A | 0.639257 | 0.648743 | 0.538462 | 0.105231 | -0.000884 |
| V1B | 0.639257 | 0.648743 | 0.595049 | 0.127754 | -0.000884 |

V1B 虽提高了 GIN 分支 AUC 和有效秩，但：

- attention 归一化熵中位数为 `0.993657`，仍接近均匀分配；
- attention 屏蔽前后 final AUC 均为 `0.639257`；
- attention 门控为 `0.002142`；
- V1B final AUC 与 V1A 完全相同。

因此按“无验证增益不增加复杂度”的冻结规则，唯一 OOF 候选为 V1A。

## 5. WMRC 3-fold OOF

固定并复用：

- 原 fold assignments；
- fold-local selector；
- hard graph cache；
- train-only scaler；
- seed 42；
- 每折 inner-validation 阈值。

### 总体对比

| 指标 | Late-fusion 基线 | V1A | V1A − 基线 |
|---|---:|---:|---:|
| Pooled OOF AUROC | 0.563379 | 0.554703 | -0.008676 |
| Site-stratified AUROC | 0.559149 | 0.553777 | -0.005372 |
| Mean fold AUROC | 0.561294 | 0.555648 | -0.005645 |
| Fold AUROC SD | 0.057178 | 0.033540 | -0.023638 |
| Accuracy | 0.564103 | 0.525641 | -0.038462 |
| Balanced Accuracy | 0.541099 | 0.539177 | -0.001922 |
| F1 | 0.430622 | 0.529946 | +0.099324 |

### 各折 AUROC

| Fold | Late-fusion 基线 | V1A | 差值 |
|---:|---:|---:|---:|
| 0 | 0.586045 | 0.546228 | -0.039817 |
| 1 | 0.482251 | 0.520099 | +0.037848 |
| 2 | 0.615584 | 0.600618 | -0.014966 |

V1A 提高了原本最弱的 fold 1，但降低了 fold 0 和 fold 2，因此方差降低而
总体 AUROC 下降。

### 残差专家的实际贡献

对同一批 546 个 OOF 样本直接汇总：

| 输出路径 | Pooled AUROC |
|---|---:|
| V1A final | 0.554703 |
| Static-spectral anchor | 0.554483 |
| GIN expert | 0.524544 |
| Variation expert | 0.531943 |

Final 与 static 概率的平均绝对差仅为 `0.000332`，最大绝对差为
`0.001220`。这说明安全门控避免了负迁移，但神经残差几乎没有参与最终
判断。

## 6. 闸门判定

| 闸门 | 结果 |
|---|---|
| V1A/V1B 表示与非负门控检查 | 通过 |
| V1B attention 有非冗余贡献 | 未通过 |
| V1A 相对 static 的 fusion regret ≤ 0.01 | 通过 |
| WMRC 相对当前 late-fusion 基线退化不超过 0.01 | 通过 |
| WMRC pooled OOF 有实际增益 | 未通过 |
| 神经残差对 final 有实质贡献 | 未通过 |
| 直接进入 V2 | 不允许 |

## 7. 下一步

1. 等待正在其他服务器运行的 ADHD 当前架构 OOF 完整结束；
2. 不观察中间 outer-fold 结果，不据此修改 V1A；
3. 复用 ADHD 的 fold-local selector、cache 和 scaler，运行已冻结 V1A；
4. 若 ADHD V1A pooled OOF 有预定义实际增益，且 WMRC 的 `-0.008676`
   仍在 `0.01` 容忍范围内，则 V1 跨数据集闸门可能通过；
5. 只有届时才进入 V2 edge-aware signed message passing；
6. 若 ADHD 也无增益，则应先修复“专家学习但门控无法形成有效残差”的瓶颈，
   不应继续堆叠 V2/V3。

## 8. 产物

- 代码分支：`codex/sv-neural-upgrade-v1`
- V1A 实现提交：`6dd7a5b`
- V1B 实现提交：`ed377fa`
- 候选冻结工具提交：`aa18c2d`
- 本地归档：
  `analysis_artifacts/sv_neural_upgrade_v1_wmrc_20260730/`
- 归档 SHA-256：
  `10871e987a5d527139d3bb415bd37ccc8d4bc6fca196b0977c72a36a7cba5f20`
