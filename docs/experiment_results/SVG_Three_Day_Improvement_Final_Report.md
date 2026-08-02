# SVG 三日精简改进方案最终报告

## 1. 实验协议

- 数据集：ADHD、WMRC。
- 开发筛选：fold 0–1、seed 42，仅使用 inner-validation，不读取 outer-test。
- 筛选候选：D1（社区层级 pooling）、H1（站点–类别平衡采样）、E1（三预算 GIN 表示等权平均）。
- 唯一允许的组合：D1_H1；由于 D1 未进入候选池，按预注册规则未运行该组合。
- 冻结候选：依据 inner-validation 综合分数 \(J\) 从候选池中选择 H1。
- 确认实验：BASELINE 与 H1 在两个数据集上分别执行 3 folds × seeds 42/43/44；各折阈值只由该折 inner-validation 冻结后用于 outer-test。
- 主指标：每个 seed 的 outer-fold AUROC 算术平均，再跨 seed 汇总。Pooled OOF AUROC 仅作辅助诊断。

## 2. 两折开发筛选

| 候选 | ADHD Mean-fold AUC | ΔAUC | WMRC Mean-fold AUC | ΔAUC | 冻结分数 J |
|---|---:|---:|---:|---:|---:|
| D1 | 0.512235 | -0.029411 | 0.658267 | -0.034483 | -0.02655 |
| H1 | 0.558392 | +0.016746 | 0.661362 | -0.031388 | **+0.01277** |
| E1 | 0.560533 | +0.018888 | 0.660035 | -0.032714 | -0.00195 |

筛选阶段 H1 和 E1 进入低门槛候选池。H1 的综合分数最高，因此冻结 H1 进入严格确认；D1 未通过，故不追加 D1_H1 组合实验。

## 3. 三折三 seed 确认结果

### 3.1 ADHD

| 模型 | Mean-fold AUROC | Mean-fold Site-AUC | BA | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| BASELINE | **0.536341 ± 0.007409** | 0.497075 ± 0.003894 | **0.519711** | **0.499645** | **0.477299** |
| H1 | 0.502788 ± 0.008690 | **0.523071 ± 0.004364** | 0.490717 | 0.480810 | 0.436596 |

- H1 配对 Δmean-fold AUROC：**−0.033553 ± 0.005046**。
- H1 优于 BASELINE 的 seed 数：**0/3**。
- H1 的 Site-AUC 提高约 0.0260，但主 AUROC、BA、Accuracy 和 F1 均下降。

### 3.2 WMRC

| 模型 | Mean-fold AUROC | Mean-fold Site-AUC | BA | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| BASELINE | **0.559516 ± 0.008206** | **0.554916 ± 0.011198** | 0.527958 | 0.521368 | 0.503480 |
| H1 | 0.555267 ± 0.021571 | 0.547422 ± 0.025998 | **0.549308** | **0.550672** | **0.504275** |

- H1 配对 Δmean-fold AUROC：**−0.004249 ± 0.028623**。
- H1 优于 BASELINE 的 seed 数：**1/3**。
- H1 的 BA 和 Accuracy 有所提高，但主 AUROC、Site-AUC 与稳定性均未优于 BASELINE。

## 4. 最终结论

H1 在两折 inner-validation 筛选中的 ADHD 增益未能在严格三折三 seed OOF 中复现。其 ADHD 主指标显著下降，WMRC 主指标也略降。因此：

1. H1 **未通过最终确认**，不应替代当前 SVG BASELINE。
2. D1、E1 和 D1_H1 均不进入后续正式验证。
3. 当前默认架构保持 `signed_gin_multibranch_late_fusion`。
4. 本轮最可靠的发现是：站点–类别平衡采样可以改变阈值相关指标，但没有稳定提高跨折排序能力，不能仅凭 BA、Accuracy 或单次 Site-AUC 增益认定架构升级成功。

## 5. 权威产物

- ADHD：`outputs/svg_three_day_improvement/adhd_h1_confirmatory_v1/confirmatory_summary/summary.json`
- WMRC：`outputs/svg_three_day_improvement/wmrc_h1_confirmatory_v1/confirmatory_summary/summary.json`
- 两数据集筛选：`outputs/svg_three_day_improvement/two_dataset_screen_summary_v1/screen_summary.json`

