# SV-HardSGW SVG 默认架构决策

更新日期：2026-07-31

## 1. 决策

项目的研究默认分类架构由 `S`（Static-spectral only）调整为
`SVG`（Static-spectral + Variation + SignedGIN）。

对应代码变体：

```text
signed_gin_multibranch_late_fusion
```

上游继续使用原始 `current` selector。`full_soft_hard` selector 保留为理论
对齐消融，不替代默认 selector。

`S` 继续保留为主要精简基线和删除消融，不删除其代码、checkpoint 或实验入口。

## 2. 默认流程

```text
带符号完整图序列
→ current selector
→ 冻结的带符号硬关键子图序列
→ ┌ Static-spectral 分支
  ├ Variation 动态演化分支
  └ SignedGIN 关键结构分支
→ 三分支独立分类头
→ 非负 softmax logit 后期融合
→ 二分类输出
```

### 2.1 Static-spectral

每个有效硬图窗口计算 signed-Laplacian 的 16 维固定谱分位表示，再跨有效
窗口聚合。该分支对应关键子图的静态谱状态。

### 2.2 Variation

对相邻有效窗口的 16 维谱状态计算绝对变化并聚合，形成 16 维动态演化表示。
该分支显式保留跨窗口演化信息。

### 2.3 SignedGIN

SignedGIN 从硬关键图重新构造的 15 维节点特征学习局部关键结构。正式默认配置为：

- signed-normalized 消息；
- 2 层、隐藏维数 64；
- residual connection；
- Jumping Knowledge；
- mean + std pooling；
- compact readout；
- SafeBatchNorm；
- 保留正边和负边的符号及幅值。

### 2.4 后期融合

三个分支分别投影到 16 维并输出二分类 logits。全局非负 softmax 权重完成
logit 后期融合；分支辅助分类损失权重为 `0.25`。

## 3. 选择依据

相同三折、selector、硬图缓存、train-only scaler、seed 和训练协议下：

| 数据集 | S OOF AUROC | SVG OOF AUROC | SVG − S |
|---|---:|---:|---:|
| WMRC | 0.566522 | 0.563379 | -0.003144 |
| ADHD | 0.535752 | 0.543843 | +0.008092 |

两数据集 AUROC 点估计的简单平均：

- S：`0.551137`
- SVG：`0.553611`
- SVG − S：`+0.002474`

SVG 的平均点估计略高，并显式同时表达静态谱状态、动态谱变化和关键子图神经
结构，因而更完整地对应谱–GW演化理论。基于“性能不明显受损且理论覆盖更完整”
的研究偏好，将 SVG 设为默认架构。

## 4. 解释边界

该决策不是“SVG 已显著优于 S”的统计声明：

- WMRC 上 S 的 AUROC 略高；
- ADHD 上 SVG 的 AUROC 略高；
- 两个数据集的 `SVG − S` 置信区间均跨越 0；
- 当前比较只包含一个训练 seed。

因此，论文和报告中应表述为：

> SVG 是理论契合优先的默认研究架构；S 是主要精简基线。当前 OOF 结果未检测
> 到两者分类性能存在显著差异。

## 5. 默认值与复现

- 新训练入口未显式指定 `--variant` 时使用 SVG。
- 新 cross-fit 入口未显式指定 `--variants` 时只训练 SVG。
- 显式指定 `static_spectral_only`、`signed_gin_static_variation` 等旧变体时，
  仍可复现对应消融。
- 已有 checkpoint 按其保存的完整模型配置加载，不受默认值调整影响。

