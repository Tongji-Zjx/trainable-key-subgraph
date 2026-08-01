# SVG-v2 五天精简方案：本地实现报告

## 1. 实现范围

本轮只增加独立 SVG-v2 实验能力，不修改下列冻结对象：

- current selector；
- 已有硬关键图缓存；
- 当前默认 SVG variant；
- 数据划分、标签和 outer-fold 定义；
- 带符号边存在与消息传播语义。

开发筛选仅允许读取 train 与 inner-validation。只有候选冻结后的
`confirmatory` 模式才生成 outer-test 预测。

## 2. 公共谱–扩散层

新增精确谱–扩散侧车缓存，逐窗口保存：

- 有符号正则化拉普拉斯特征值与特征向量；
- 六尺度 HKS：`0.1, 0.5, 1, 2, 5, 10`；
- 16维谱状态；
- 原始窗口与转移 mask；
- 源硬图、manifest、protocol、selector checkpoint 的 SHA256 来源信息。

训练阶段利用缓存特征分解计算精确 `exp(-tL)H`，不在每个 epoch
重新做特征分解，也不显式保存每个尺度的稠密热核矩阵。

所有 HKS 与谱差分标准化参数只由 train split 拟合。

## 3. 已实现候选

| 编号 | 实现 |
|---|---|
| A1 | 作者式训练配方：较低学习率、有效 batch 32、标签平滑、余弦调度、平衡采样 |
| B1 | 原15维节点特征加6维HKS |
| C3 | B1加三尺度精确热扩散消息：`0.5, 2, 10` |
| F1 | Static-spectral anchor 加零初始化门控神经残差 |
| G2 | 只作辅助监督的16维有符号谱方向预测头 |
| C3_F1 | 预定双组件组合 |
| C3_G2 | 预定双组件组合 |

G2目标不进入最终分类表示；缺失窗口会中断相邻转移，不会跨越 padding
或缺失时间点构造伪差分。

## 4. 实验入口

- `scripts/build_sv_spectral_diffusion_cache.py`：可恢复的谱–扩散缓存。
- `scripts/fit_sv_spectral_diffusion_scaler.py`：train-only标准化器。
- `scripts/run_svg_v2_five_day_fold.py`：单fold、可恢复的筛选或确认性运行。
- `scripts/summarize_svg_v2_screen.py`：ADHD与WMRC两折validation-only筛选。
- `scripts/summarize_sv_signed_gin_crossfit.py --run-name ...`：汇总候选目录的确认性OOF。
- `scripts/fit_evaluate_svg_v2_f0_fusion.py`：校准后的非负L1收缩F0融合。

## 5. 防泄漏与产物隔离

- 筛选模式不创建test谱缓存、不运行test评估。
- selector与基础硬图缓存只读复用。
- 侧车数据逐样本核对源硬图SHA256和完整样本集合。
- train-only scaler与split、protocol、selector来源绑定。
- F0拟合集和评估集必须完全不重叠。
- F0权重非负，无贡献分支允许收缩至接近0，不设置最小非零权重。
- F0阈值只由拟合集确定并冻结到评估集。

## 6. 本地验收

- SVG-v2针对性测试：44项通过。
- 项目全量单元测试：439项通过，1项按原条件跳过。
- `compileall`：通过。
- 新增命令行入口加载检查：通过。

以上结果只证明实现与数据协议在本地通过，不代表候选已获得分类性能增益；
增益必须由服务器上的冻结两折筛选和最终确认性OOF决定。
