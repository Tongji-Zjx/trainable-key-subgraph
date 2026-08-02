# SVG 三天精简改进：本地实现报告

## 1. 实现范围

本轮在不改变默认 `signed_gin_multibranch_late_fusion` 架构的前提下，增加三个相互独立的候选模块：

- `D1`：全局 mean/std 与社区层级 pooling 融合；社区编号只用于分组，不使用 community embedding。
- `H1`：训练期站点–类别平衡采样；站点不进入模型输入，validation/test 不使用站点生成预测。
- `E1`：固定三个硬图预算 `(0.35,0.20)`、`(0.50,0.30)`、`(0.65,0.40)`，共享同一 GIN，并对三预算 GIN 表示做等权平均；禁止学习预算注意力。

唯一允许的组合候选为 `D1_H1`。默认 SVG、既有缓存格式和既有实验入口均保留。

## 2. 数据与缓存兼容

- 新缓存保存裁剪后节点的社区标签以及节点/边预算。
- artifact 与 manifest schema 升至 v2，同时仍可读取 v1 缓存。
- D1 对节点排列和社区编号重标号保持不变。
- E1 要求三个预算的 sample、label、site、subject、protocol、selector 和 selection seed 完全一致。
- 每个预算使用自己的 inner-train-only scaler，不共享统计量。

## 3. 训练与评估约束

- selector checkpoint 固定复用，不重新训练。
- selector 的硬选择 seed 固定为 42，与下游分类器 seed 解耦。
- H1 不再叠加类别加权，避免重复重加权。
- 两折开发筛选只使用 inner-validation，不生成或读取 outer-test。
- 确认阶段使用 3 folds × seeds 42/43/44，并与同 seed 的默认 SVG 配对比较。
- 分类性能主指标固定为每个 seed 的 outer-fold AUROC 算术平均，再跨 seed 汇总；pooled OOF AUROC 仅作辅助诊断。

## 4. 新增入口

- `scripts/run_svg_three_day_fold.py`：单折、断点可恢复的统一入口。
- `scripts/launch_svg_three_day_screen.sh`：D1/H1/E1 两折筛选；设置 `INCLUDE_E1=0` 可在缓存耗时过高时跳过 E1。
- `scripts/launch_svg_three_day_combination.sh`：唯一组合 D1_H1 的两折筛选。
- `scripts/launch_svg_three_day_confirmatory.sh`：冻结候选与默认 SVG 的三折三 seed 配对确认。
- `scripts/summarize_svg_three_day_confirmatory.py`：以 mean-fold AUROC 为首要指标生成汇总。

## 5. 本地验收

局部与回归测试覆盖：

- D1 节点置换与社区重标号不变性；
- H1 可复现性、类别平衡与站点轮换；
- E1 三预算对齐、独立 scaler 与固定表示均值；
- signed edge、variable-length window 和既有 SVG 行为不变；
- 筛选阶段不读取 test；
- 确认性汇总将 mean-fold AUROC 标记为主指标。

本地不执行正式训练；正式运行仍应放在服务器 GPU 上完成。
