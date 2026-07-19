# OOF 关键结构传递实验：现有实现审计

## 审计范围

本审计依据 `docs/adopted_theory_and_oof_experiment_revision_final.md`，覆盖数据划分、提取器、硬子图导出、下游模型、控制子图、边扰动、统计和审计能力。

## 可直接复用

| 能力 | 现有实现 | 结论 |
|---|---|---|
| 样本索引与真实标签 | `data/sample_index.csv` 及 `sample_index.py` | 复用 |
| 基础 group-aware 分配算法 | `data_split.py` | 复用分配思想和校验函数 |
| 图序列 Dataset/DataLoader | `graph_dataset.py` | 复用 |
| Soft extractor 训练与 checkpoint | `soft_extractor.py`、`trainer.py`、`train_soft_extractor.py` | 复用 |
| 冻结提取器硬子图导出 | `hard_extractor.py`、`export_hard_subgraphs.py` | 复用 |
| Signed 子图编码 | `baseline_subgraph_encoder.py`、`signed_message_passing.py` | 复用 |
| Independent-bag 聚合 | `baseline_classifier.py` | 复用 |
| 下游训练、最佳 checkpoint 与阈值保存 | `baseline_trainer.py`、`train_baseline.py` | 复用 |
| 变长子图序列 batching | `baseline_collate.py` | 复用 |
| Random/Top-degree/Low-score 构造基础 | `analysis/controls.py`、`baseline_controls.py` | 复用 Random 生成逻辑 |
| 剂量扰动基础 | `edge_perturbation.py` | 复用并调整为冻结的 0/25/50 与五个随机重复 |
| SHA-256 与 immutable artifact | `data_split.py`、各 manifest 模块 | 复用 |
| BH-FDR | `analysis/statistics.py` | 复用 |

## 需要扩展

| 能力 | 当前缺口 | 扩展方向 |
|---|---|---|
| 外层划分 | 仅有 train/validation/test 三路划分 | 增加固定 5-fold subject-group-aware outer split |
| 内层划分 | 仅针对单个下游三路划分 | 增加每个 outer-dev 的固定 inner train/validation |
| 数据协议 | 当前协议只描述一个固定三路或 all 集合 | 增加 fold protocol 与 outer-test 隔离元数据 |
| Random 控制 | 当前一次构建 Key/Low/Top/Random 全来源 | 增加每折 Random-only 冻结 manifest 和共同 cohort 校验 |
| 剂量扰动 | 默认含 10%，单个随机删除顺序，删除数使用近似四舍五入 | 固定 0/25/50、至少保留一边、支持 5 个 random repeats，并对齐理论删除数 |
| 基线模型配置 | 编码器固定为 Signed | 增加 encoder_type 注册并兼容 checkpoint |
| 训练调度 | 单次 manifest 训练 | 增加 A–D × fold × seed 矩阵计划与命令生成 |
| 评估输出 | 主要输出聚合指标 | 增加逐样本概率、loss、fold、seed、subject/session 元数据 |

## 必须新增

| 模块 | 职责 |
|---|---|
| `data/crossfit_split.py` | 5 折 outer split、inner split、不可变 artifact 和校验 |
| `models/node_only_subgraph_encoder.py` | 无显式边消息传递的节点独立编码器 |
| `crossfit/model_matrix.py` | A–D 注册、配置冻结、运行计划和完整性检查 |
| `crossfit/oof_statistics.py` | DSC、SEG、TPA、D(r)、过原点剂量斜率和 subject bootstrap |
| `crossfit/audit.py` | 泄漏、OOF 唯一性、模型矩阵、扰动和 hash 审计 |
| 本地 dummy 验收脚本 | 无正式训练地串联 split、A–D forward、扰动、统计和审计 |

## 关键解释约束

1. Node-only 仍接收由原邻接派生的节点级结构摘要，只能称为“无显式边消息传递基线”。
2. TPA 是 OOF log-loss 的差中差，不是互信息或因果效应。
3. Targeted/Random 扰动必须复用 A 模型 checkpoint，不得为每个剂量重新训练。
4. 正剂量斜率只表示正的剂量加权趋势；只有 `0 < D(0.25) < D(0.50)` 才称为随剂量增强。
5. 正式 OOF 仅包含 A–D；Top-degree、Low-score、Current-only 和 embedding permutation 不进入本轮。

## 工作树保护

审计时工作树包含用户已有的未跟踪实验报告、分析脚本和压缩包。本轮实现不得删除、覆盖或批量暂存这些文件；只修改与 OOF 实现直接相关的文件。

## 本地与服务器边界

本地只完成实现、单元测试、dummy forward/backward、dummy OOF 汇总和小规模 bootstrap。Fold-specific extractor、真实 Fold 0 dry run、60 个正式下游模型及完整 5000 次真实 subject bootstrap 均留在服务器执行。
