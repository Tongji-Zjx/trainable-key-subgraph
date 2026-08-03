# 理论引导多视图关键子图网络：本地实现报告

## 1. 完成范围

本次实现不是最小可运行版本。正式架构中的以下路径均已进入真实数据、真实前向和梯度链路：

- 冻结 selector 的软图到硬图流程，以及逐窗口合并硬关键图；
- S 分支的 28 维稳定锚点、9 维基底不变谱初始化、6 维有符号边特征、双极性 Signed Spectral GCNII、mean/std/attention 池化和 16 维 `Q` decoder；
- V 分支的社区诱导连通分量对象、独立参数对象编码器、对象规模与正负密度上下文、signed diffusion-FGW 代价、UOT 对应、对象 coupling、单向门控残差 GRU 和 18 维 `Delta-Q / Delta-t` decoder；
- G 分支的完整图有符号神经编码与时间池化，且没有额外 decoder；
- S/V/G 关键通道的零/小门控残差融合和独立分类头；
- 冻结作者短期分支与冻结关键通道后的表示级残差融合，包括可训练 `P_ST`、`P_C`、门控和融合分类头；初始状态严格回退作者短期分支；
- 原始时间顺序、可变窗口数、可变节点数和可变对象数的 list-based batching；
- train-only scaler、不可变 artifact、SHA256 provenance、冻结 validation 阈值和 test 只读评估；
- Stage-0 对象/UOT/FGW/耗时/显存审计、表示秩、attention、signed message 和理论 decoder 诊断；
- 真实 UOT 对应与确定性 shuffled-correspondence 负对照。

## 2. 主要代码入口

- 特征与对象对应：`src/keysubgraph/features/multiview_critical.py`
- S/V/G 与最终融合模型：`src/keysubgraph/models/multiview_critical.py`
- artifact、scaler、Dataset/DataLoader：`src/keysubgraph/data/multiview_critical.py`
- 关键通道训练：`src/keysubgraph/training/multiview_critical_trainer.py`
- 缓存：`scripts/precompute_multiview_critical.py`
- scaler：`scripts/fit_multiview_critical_scaler.py`
- Stage-0 审计：`scripts/audit_multiview_critical_cache.py`
- 关键通道训练与评估：`scripts/train_multiview_critical.py`、`scripts/evaluate_multiview_critical.py`
- 表示诊断：`scripts/diagnose_multiview_critical.py`
- 作者短期融合训练与评估：`scripts/train_multiview_short_term_fusion.py`、`scripts/evaluate_multiview_short_term_fusion.py`
- 可断点续跑总入口：`scripts/run_multiview_critical_experiment.py`

## 3. 本地验收

- Python 3.7.16、PyTorch 1.13.1 环境下所有新增模块均通过语法与 CLI 导入检查；
- 新增 6 个定向单元测试全部通过，总耗时约 1 秒；
- 相关既有 selector、硬图特征、Proxy 和作者短期分支 19 个回归测试全部通过；
- ADHD 真实样本 CUDA smoke 成功：30 个窗口、29 个转移、104 个可变关键对象；
- 真实样本完成不可变缓存、train-only scaler 和完整 S/V/G CUDA forward，输出有限；
- dummy 流程完成分类损失、两个 decoder 损失、backward、checkpoint 保存和严格加载；
- 节点一致置换不改变图级预测，负边在缓存和前向中保持为负；
- 梯度到达 Signed Spectral GCNII 初始层、6 维边门控、对象编码器、对象上下文、`Q` decoder 和 `Delta-Q` decoder。

本地未执行全量缓存、长时间训练、16 样本长期过拟合或 OOF；这些属于服务器计算任务，不属于代码缺失。
