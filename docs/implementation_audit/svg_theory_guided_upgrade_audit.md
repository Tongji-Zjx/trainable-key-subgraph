# SVG 理论引导升级实施审计

## 1. 审计范围

本审计对应 `SV_HardSGW_Theory_Guided_Neural_Upgrade_Stage0_to_Stage4_Implementation_Spec.md` 的 Step 1。当前阶段只检查既有实现，不改变现有模型、selector、缓存、训练入口或实验产物。

审计覆盖：

- `src/keysubgraph/features/sv_hard_graph_features.py`
- `src/keysubgraph/models/sv_signed_gin.py`
- `src/keysubgraph/training/sv_signed_gin_trainer.py`
- `src/keysubgraph/crossfit/sv_signed_gin_runner.py`
- SV 硬图 artifact、manifest、dataset 与 train-only scaler
- current selector 损失与 hard graph 导出接口
- Exact 18/34 维谱–GW特征实现
- OOF 汇总、阈值冻结及来源追踪

## 2. 当前默认路径

当前默认研究 variant 为：

```text
signed_gin_multibranch_late_fusion
```

实际流程为：

```text
动态图序列
  -> current selector（节点0.50、边0.30）
  -> 冻结硬关键图缓存
  -> SignedGIN / Static-spectral / Variation 独立分类头
  -> softmax非负logit后期融合
```

现有入口必须保持不变；Stage 0–4全部使用新 artifact type、schema、variant 和输出目录。

## 3. 可直接复用的实现

### 3.1 数据与硬图

- `ExactSTSEDataset` 已支持样本相关窗口数和节点数。
- downstream 使用 list-based batching，不截断图和窗口。
- 硬图缓存保存逐窗口邻接矩阵、15维节点特征、时间、window mask 和 transition mask。
- 硬邻接矩阵保留正负符号，边存在性由统一阈值确定。
- 硬图节点特征在裁剪后的图上重新计算，不复用完整图度数或社区强度。
- 节点时间差分优先使用稳定 `node_ids`，否则使用 `node_names`；缺失对应关系由有效性特征表达。

### 3.2 当前特征

- 节点特征：15维。
- Static：28维，其中前16维为窗口谱分位状态的时间均值，后12维为结构统计。
- Variation：16维，为相邻有效窗口谱分位绝对变化的均值。
- Exact-SGW基础实现已经输出：
  - 18维 `h_core = [mean(delta Q), mean(spectral speed), mean(GW speed)]`；
  - 16维绝对谱变化；
  - 合计34维分类表示。

Stage 0应复用 `SGWFeatureExtractor`、现有谱分位网格、signed Laplacian、heat kernel和Exact GW参数，不另建数学口径。

### 3.3 当前神经模型

当前 SignedGIN 已支持：

- signed weighted / signed normalized 消息；
- 两层图编码；
- residual；
- Jumping Knowledge；
- Mean+Std pooling；
- window mean或Mean+Std聚合；
- 分支独立分类头及非负logit融合。

它不满足 Stage 1 的部分包括：

- 正负边没有独立参数的消息MLP；
- 消息不显式读取6维边特征；
- 缓存没有逐边 `delta_A`、delta有效mask和同社区标志；
- 没有谱状态FiLM；
- 没有逐窗口Q或逐transition Gamma辅助监督；
- 没有EMA类别中心。

因此 N1 必须是新编码器，不能只修改现有 `SignedGINLayer` 的模式开关。

### 3.4 训练与OOF

当前训练器已具备：

- train-only类别权重；
- gradient accumulation；
- ROC-AUC和site-stratified AUC；
- composite AUC checkpoint；
- validation阈值拟合并冻结到outer-test；
- checkpoint provenance检查。

当前crossfit已经具备固定fold protocol、逐折selector、硬图缓存、train-only scaler、模型训练、outer-test评估和严格OOF覆盖检查，可扩展而不应重写。

## 4. 必须扩展的artifact与来源链

Stage 0/1不能静默修改现有 schema version 1。建议新增理论升级专用 sidecar，保持旧缓存可读：

```text
fold_k/
  cache/{train,validation,test}/              # 旧SV硬图缓存，不改
  theory_upgrade/
    edge_features/{split}/                    # 6维逐边特征及mask
    exact_core/{split}/                       # Q、Gamma、18维core
    stage0/                                   # full/hard配对诊断
```

每个sidecar必须至少绑定：

- sample key；
- split和fold；
- protocol SHA256；
- fold assignment SHA256；
- selector checkpoint SHA256；
- hard graph manifest SHA256；
- edge threshold；
- quantile grid；
- Laplacian eta；
- diffusion time；
- GW solver配置；
- feature schema SHA256；
- code commit。

任何来源不一致时拒绝训练或汇总。

## 5. 已冻结的实施定义

根据实施前确认，采用以下定义：

1. Stage 0理论主结果使用未标准化18维表示上的欧氏ground metric；inner-train z-score结果只作敏感性报告。
2. N0是当前修复后SignedGIN编码器加独立分类头，不包含Static-spectral或Variation。
3. FiLM位于JK投影之后、节点Mean+Std pooling之前。
4. Stage 2的T0/T1/T2加载相同Stage 1 checkpoint，并在完全相同训练协议下端到端微调神经编码器；selector和硬图保持冻结。
5. Stage 0–3严格复用已有current selector；Stage 4的R0–R3统一修复类别加权并使用有效batch至少8，旧current checkpoint作为外部冻结基线。
6. 站点探针主指标为balanced accuracy。
7. 明显下降定义为pooled或site-stratified OOF AUC下降超过0.01；配对bootstrap 95% CI跨0时只表述为趋势。
8. Stage 0主OT使用精确离散OT，不用熵正则结果替代理论主结果；缓存pairwise距离矩阵。

## 6. 需要特别处理的现有问题

### 6.1 Selector类别权重

current selector训练通常使用物理batch size 1。其当前weighted CE除以当前batch权重和，因此单样本batch下类别权重会被抵消。

处理原则：

- 不改动Stage 0–3使用的历史selector checkpoint；
- Stage 4统一修复R0–R3的类别加权；
- 报告中将历史current与修复后R0分别标识，禁止混称。

### 6.2 N0名称与当前variant不等价

当前没有纯SignedGIN-only正式variant：`signed_gin_variation`仍含Variation，默认SVG含三个分支。因此必须增加独立N0，不能用现有名称冒充。

### 6.3 现有Residual不等于Stage 3

已有static-anchor residual是分支logit相加。Stage 3要求：

```text
[neural 64; theory-core 16; static 16]
  -> joint residual head
  -> zero-initialized delta logits
  -> near-zero nonnegative gate
  -> frozen static logits + residual
```

必须使用新variant，不能改变旧residual variant语义。

### 6.4 精确OT与Bootstrap成本

10,000次bootstrap若每次重新构造特征和Exact GW不可接受。正确流程是：

1. 每个样本只计算一次full/hard Exact core；
2. 预计算类别所需距离矩阵；
3. bootstrap仅重采样索引并求离散OT；
4. 开发阶段2,000次，最终服务器报告10,000次。

## 7. Stage 0实现边界

Stage 0只允许：

- 读取冻结protocol、fold assignment、selector checkpoint和硬图；
- 计算full/hard的Q、Gamma与18维core；
- 计算类别间隔、paired radius、OT radius、理论下界；
- 执行分层bootstrap和误差分解；
- 生成不可变artifact和报告。

Stage 0禁止：

- 更新模型参数；
- 重新选择selector checkpoint；
- 用Proxy代替Exact GW；
- 使用validation/test拟合主定义；
- 实现或训练N1及后续模块。

## 8. Step 1结论

当前工程具有完成Stage 0–4所需的数据协议、硬图导出、Exact-SGW、训练和OOF骨架，但新路线不是配置层改动。最小安全实现顺序必须是：

```text
Stage 0 sidecar与诊断
  -> 服务器生成完整结果
  -> 审查理论闸门
  -> N0/N1及边特征sidecar
  -> N2/N3/N4
  -> Stage 1双数据集配对OOF
  -> 后续Stage按闸门推进
```

Step 1审计通过，可以进入Stage 0本地实现。
