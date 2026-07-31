# SV-HardSGW 多尺度扩散几何与有符号谱方向扩展

## 1. 实现目标

本扩展在冻结硬关键图序列上补充两类理论特征：

1. 有符号谱演化方向；
2. 多尺度扩散几何。

Selector、硬图生成、原始 signed edge、Static-spectral、Variation 和
SignedGIN 均不被修改。新增特征以独立 sidecar 保存，因此旧缓存、旧模型和
旧 checkpoint 仍可直接读取。

在正式 OOF 结果证明新分支有效之前，默认架构仍为
`signed_gin_multibranch_late_fusion`（SVG），不自动替换。

## 2. 固定特征

### 2.1 有符号谱演化方向

对第 \(m\) 个有效硬关键图构造正则化 normalized signed Laplacian：

\[
L_{\eta}^{(m)}
=D_{\eta}^{-1/2}(D-A+\eta I)D_{\eta}^{-1/2},
\qquad
D_{ii}=\sum_j |A_{ij}|.
\]

使用与既有 Static-spectral 和 Variation 完全一致的 16 个谱分位点得到
\(Q^{(m)}\in\mathbb R^{16}\)。相邻有效窗口的方向变化为：

\[
\Delta Q^{(m)}=Q^{(m+1)}-Q^{(m)}.
\]

样本级方向特征为：

\[
H_{\mathrm{direction}}
=\operatorname{MeanMask}_m\Delta Q^{(m)}
\in\mathbb R^{16}.
\]

它与既有
\(\operatorname{MeanMask}_m|\Delta Q^{(m)}|\) 不同：新特征保留增加或
减少的方向，既有 Variation 保留变化幅度。缺失窗口会切断相邻转移，不会
跨越 padding 计算差分。

### 2.2 多尺度扩散几何

固定时间尺度：

\[
\mathcal T=\{0.25,0.50,1.00,2.00\}.
\]

对每个尺度构造热核：

\[
K_t=\exp(-tL_\eta),
\]

以及节点间扩散距离：

\[
d_t(i,j)=\|K_t(i,:)-K_t(j,:)\|_2.
\]

每个窗口、每个尺度对所有无序节点对汇总：

\[
[\operatorname{mean},\operatorname{std},
q_{0.10},q_{0.25},q_{0.50},q_{0.75},q_{0.90}].
\]

四个尺度拼接后得到 28 维窗口表示，再只对有效窗口取均值：

\[
H_{\mathrm{diffusion}}\in\mathbb R^{4\times7}
=\mathbb R^{28}.
\]

该表示保持节点一致置换不变。负边通过 signed Laplacian 保留；边存在仍按
\(|A_{ij}|>0\) 判断，不使用 \(A_{ij}>0\)。

## 3. 数据与泄漏控制

新增 sidecar 不重写原 SV hard-cache：

- `train_theory.pt`
- `validation_theory.pt`
- `test_theory.pt`
- `theory_scaler.json`

sidecar 绑定以下 provenance：

- 原 SV manifest SHA256；
- protocol SHA256；
- selector checkpoint SHA256；
- selection mode；
- selection seed；
- split。

均值和尺度只允许从 train sidecar 拟合。Validation 和 test 只能复用冻结的
train-only scaler。训练、评估和审计入口均会校验上述 hash 和样本键集合。

预计算按样本流式把图送入 CPU/GPU，不会把整个数据集同时复制到显存。

## 4. 新增模型变体

| 变体 | 新增分支 |
|---|---|
| `signed_gin_multibranch_spectral_direction` | 16 维谱方向 |
| `signed_gin_multibranch_diffusion_geometry` | 28 维多尺度扩散几何 |
| `signed_gin_multibranch_theory_geometry` | 两者同时加入 |

每个新分支采用独立投影和独立辅助分类头，再与 SVG 的
Static-spectral、Variation、SignedGIN logits 做非负 softmax 加权融合。
这样可以分别读取分支 AUROC 和融合权重，并完成严格删除消融。

建议正式比较顺序：

1. 原 SVG；
2. SVG + signed spectral direction；
3. SVG + multi-scale diffusion geometry；
4. SVG + 两者。

只有在相同 OOF folds 上稳定优于 SVG 时，才考虑更改默认架构。

## 5. 程序入口

- 固定特征：`src/keysubgraph/features/sv_theory_geometry.py`
- sidecar、scaler、Dataset：
  `src/keysubgraph/data/sv_theory_geometry.py`
- 预计算：`scripts/precompute_sv_theory_geometry.py`
- train-only scaler：`scripts/fit_sv_theory_geometry_scaler.py`
- 输入审计：`scripts/audit_sv_theory_geometry_inputs.py`
- 训练：`scripts/train_sv_signed_gin.py`
- 评估：`scripts/evaluate_sv_signed_gin.py`

## 6. 本地验收

已覆盖：

- 谱方向随时间反转而反号；
- 节点一致置换不改变谱方向或扩散几何；
- padding/缺失窗口不产生伪转移；
- 负边不被删除，也不被替换为正边；
- sidecar round-trip、hash/provenance 和 train-only scaling；
- 新分支 forward、loss、backward；
- 旧 SVG 配置兼容；
- 全项目回归测试。

