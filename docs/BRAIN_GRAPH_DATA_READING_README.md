# Brain Graph `.pt` 数据读取规范

本文档面向只需要**读取数据**的项目成员，不要求使用原训练脚本。  
下面总结 `.pt` 数据的目录组织、字段规范、标签读取方式，以及一个独立的 PyTorch Dataset 读取方式。

---

## 1. 数据类型

数据分为两类：

### 1.1 短期动态图数据 Local Dynamic Graph

每个被试对应一个动态图序列：

```text
adjacency: (T, N, N)
```

含义：

- `T`：时间窗口数量；
- `N`：ROI 数量；
- `adjacency[t]`：第 `t` 个时间窗口的 ROI × ROI 功能连接矩阵。

推荐 `.pt` 字段：

```python
{
    "adjacency": Tensor,             # required, shape: (T, N, N)
    "community_sequence": Tensor,    # optional, shape: (T, N)
    "coords": Tensor                 # optional, shape: (N, 3)
}
```

---

### 1.2 全局静态图数据 Global Static Graph

每个被试对应一张全局静态脑功能图：

```text
adjacency: (N, N)
```

含义：

- `N`：ROI 数量；
- `adjacency`：该被试的全局静态功能连接矩阵。

推荐 `.pt` 字段：

```python
{
    "adjacency": Tensor,            # required, shape: (N, N)
    "community_labels": Tensor,     # optional, shape: (N,)
    "coords": Tensor,               # optional, shape: (N, 3)
    "node_names": list[str],        # optional
    "global_threshold": float,      # optional
    "graph_density": float,         # optional
    "t_r": float,                   # optional
    "n_time_points": int            # optional
}
```

如果 Global 的 `adjacency` 被保存成 `(1, N, N)`，读取时可以取：

```python
adjacency = adjacency[0]
```

但更推荐直接保存成 `(N, N)`。

---

## 2. 目录结构

如果只读取一种数据，可以单独准备一个目录。

例如只读 Global：

```text
global_static_graphs/
    0/
        sub001.pt
        sub002.pt
    1/
        sub101.pt
        sub102.pt
```

例如只读 Local：

```text
local_dynamic_graphs/
    0/
        sub001.pt
        sub002.pt
    1/
        sub101.pt
        sub102.pt
```

如果需要同时读取 Local 和 Global，两个目录必须保持相同的相对路径：

```text
local_dynamic_graphs/
    0/
        sub001.pt
    1/
        sub101.pt

global_static_graphs/
    0/
        sub001.pt
    1/
        sub101.pt
```

读取时会用相对路径匹配：

```text
local_dynamic_graphs/0/sub001.pt
global_static_graphs/0/sub001.pt
```

这两个文件表示同一个被试的 Local 和 Global 数据。

---

## 3. 标签读取方式

支持两种方式。

### 3.1 从目录名推断标签

如果文件路径中包含目录 `0` 或 `1`，可以直接把它作为 label。

例如：

```text
global_static_graphs/0/sub001.pt -> label = 0
global_static_graphs/1/sub101.pt -> label = 1
```

一般可以约定：

```text
0 = Control
1 = ADHD
```

具体含义以项目设定为准。

---

### 3.2 从 CSV 读取标签

也可以使用一个 CSV 文件显式指定标签：

```csv
file,label
sub001.pt,0
sub101.pt,1
```

注意：这里的 `file` 建议使用文件名，例如 `sub001.pt`，而不是完整路径。

---

## 4. 读取后得到的数据格式

### 4.1 只读取 Global

单个样本返回：

```python
{
    "global_adj": Tensor,       # (N, N)
    "global_coords": Tensor,    # (N, 3) or None
    "global_comm": Tensor,      # (N,) or None
    "label": Tensor,            # scalar
    "file": str
}
```

DataLoader batch 后：

```python
batch["global_adj"]     # (B, N, N)
batch["global_coords"]  # (B, N, 3) or None
batch["global_comm"]    # (B, N) or None
batch["label"]          # (B,)
batch["file"]           # list[str]
```

---

### 4.2 只读取 Local

单个样本返回：

```python
{
    "local_adj": Tensor,        # (T, N, N)
    "local_coords": Tensor,     # (N, 3) or None
    "local_comm": Tensor,       # (T, N) or None
    "local_len": int,           # T
    "label": Tensor,            # scalar
    "file": str
}
```

DataLoader batch 后：

```python
batch["local_adj"]      # (B, T_max, N, N)
batch["local_coords"]   # (B, N, 3) or None
batch["local_comm"]     # (B, T_max, N) or None
batch["local_len"]      # (B,)
batch["label"]          # (B,)
batch["file"]           # list[str]
```

其中 `T_max` 是当前 batch 内最大的时间窗口数量，较短序列会 padding。

---

### 4.3 同时读取 Local + Global

单个样本返回：

```python
{
    "local_adj": Tensor,        # (T, N, N)
    "local_coords": Tensor,     # (N, 3) or None
    "local_comm": Tensor,       # (T, N) or None
    "local_len": int,

    "global_adj": Tensor,       # (N, N)
    "global_coords": Tensor,    # (N, 3) or None
    "global_comm": Tensor,      # (N,) or None

    "label": Tensor,
    "file": str
}
```

DataLoader batch 后：

```python
batch["local_adj"]      # (B, T_max, N, N)
batch["local_coords"]   # (B, N, 3) or None
batch["local_comm"]     # (B, T_max, N) or None
batch["local_len"]      # (B,)

batch["global_adj"]     # (B, N, N)
batch["global_coords"]  # (B, N, 3) or None
batch["global_comm"]    # (B, N) or None

batch["label"]          # (B,)
batch["file"]           # list[str]
```

---

## 5. Padding 规则

不同被试的节点数或 Local 时间窗口数可能不同。读取时建议统一 padding。

### 5.1 节点 padding

如果当前样本节点数是 `n`，数据集最大节点数是 `N`：

```text
adjacency: (n, n) -> (N, N)
coords:    (n, 3) -> (N, 3)
comm:      (n,)   -> (N,)
```

padding 值：

```text
adjacency: 0.0
coords:    0.0
comm:      -1
```

### 5.2 Local 时间维 padding

如果当前 batch 最大时间窗口数是 `T_max`：

```text
local_adj:  (T, N, N) -> (T_max, N, N)
local_comm: (T, N)    -> (T_max, N)
```

padding 值：

```text
local_adj:  0.0
local_comm: -1
```

同时保留：

```python
local_len = T
```

用于记录每个样本真实时间长度。

---

## 6. 最小读取示例

见 `read_brain_graph_data_example.py`。该脚本提供：

- 单独读取 Global；
- 单独读取 Local；
- 同时读取 Local + Global；
- 自动标签推断；
- 可选 CSV 标签；
- batch padding。

基本使用方式：

```python
from read_brain_graph_data_example import BrainGraphDataset, brain_graph_collate
from torch.utils.data import DataLoader

dataset = BrainGraphDataset(
    local_dir="/path/to/local_dynamic_graphs",
    global_dir="/path/to/global_static_graphs",
    labels_csv=None
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=brain_graph_collate
)

for batch in loader:
    print(batch["local_adj"].shape)
    print(batch["global_adj"].shape)
    print(batch["label"].shape)
    break
```

---

## 7. 常见检查项

读取数据前建议检查：

1. Local 的 `adjacency` 是否为 `(T, N, N)`；
2. Global 的 `adjacency` 是否为 `(N, N)`；
3. Local 和 Global 是否使用相同 ROI 顺序；
4. 如果双通道读取，两个目录的相对路径是否一致；
5. adjacency 是否对称；
6. adjacency 对角线是否为 0；
7. label 是否与目录或 CSV 一致；
8. community 缺失时是否允许返回 `None`；
9. coords 缺失时是否允许返回 `None`。

---

## 8. 字段名总结

| 数据类型 | 字段名 | 形状 | 是否必需 |
|---|---|---:|---|
| Local | `adjacency` | `(T, N, N)` | 必需 |
| Local | `community_sequence` | `(T, N)` | 可选 |
| Local | `coords` | `(N, 3)` | 可选 |
| Global | `adjacency` | `(N, N)` | 必需 |
| Global | `community_labels` | `(N,)` | 可选 |
| Global | `coords` | `(N, 3)` | 可选 |
| Global | `node_names` | `list[str]` | 可选 |
| Global | `global_threshold` | `float` | 可选 |
| Global | `graph_density` | `float` | 可选 |
| Global | `t_r` | `float` | 可选 |
| Global | `n_time_points` | `int` | 可选 |
