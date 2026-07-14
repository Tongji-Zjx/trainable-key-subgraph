"""
read_brain_graph_data_example.py

独立数据读取示例脚本。

用途：
1. 读取 Local 短期动态图数据：
   adjacency: (T, N, N)
   community_sequence: (T, N), optional
   coords: (N, 3), optional

2. 读取 Global 全局静态图数据：
   adjacency: (N, N)
   community_labels: (N,), optional
   coords: (N, 3), optional

3. 支持 Local-only、Global-only、Local+Global 三种模式。

该脚本不依赖原训练脚本，可以直接复制到其他项目中使用。
"""

import os
import glob
import csv
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import Dataset


def load_labels_csv(csv_path: str) -> Dict[str, int]:
    """
    读取标签 CSV。

    CSV 格式：
        file,label
        sub001.pt,0
        sub101.pt,1

    注意：
        这里默认使用文件名作为 key，而不是完整路径。
    """
    labels = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["file"]] = int(row["label"])
    return labels


def infer_label_from_path(path: str) -> int:
    """
    从路径中的目录名推断标签。

    例如：
        /data/global/0/sub001.pt -> 0
        /data/global/1/sub101.pt -> 1

    如果路径中没有 0 或 1 目录，则报错。
    """
    parts = os.path.normpath(path).split(os.sep)
    for part in reversed(parts[:-1]):
        if part in ("0", "1"):
            return int(part)
    raise ValueError(f"Cannot infer label from path: {path}")


def pad_square_adj(adj: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    padding 单张图邻接矩阵到 (target_n, target_n)。

    输入：
        adj: (n, n)

    输出：
        padded_adj: (target_n, target_n)
    """
    n = adj.shape[0]
    if n == target_n:
        return adj
    if n > target_n:
        return adj[:target_n, :target_n]

    out = torch.zeros((target_n, target_n), dtype=adj.dtype)
    out[:n, :n] = adj
    return out


def pad_dynamic_adj(adj: torch.Tensor, target_n: int) -> torch.Tensor:
    """
    padding 动态图邻接矩阵到 (T, target_n, target_n)。

    输入：
        adj: (T, n, n)

    输出：
        padded_adj: (T, target_n, target_n)
    """
    t, n, _ = adj.shape
    if n == target_n:
        return adj
    if n > target_n:
        return adj[:, :target_n, :target_n]

    out = torch.zeros((t, target_n, target_n), dtype=adj.dtype)
    out[:, :n, :n] = adj
    return out


def pad_coords(coords: Optional[torch.Tensor], target_n: int) -> Optional[torch.Tensor]:
    """
    padding ROI 坐标到 (target_n, 3)。

    如果 coords 为 None，则返回 None。
    """
    if coords is None:
        return None

    n = coords.shape[0]
    if n == target_n:
        return coords
    if n > target_n:
        return coords[:target_n]

    out = torch.zeros((target_n, coords.shape[1]), dtype=coords.dtype)
    out[:n] = coords
    return out


def pad_comm_1d(comm: Optional[torch.Tensor], target_n: int) -> Optional[torch.Tensor]:
    """
    padding Global community_labels 到 (target_n,)。

    padding 值为 -1。
    """
    if comm is None:
        return None

    n = comm.shape[0]
    if n == target_n:
        return comm
    if n > target_n:
        return comm[:target_n]

    out = torch.full((target_n,), -1, dtype=comm.dtype)
    out[:n] = comm
    return out


def pad_comm_2d(comm: Optional[torch.Tensor], target_n: int) -> Optional[torch.Tensor]:
    """
    padding Local community_sequence 到 (T, target_n)。

    padding 值为 -1。
    """
    if comm is None:
        return None

    t, n = comm.shape
    if n == target_n:
        return comm
    if n > target_n:
        return comm[:, :target_n]

    out = torch.full((t, target_n), -1, dtype=comm.dtype)
    out[:, :n] = comm
    return out


class BrainGraphDataset(Dataset):
    """
    独立脑图数据读取 Dataset。

    支持三种模式：
        1. local_dir only: 只读取 Local 短期动态图
        2. global_dir only: 只读取 Global 全局静态图
        3. local_dir + global_dir: 同时读取 Local 和 Global

    双通道模式下：
        local_dir 和 global_dir 必须具有相同的相对路径结构。
    """

    def __init__(
        self,
        local_dir: Optional[str] = None,
        global_dir: Optional[str] = None,
        labels_csv: Optional[str] = None,
    ):
        if local_dir is None and global_dir is None:
            raise ValueError("At least one of local_dir or global_dir must be provided.")

        self.local_dir = local_dir
        self.global_dir = global_dir
        self.labels_map = load_labels_csv(labels_csv) if labels_csv else {}

        self.has_local = local_dir is not None
        self.has_global = global_dir is not None

        self.samples = self._collect_samples()
        self.max_n = self._compute_max_nodes()

    def _collect_samples(self) -> List[Dict[str, Any]]:
        """
        收集样本路径。

        如果同时读取 Local 和 Global：
            以 Local 为主，要求 Global 中存在相同相对路径文件。
        """
        samples = []

        if self.has_local:
            local_files = sorted(glob.glob(os.path.join(self.local_dir, "**", "*.pt"), recursive=True))
        else:
            local_files = []

        if self.has_global:
            global_files = sorted(glob.glob(os.path.join(self.global_dir, "**", "*.pt"), recursive=True))
        else:
            global_files = []

        if self.has_local and self.has_global:
            # 建立 Global 相对路径索引
            global_map = {
                os.path.relpath(p, self.global_dir): p
                for p in global_files
            }

            for local_path in local_files:
                rel = os.path.relpath(local_path, self.local_dir)
                if rel not in global_map:
                    # 双通道时，如果没有对应 global 文件，则跳过
                    continue

                label = self._get_label(local_path)
                samples.append({
                    "local_path": local_path,
                    "global_path": global_map[rel],
                    "label": label,
                })

        elif self.has_local:
            for local_path in local_files:
                label = self._get_label(local_path)
                samples.append({
                    "local_path": local_path,
                    "global_path": None,
                    "label": label,
                })

        else:
            for global_path in global_files:
                label = self._get_label(global_path)
                samples.append({
                    "local_path": None,
                    "global_path": global_path,
                    "label": label,
                })

        if len(samples) == 0:
            raise RuntimeError("No valid .pt samples found.")

        return samples

    def _get_label(self, path: str) -> int:
        """
        优先从 labels_csv 读取标签。
        若 CSV 中不存在，则从目录名 0/1 推断。
        """
        fname = os.path.basename(path)
        if fname in self.labels_map:
            return int(self.labels_map[fname])
        return infer_label_from_path(path)

    def _compute_max_nodes(self) -> int:
        """
        扫描所有样本，确定最大 ROI 数 N。
        后续所有样本都会 padding 到该 N。
        """
        max_n = 0

        for sample in self.samples:
            if sample["local_path"] is not None:
                data = torch.load(sample["local_path"], map_location="cpu")
                adj = data["adjacency"]
                # Local adjacency: (T, N, N)
                n = int(adj.shape[1])
                max_n = max(max_n, n)

            if sample["global_path"] is not None:
                data = torch.load(sample["global_path"], map_location="cpu")
                adj = data["adjacency"]
                # Global adjacency: (N, N) or (1, N, N)
                if adj.dim() == 3:
                    n = int(adj.shape[1])
                else:
                    n = int(adj.shape[0])
                max_n = max(max_n, n)

        return max_n

    def __len__(self) -> int:
        return len(self.samples)

    def _read_local(self, path: str) -> Dict[str, Any]:
        """
        读取一个 Local 短期动态图 `.pt`。
        """
        data = torch.load(path, map_location="cpu")

        adj = data["adjacency"].float()
        if adj.dim() != 3:
            raise ValueError(f"Local adjacency must be (T, N, N), got {tuple(adj.shape)} in {path}")

        comm = data.get("community_sequence", None)
        if comm is not None:
            comm = comm.long()

        coords = data.get("coords", None)
        if coords is not None:
            coords = coords.float()

        adj = pad_dynamic_adj(adj, self.max_n)
        comm = pad_comm_2d(comm, self.max_n)
        coords = pad_coords(coords, self.max_n)

        return {
            "local_adj": adj,             # (T, N, N)
            "local_comm": comm,           # (T, N) or None
            "local_coords": coords,       # (N, 3) or None
            "local_len": adj.shape[0],    # T
        }

    def _read_global(self, path: str) -> Dict[str, Any]:
        """
        读取一个 Global 全局静态图 `.pt`。
        """
        data = torch.load(path, map_location="cpu")

        adj = data["adjacency"].float()
        if adj.dim() == 3:
            # 如果是 (1, N, N)，取第一张图
            adj = adj[0]
        if adj.dim() != 2:
            raise ValueError(f"Global adjacency must be (N, N), got {tuple(adj.shape)} in {path}")

        comm = data.get("community_labels", None)
        if comm is not None:
            comm = comm.long()

        coords = data.get("coords", None)
        if coords is not None:
            coords = coords.float()

        adj = pad_square_adj(adj, self.max_n)
        comm = pad_comm_1d(comm, self.max_n)
        coords = pad_coords(coords, self.max_n)

        return {
            "global_adj": adj,          # (N, N)
            "global_comm": comm,        # (N,) or None
            "global_coords": coords,    # (N, 3) or None
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        item = {
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "file": sample["local_path"] or sample["global_path"],
        }

        if sample["local_path"] is not None:
            item.update(self._read_local(sample["local_path"]))

        if sample["global_path"] is not None:
            item.update(self._read_global(sample["global_path"]))

        return item


def brain_graph_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader collate function。

    作用：
        1. 将 Local 动态图 padding 到当前 batch 的 T_max；
        2. 将样本堆叠成 batch tensor；
        3. 保留 label 和 file。
    """
    out = {}

    labels = torch.stack([b["label"] for b in batch])
    files = [b["file"] for b in batch]

    out["label"] = labels
    out["file"] = files

    has_local = "local_adj" in batch[0]
    has_global = "global_adj" in batch[0]

    if has_local:
        batch_size = len(batch)
        t_max = max(int(b["local_len"]) for b in batch)
        _, n, _ = batch[0]["local_adj"].shape

        local_adj = torch.zeros((batch_size, t_max, n, n), dtype=torch.float32)
        local_len = torch.zeros((batch_size,), dtype=torch.long)

        local_comm = None
        if batch[0]["local_comm"] is not None:
            local_comm = torch.full((batch_size, t_max, n), -1, dtype=torch.long)

        local_coords = None
        if batch[0]["local_coords"] is not None:
            local_coords = torch.zeros((batch_size, n, 3), dtype=torch.float32)

        for i, b in enumerate(batch):
            t = int(b["local_len"])
            local_adj[i, :t] = b["local_adj"]
            local_len[i] = t

            if local_comm is not None:
                local_comm[i, :t] = b["local_comm"]

            if local_coords is not None:
                local_coords[i] = b["local_coords"]

        out["local_adj"] = local_adj
        out["local_comm"] = local_comm
        out["local_coords"] = local_coords
        out["local_len"] = local_len

    if has_global:
        global_adj = torch.stack([b["global_adj"] for b in batch], dim=0)

        global_comm = None
        if batch[0]["global_comm"] is not None:
            global_comm = torch.stack([b["global_comm"] for b in batch], dim=0)

        global_coords = None
        if batch[0]["global_coords"] is not None:
            global_coords = torch.stack([b["global_coords"] for b in batch], dim=0)

        out["global_adj"] = global_adj
        out["global_comm"] = global_comm
        out["global_coords"] = global_coords

    return out


if __name__ == "__main__":
    """
    命令行示例：

    只读取 Global：
        python read_brain_graph_data_example.py --global_dir /path/to/global

    只读取 Local：
        python read_brain_graph_data_example.py --local_dir /path/to/local

    同时读取 Local + Global：
        python read_brain_graph_data_example.py --local_dir /path/to/local --global_dir /path/to/global
    """
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--global_dir", type=str, default=None)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    dataset = BrainGraphDataset(
        local_dir=args.local_dir,
        global_dir=args.global_dir,
        labels_csv=args.labels_csv,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=brain_graph_collate,
    )

    print(f"Number of samples: {len(dataset)}")
    print(f"Max number of ROI nodes: {dataset.max_n}")

    for batch in loader:
        print("Batch keys:", list(batch.keys()))

        if "local_adj" in batch:
            print("local_adj:", tuple(batch["local_adj"].shape))
            print("local_len:", tuple(batch["local_len"].shape))

        if "global_adj" in batch:
            print("global_adj:", tuple(batch["global_adj"].shape))

        print("label:", tuple(batch["label"].shape))
        print("first file:", batch["file"][0])
        break
