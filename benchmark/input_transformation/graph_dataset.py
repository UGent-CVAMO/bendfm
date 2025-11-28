import json
import random
import pathlib

import numpy as np
import torch
import dgl
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
from dgl.data.utils import load_graphs


def rotate_uvgrid(inp, rotation):
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].view(-1, 3), Rmat).view(orig_size)
    inp[..., 3:6] = torch.mm(inp[..., 3:6].view(-1, 3), Rmat).view(orig_size)
    return inp


def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp


def get_random_rotation():
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    angles = [0.0, 45.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def triangle_counts_dgl(g):
    nx_g = g.to_networkx().to_undirected()
    tri_dict = nx.triangles(nx_g)
    n = g.num_nodes()
    tri_counts = torch.zeros(n, dtype=torch.float32)
    for node, count in tri_dict.items():
        tri_counts[node] = count
    return tri_counts.unsqueeze(1)


def add_topological_features(g, lap_k=8, ppr_k=8, alpha=0.85):
    n = g.num_nodes()
    tri_counts = triangle_counts_dgl(g)
    lap_eig = dgl.lap_pe(g, k=lap_k).float()
    nx_g = g.to_networkx().to_undirected()
    pr_matrix = []
    for node in range(n):
        pr = nx.pagerank(nx_g, alpha=alpha, personalization={node: 1})
        pr_vec = np.array([pr[i] for i in range(n)])
        pr_matrix.append(pr_vec)
    pr_matrix_np = np.array(pr_matrix)
    pr_tensor = torch.from_numpy(pr_matrix_np).float()
    ppr_topk, _ = torch.topk(pr_tensor, k=ppr_k, dim=1)
    combined_feats = torch.cat([tri_counts, lap_eig, ppr_topk], dim=1)
    g.ndata['topo_feats'] = combined_feats
    return g


class GraphDataset(Dataset):
    def __init__(self, root_dir, files, labels, label_key, split="train",
                 center_and_scale=True, random_rotate=False, labels_dir=None):
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.files = files
        self.labels = labels
        self.label_key = label_key
        self.random_rotate = random_rotate
        self.labels_dir = pathlib.Path(labels_dir) if labels_dir else self.root_dir.parent
        self.center_and_scale = center_and_scale

    def load_one_graph(self, file_path):
        return load_graphs(str(file_path))[0][0]

    def _get_label_from_json(self, filename):
        if self.label_key is None:
            raise ValueError("label_key must be provided to extract labels from JSON.")
        json_path = self.labels_dir / f"{filename}_labels.json"
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data[self.label_key]
        except FileNotFoundError:
            print(f"Warning: Label file not found at {json_path}")
        except KeyError:
            print(f"Warning: Label key '{self.label_key}' not found in {json_path}")
        except Exception as e:
            print(f"Error reading label from {json_path}: {e}")
        return -1

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        collated = {"graph": batched_graph, "filename": batched_filenames}
        collated["label"] = torch.cat([x["label"] for x in batch], dim=0)
        return collated


    # No longer needed; handled per-sample

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        graph = self.load_one_graph(file_path)
        # Center and scale if requested
        if self.center_and_scale:
            graph.ndata["x"], center, scale = center_and_scale_uvgrid(graph.ndata["x"], return_center_scale=True)
            graph.edata["x"][..., :3] -= center
            graph.edata["x"][..., :3] *= scale
        # Convert to float32
        graph.ndata["x"] = graph.ndata["x"].float()
        graph.edata["x"] = graph.edata["x"].float()
        # Random rotation if requested
        if self.random_rotate:
            rotation = get_random_rotation()
            graph.ndata["x"] = rotate_uvgrid(graph.ndata["x"], rotation)
            graph.edata["x"] = rotate_uvgrid(graph.edata["x"], rotation)
        sample = {"graph": graph, "filename": file_path.stem, "label": label}
        return sample

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=False,
        )
def prepare_dataset(data_root, label_key, center_and_scale=True, **kwargs):
    """Load pre-split train/val/test graph datasets.

    Expects the following directory layout under *data_root*::

        data_root/
        ├── train/
        │   ├── graphs/*.bin          # DGL graph files
        │   └── *_labels.json         # label files
        ├── val/   (same structure)
        └── test/  (same structure)

    Args:
        data_root: Path to dataset root (e.g. ``data/bendfm``).
        label_key: Key to extract from ``*_labels.json`` files.
        center_and_scale: Whether to center and scale UV-grids.

    Returns:
        (train_dataset, val_dataset, test_dataset, avg_label)
    """
    data_root = pathlib.Path(data_root).resolve()

    for split in ("train", "val", "test"):
        graphs_dir = data_root / split / "graphs"
        if not graphs_dir.exists():
            raise FileNotFoundError(
                f"Expected graphs directory not found: {graphs_dir}\n"
                f"Run benchmark/input_transformation/features.py first to generate graph files."
            )

    def _collect_split(split_name):
        split_dir = data_root / split_name
        graphs_dir = split_dir / "graphs"
        files = sorted(graphs_dir.glob("*.bin"))
        valid_files, labels_list = [], []
        for f in files:
            json_file = split_dir / f"{f.stem}_labels.json"
            try:
                with open(json_file, "r") as jf:
                    data = json.load(jf)
                label_val = data.get(label_key, None)
                if label_val is None:
                    print(f"Warning: '{label_key}' not found in {json_file}")
                    continue
                valid_files.append(f)
                labels_list.append(label_val)
            except FileNotFoundError:
                print(f"Warning: Could not find label file {json_file}")
            except Exception as e:
                print(f"Error reading label from {json_file}: {e}")
        return split_dir, graphs_dir, valid_files, np.array(labels_list)

    def _make_dataset(split_name, graphs_dir, files, labels_np, labels_dir):
        label_tensors = [torch.FloatTensor([float(l)]) for l in labels_np]
        return GraphDataset(
            graphs_dir, files, label_tensors, label_key,
            split=split_name, labels_dir=labels_dir,
            center_and_scale=center_and_scale,
        )

    print(f"[Data] Loading pre-split datasets from: {data_root}")
    train_split_dir, train_graphs_dir, train_files, train_labels = _collect_split("train")
    val_split_dir, val_graphs_dir, val_files, val_labels = _collect_split("val")
    test_split_dir, test_graphs_dir, test_files, test_labels = _collect_split("test")

    if len(train_files) == 0 or len(val_files) == 0 or len(test_files) == 0:
        raise ValueError("One or more splits are empty. Check your split directories and label key.")

    avg_label = float(np.mean(train_labels))
    print(f"[Data] Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}  "
          f"Mean label (train): {avg_label:.4f}")

    train_dataset = _make_dataset("train", train_graphs_dir, train_files, train_labels, train_split_dir)
    val_dataset = _make_dataset("val", val_graphs_dir, val_files, val_labels, val_split_dir)
    test_dataset = _make_dataset("test", test_graphs_dir, test_files, test_labels, test_split_dir)

    return train_dataset, val_dataset, test_dataset, avg_label