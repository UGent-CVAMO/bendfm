"""PointNeXt training and evaluation script for BenDFM.

Usage (run from repository root):
    python benchmark/train/pointnext.py --data_dir data/bendfm --label_key bbox_area_unfolded --regression
    python benchmark/train/pointnext.py --data_dir data/bendfm_u --label_key y_unfolding_collision
"""

import argparse
import sys
import pathlib
import torch
import json
import warnings
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
import random

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)
from tqdm import tqdm

# Add the openpoints model directory to Python path
_openpoints_path = pathlib.Path(__file__).parent.parent / "models" / "openpoints"
sys.path.insert(0, str(_openpoints_path))

warnings.simplefilter(action="ignore", category=FutureWarning)


def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PointCloudDataset(Dataset):
    """Dataset that loads point clouds (.npy) and labels from JSON files."""

    def __init__(self, npy_files, label_key, labels_dir=None, num_points=1024):
        self.npy_files = npy_files
        self.num_points = num_points
        self.label_key = label_key
        self.labels_dir = Path(labels_dir) if labels_dir else Path(npy_files[0]).parent.parent

    def __len__(self):
        return len(self.npy_files)

    def _get_label_from_json(self, filename):
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

    def __getitem__(self, idx):
        pc_path = self.npy_files[idx]
        pc = np.load(pc_path)
        if pc.shape[0] >= self.num_points:
            indices = np.random.choice(pc.shape[0], self.num_points, replace=False)
            pc = pc[indices]
        else:
            repeat = self.num_points - pc.shape[0]
            pc = np.concatenate([pc, pc[:repeat]], axis=0)

        xyz = torch.tensor(pc, dtype=torch.float)
        label = self._get_label_from_json(pc_path.stem)
        label = torch.tensor(float(label), dtype=torch.float32)

        return xyz, xyz, label  # (positions, features, label)


def evaluate(model, loader, device, criterion, regression, mean_label=0.5):
    """Evaluate model on a data loader.
    - If regression: returns (avg_loss, mse, mae, rmse)
    - If classification: returns (avg_loss, acc, auc, bal_acc, brier, f1)
    """
    model.eval()
    total_loss = 0.0
    preds_list, labels_list = [], []

    with torch.no_grad():
        for xyz, feats, labels in loader:
            xyz = xyz.to(device).float().contiguous()  # Shape: (batch, 3, 1024)
            feats = feats.to(device).float().contiguous()  # Shape: (batch, 3, 1024)
            labels = labels.to(device).float()
            logits = model(xyz, feats).squeeze()
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

    preds = np.concatenate(preds_list).ravel()
    gts = np.concatenate(labels_list).ravel()
    avg_loss = total_loss / max(1, len(loader))

    if regression:
        mse = mean_squared_error(gts, preds)
        mae = mean_absolute_error(gts, preds)
        rmse = float(np.sqrt(mse))
        return avg_loss, mse, mae, rmse
    else:
        probs = 1 / (1 + np.exp(-preds))
        yhat = (probs > mean_label).astype(int)
        acc = accuracy_score(gts, yhat)
        try:
            auc = roc_auc_score(gts, probs)
        except ValueError:
            auc = float('nan')
        try:
            bal_acc = balanced_accuracy_score(gts, yhat)
        except ValueError:
            bal_acc = float('nan')
        try:
            brier = brier_score_loss(gts, probs)
        except ValueError:
            brier = float('nan')
        try:
            f1 = f1_score(gts, yhat)
        except ValueError:
            f1 = float('nan')
        return avg_loss, acc, auc, bal_acc, brier, f1


def main(args, SEED=42):
    # ── Configuration ───────────────────────────────────────────────────
    LABEL_KEY = args.label_key
    REGRESSION = args.regression
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    PATIENCE = args.patience
    splits_root = Path(args.data_dir)

    # ── Dataset ─────────────────────────────────────────────────────────
    train_pcs = sorted((splits_root / "train" / "pcs").glob("*.npy"))
    val_pcs = sorted((splits_root / "val" / "pcs").glob("*.npy"))
    test_pcs = sorted((splits_root / "test" / "pcs").glob("*.npy"))

    train_set = PointCloudDataset(train_pcs, label_key=LABEL_KEY, labels_dir=splits_root / "train", num_points=1024)
    val_set = PointCloudDataset(val_pcs, label_key=LABEL_KEY, labels_dir=splits_root / "val", num_points=1024)
    test_set = PointCloudDataset(test_pcs, label_key=LABEL_KEY, labels_dir=splits_root / "test", num_points=1024)

    mean_label = float(np.mean([train_set._get_label_from_json(Path(p).stem) for p in train_pcs])) if len(train_pcs) else 0.5
    print(f"Train/Val/Test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}")
    print(f"Mean label (train): {mean_label:.4f}")

    def collate_fn(batch):
        xyz, feats, labels = zip(*batch)
        xyz = torch.stack(xyz).permute(0, 2, 1)
        feats = torch.stack(feats).permute(0, 2, 1)
        labels = torch.stack(labels)
        return xyz, feats, labels

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=22, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=22, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=22, collate_fn=collate_fn, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────
    from openpoints.models.backbone.pointnext import PointNextEncoder
    from openpoints.models.classification.cls_base import ClsHead

    class AttrDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            try:
                del self[key]
            except KeyError:
                raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    encoder_args = {
        'in_channels': 3,
        'width': 32,
        'blocks': [1, 1, 1, 1, 1, 1],
        'strides': [1, 2, 2, 2, 2, 1],
        'radius': 0.15,
        'radius_scaling': 1.5,
        'sa_layers': 2,
        'sa_use_res': True,
        'nsample': 32,
        'expansion': 4,
        'aggr_args': {'feature_type': 'dp_fj', 'reduction': 'max'},
        'group_args': AttrDict({'NAME': 'ballquery', 'normalize_dp': True}),
        'conv_args': {'order': 'conv-norm-act'},
        'act_args': {'act': 'relu'},
        'norm_args': {'norm': 'bn'},
        'block': 'InvResMLP',
        'sampler': 'fps',
        'use_res': True,
    }
    cls_args = {
        'num_classes': 1,
        'in_channels': 512,
        'mlps': [256],
        'dropout': 0.5,
        'global_feat': None,
    }

    class PointNextClassifier(nn.Module):
        def __init__(self, encoder_args, cls_args):
            super().__init__()
            self.encoder = PointNextEncoder(**encoder_args)
            self.cls_head = ClsHead(**cls_args)

        def forward(self, p0, f0):
            global_feat = self.encoder.forward_cls_feat(p0, f0)
            return self.cls_head(global_feat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNextClassifier(encoder_args, cls_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() if REGRESSION else nn.BCEWithLogitsLoss()

    print(f"Using device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Training ────────────────────────────────────────────────────────
    dataset_name = splits_root.name
    save_dir = Path(f"results/pointnext/{dataset_name}/{time.strftime('%m%d-%H%M')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (xyz, feats, label) in enumerate(loop):
            xyz = xyz.to(device).float().contiguous()  # Shape: (batch, 3, 1024)
            feats = feats.to(device).float().contiguous()  # Shape: (batch, 3, 1024)
            label = label.to(device).float()
            optimizer.zero_grad()
            preds = model(xyz, feats).squeeze()
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # ── Epoch metrics ───────────────────────────────────────────
        train_preds = np.concatenate(all_preds).ravel()
        train_labels = np.concatenate(all_labels).ravel()
        avg_train_loss = epoch_loss / len(train_loader)

        if REGRESSION:
            train_mae = mean_absolute_error(train_labels, train_preds)
            val_loss, val_mse, val_mae, val_rmse = evaluate(model, val_loader, device, criterion, True)

            tqdm.write(
                f"  Train -> Loss: {avg_train_loss:.4f}  MAE: {train_mae:.4f}\n"
                f"  Val   -> Loss: {val_loss:.4f}  MAE: {val_mae:.4f}  RMSE: {val_rmse:.4f}"
            )
        else:
            train_probs = 1 / (1 + np.exp(-train_preds))
            train_yhat = (train_probs > mean_label).astype(int)
            train_acc = accuracy_score(train_labels, train_yhat)
            try:
                train_auc = roc_auc_score(train_labels, train_probs)
            except ValueError:
                train_auc = float('nan')
            val_loss, val_acc, val_auc, val_bal_acc, val_brier, val_f1 = evaluate(
                model, val_loader, device, criterion, False, mean_label
            )

            tqdm.write(
                f"  Train -> Loss: {avg_train_loss:.4f}  Acc: {train_acc:.4f}  AUC: {train_auc:.4f}\n"
                f"  Val   -> Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}  "
                f"BalAcc: {val_bal_acc:.4f}  F1: {val_f1:.4f}  Brier: {val_brier:.4f}"
            )

        # ── Early stopping ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            tqdm.write("  Saved best model.")
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            tqdm.write(f"  Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

    # ── Test evaluation ─────────────────────────────────────────────────
    best_ckpt = save_dir / "best_model.pth"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    if REGRESSION:
        test_loss, test_mse, test_mae, test_rmse = evaluate(model, test_loader, device, criterion, True)
        test_preds, test_labels = [], []
        with torch.no_grad():
            for xyz, feats, labels in test_loader:
                xyz = xyz.to(device).float().transpose(1, 2)
                feats = feats.to(device).float().transpose(1, 2)
                preds = model(xyz, feats).squeeze()
                test_preds.append(preds.cpu().numpy())
                test_labels.append(labels.numpy())
        test_preds = np.concatenate(test_preds).ravel()
        test_labels = np.concatenate(test_labels).ravel()
        test_mape = np.mean(np.abs((test_labels - test_preds) / np.where(test_labels != 0, test_labels, 1))) * 100
        test_r2 = 1 - (np.sum((test_labels - test_preds) ** 2) / np.sum((test_labels - np.mean(test_labels)) ** 2))

        null_preds = np.full_like(test_labels, fill_value=mean_label, dtype=np.float32)
        null_mse = mean_squared_error(test_labels, null_preds)
        null_mae = mean_absolute_error(test_labels, null_preds)
        null_rmse = float(np.sqrt(null_mse))
        null_mape = np.mean(np.abs((test_labels - null_preds) / np.where(test_labels != 0, test_labels, 1))) * 100
        null_r2 = 1 - (np.sum((test_labels - null_preds) ** 2) / np.sum((test_labels - np.mean(test_labels)) ** 2))

        print(f"\nTest  -> MSE: {test_mse:.4f}  MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  MAPE: {test_mape:.2f}%  R2: {test_r2:.4f}")
        print(f"Null  -> MSE: {null_mse:.4f}  MAE: {null_mae:.4f}  RMSE: {null_rmse:.4f}  MAPE: {null_mape:.2f}%  R2: {null_r2:.4f}")
    else:
        test_loss, acc, auc, bal_acc, brier, f1 = evaluate(model, test_loader, device, criterion, False, mean_label)

        print(
            f"\nTest -> Loss: {test_loss:.4f}  Acc: {acc:.4f}  AUC: {auc:.4f}  "
            f"BalAcc: {bal_acc:.4f}  F1: {f1:.4f}  Brier: {brier:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PointNeXt on BenDFM.")
    parser.add_argument("--data_dir", type=str, default="data/bendfm",
                        help="Path to dataset root (e.g. data/bendfm or data/bendfm_u).")
    parser.add_argument("--label_key", type=str, default="bbox_area_unfolded",
                        help="Label key from *_labels.json to predict.")
    parser.add_argument("--regression", action="store_true", default=True,
                        help="Regression task (default). Use --no-regression for classification.")
    parser.add_argument("--no-regression", dest="regression", action="store_false")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nRunning with seed: {args.seed}")
    set_seeds(args.seed)
    main(args, SEED=args.seed)
