"""UV-Net training and evaluation script for BenDFM.

Usage (run from repository root):
    python benchmark/train/uvnet.py --data_dir data/bendfm --label_key bbox_area_unfolded --regression
    python benchmark/train/uvnet.py --data_dir data/bendfm --label_key y_tool_collision
    python benchmark/train/uvnet.py --data_dir data/bendfm_u --label_key y_unfolding_collision
"""

import argparse
import pathlib
import sys
import numpy as np
import torch
import random
import time

# Add the benchmark directory to Python path
src_path = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)
from models.uvnet.UVNet import UVNet
from input_transformation.graph_dataset import prepare_dataset

torch.set_float32_matmul_precision('medium')


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


def main(args, SEED):
    # ── Configuration ───────────────────────────────────────────────────
    DATA_DIR = args.data_dir
    LABEL_KEY = args.label_key
    REGRESSION = args.regression
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    PATIENCE = args.patience

    # ── Dataset ─────────────────────────────────────────────────────────
    train_dataset, val_dataset, test_dataset, avg_label = prepare_dataset(
        DATA_DIR, label_key=LABEL_KEY, center_and_scale=False,
    )

    train_loader = train_dataset.get_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=22)
    val_loader = val_dataset.get_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=22)
    test_loader = test_dataset.get_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=22)

    # ── Model ───────────────────────────────────────────────────────────
    model = UVNet(num_classes=1, learning_rate=LEARNING_RATE, threshold=avg_label, regression=REGRESSION)

    # ── Logging & callbacks ─────────────────────────────────────────────
    dataset_name = pathlib.Path(DATA_DIR).name
    results_path = pathlib.Path("results") / "uvnet" / dataset_name / LABEL_KEY
    run_path = results_path / time.strftime("%m%d") / time.strftime("%H%M")
    run_path.mkdir(parents=True, exist_ok=True)

    loggers = [TensorBoardLogger(str(results_path), name=time.strftime("%m%d"), version=time.strftime("%H%M"))]
    callbacks = [
        ModelCheckpoint(monitor="val_loss", dirpath=str(run_path), filename="best", save_last=True, save_top_k=1, mode="min"),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
        TQDMProgressBar(refresh_rate=10),
    ]

    # ── Training ────────────────────────────────────────────────────────
    trainer = Trainer(max_epochs=1000, callbacks=callbacks, logger=loggers, accelerator="gpu", devices=1, log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)

    # ── Test evaluation ─────────────────────────────────────────────────
    checkpoint_path = str(run_path / "best.ckpt")
    model = UVNet.load_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_labels, all_losses, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            graphs = batch['graph'].to(device)
            labels = batch['label'].to(device).float()
            graphs.ndata["x"] = graphs.ndata["x"].permute(0, 3, 1, 2)
            graphs.edata["x"] = graphs.edata["x"].permute(0, 2, 1)

            logits = model(graphs).squeeze()

            if REGRESSION:
                loss = torch.nn.functional.mse_loss(logits, labels)
                all_preds.extend(logits.cpu().numpy())
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
                probs = torch.sigmoid(logits).squeeze()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs > avg_label).long().cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_losses.append(loss.item())

    all_preds = np.asarray(all_preds, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.float32)

    if REGRESSION:
        test_mse = mean_squared_error(all_labels, all_preds)
        test_mae = mean_absolute_error(all_labels, all_preds)
        test_rmse = float(np.sqrt(test_mse))
        test_mape = np.mean(np.abs((all_labels - all_preds) / np.where(all_labels != 0, all_labels, 1))) * 100
        test_r2 = 1 - (np.sum((all_labels - all_preds) ** 2) / np.sum((all_labels - np.mean(all_labels)) ** 2))

        null_preds = np.full_like(all_labels, fill_value=float(avg_label), dtype=np.float32)
        null_mse = mean_squared_error(all_labels, null_preds)
        null_mae = mean_absolute_error(all_labels, null_preds)
        null_rmse = float(np.sqrt(null_mse))
        null_mape = np.mean(np.abs((all_labels - null_preds) / np.where(all_labels != 0, all_labels, 1))) * 100
        null_r2 = 1 - (np.sum((all_labels - null_preds) ** 2) / np.sum((all_labels - np.mean(all_labels)) ** 2))

        print(f"\nTest  -> MSE: {test_mse:.4f}  MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  MAPE: {test_mape:.2f}%  R2: {test_r2:.4f}")
        print(f"Null  -> MSE: {null_mse:.4f}  MAE: {null_mae:.4f}  RMSE: {null_rmse:.4f}  MAPE: {null_mape:.2f}%  R2: {null_r2:.4f}")
    else:
        all_probs = np.asarray(all_probs, dtype=np.float32)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float('nan')
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        brier = brier_score_loss(all_labels, all_probs)

        print(
            f"\nTest -> Loss: {np.mean(all_losses):.4f}  Acc: {acc:.4f}  AUC: {auc:.4f}  "
            f"BalAcc: {bal_acc:.4f}  F1: {f1:.4f}  Brier: {brier:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UV-Net on BenDFM.")
    parser.add_argument("--data_dir", type=str, default="data/bendfm",
                        help="Path to dataset root (e.g. data/bendfm or data/bendfm_u).")
    parser.add_argument("--label_key", type=str, default="bbox_area_unfolded",
                        help="Label key from *_labels.json to predict.")
    parser.add_argument("--regression", action="store_true", default=False,
                        help="Regression task. Default is classification.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nRunning with seed: {args.seed}")
    set_seeds(args.seed)
    main(args, SEED=args.seed)
