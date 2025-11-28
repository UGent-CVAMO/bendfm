# UV-Net model for solid classification/regression.
# Based on UV-Net by Jayaraman et al. (Autodesk Research)
# Original: https://github.com/AutodeskAILab/UV-Net
# Licensed under the MIT License. See LICENSE in this directory.
#
# Modified towards regression for BenDFM manufacturability assessment.

import lightning.pytorch as pl
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from . import encoders
from torchmetrics import AUROC
import random
import numpy as np
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryCalibrationError
from torchmetrics.regression import MeanAbsoluteError
import dgl

# Set all relevant seeds to 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

auroc = AUROC(num_classes=2,task='binary')


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor 
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x

###############################################################################
# model
###############################################################################

class UVNetModel(nn.Module):
    """
    UV-Net solid classification model
    """

    def __init__(
        self,
        num_classes,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.3,
    ):
        """
        Initialize the UV-Net solid classification model
        
        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = encoders.UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)
    

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Map to logits
        out = self.clf(graph_emb)
        return out

class UVNet(pl.LightningModule):
    def __init__(self, num_classes, regression=False, threshold=0.5, learning_rate=1e-3, agg_features=False):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetModel(num_classes=num_classes)
        self.regression = regression
        self.threshold = threshold
        self.learning_rate = learning_rate

        if regression:
            self.loss_fn = nn.MSELoss()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            self.test_mae = MeanAbsoluteError()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.train_auc = BinaryAUROC()
            self.train_acc = BinaryAccuracy(threshold=threshold)
            self.val_auc = BinaryAUROC()
            self.val_acc = BinaryAccuracy(threshold=threshold)
            self.test_auc = BinaryAUROC()
            self.test_acc = BinaryAccuracy(threshold=threshold)

    def _preprocess(self, batch):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)

        # Permute the node and edge features
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        return inputs, labels.float() if self.regression else labels.long()

    def forward(self, batched_graph):
        return self.model(batched_graph)

    def training_step(self, batch, batch_idx):
        inputs, labels = self._preprocess(batch)
        logits = self.model(inputs).squeeze(-1)
        if self.regression:
            logits = logits.view_as(labels)
        loss = self.loss_fn(logits, labels.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.regression:
            self.train_mae.update(logits, labels)
        else:
            probs = torch.sigmoid(logits)
            self.train_auc.update(probs, labels)
            self.train_acc.update(probs, labels)
        return loss

    def on_train_epoch_end(self):
        if self.regression:
            self.log("train_mae", self.train_mae.compute(), on_epoch=True, sync_dist=True)
            self.train_mae.reset()
        else:
            self.log("train_auc", self.train_auc.compute(), on_epoch=True, sync_dist=True)
            self.log("train_acc", self.train_acc.compute(), on_epoch=True, sync_dist=True)
            self.train_auc.reset()
            self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = self._preprocess(batch)
        logits = self.model(inputs).squeeze(-1)
        if self.regression:
            logits = logits.view_as(labels)
        loss = self.loss_fn(logits, labels.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.regression:
            self.val_mae.update(logits, labels)
        else:
            probs = torch.sigmoid(logits)
            self.val_auc.update(probs, labels)
            self.val_acc.update(probs, labels)
        return loss

    def on_validation_epoch_end(self):
        if self.regression:
            self.log("val_mae", self.val_mae.compute(), on_epoch=True, sync_dist=True)
            self.val_mae.reset()
        else:
            self.log("val_auc", self.val_auc.compute(), on_epoch=True, sync_dist=True)
            self.log("val_acc", self.val_acc.compute(), on_epoch=True, sync_dist=True)
            self.val_auc.reset()
            self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        inputs, labels = self._preprocess(batch)
        logits = self.model(inputs).squeeze(-1)
        if self.regression:
            logits = logits.view_as(labels)
        loss = self.loss_fn(logits, labels.float())
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.regression:
            self.test_mae.update(logits, labels)
        else:
            probs = torch.sigmoid(logits)
            self.test_auc.update(probs, labels)
            self.test_acc.update(probs, labels)
        return loss

    def on_test_epoch_end(self):
        if self.regression:
            self.log("test_mae", self.test_mae.compute(), on_epoch=True, sync_dist=True)
            self.test_mae.reset()
        else:
            self.log("test_auc", self.test_auc.compute(), on_epoch=True, sync_dist=True)
            self.log("test_acc", self.test_acc.compute(), on_epoch=True, sync_dist=True)
            self.test_auc.reset()
            self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)