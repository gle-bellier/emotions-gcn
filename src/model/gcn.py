import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.net import Net


class GraphLevelGNN(pl.LightningModule):
    def __init__(self, lr, h_sizes, batch_size=32):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.model = Net(in_size=10, h_sizes=h_sizes, num_classes=6)
        self.loss = nn.NLLLoss()

    def forward(self, data, mode="train"):

        x, edge_index, edge_attr, batch_idx = data.x, data.edge_index, data.edge_attr, data.batch

        preds = self.model(x, edge_index, edge_attr, data.batch, batch_idx)
        loss = self.loss(preds, data.y.long())

        preds = preds.argmax(-1)

        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        """Configure both generator and discriminator optimizers
        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        sc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                        verbose=True,
                                                        patience=10,
                                                        factor=0.5)

        return {
            'optimizer': opt,
            'lr_scheduler': sc,
            "monitor": "train/train_loss"
        }

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train/train_loss', loss)
        self.log('train/train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log('val/val_loss', loss)
        self.log('val/val_acc', acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc)
