import torch
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassAccuracy


class PLWrapper(pl.LightningModule):
    def __init__(self, config, model, loss):
        super().__init__()
        self.config = config
        self.model = model
        self.loss = loss
        # TODO Generalize to n output classes
        self.train_acc = MulticlassAccuracy(num_classes=12)
        self.val_acc = MulticlassAccuracy(num_classes=12)

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        x = x.float()
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.train_acc(y_hat, y)
        self.log('loss/train_loss', loss, on_step=False, on_epoch=True)
        self.log('acc/train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        x = x.float()
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        self.val_acc(y_hat, y)
        self.log('loss/val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('acc/val_acc', self.val_acc, on_step=False, on_epoch=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        x, y = batch
        x = x.float()
        x = x.view(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
        y_hat = self.model(x)
        y_label = torch.argmax(y_hat, dim=1)
        return y, y_label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        return optimizer