import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision


class PLWrapper(pl.LightningModule):
    def __init__(self, config, model, loss, model_name):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = model
        self.loss = loss

        # Accuraction for train and validation 
        self.train_acc = MulticlassAccuracy(num_classes=12)
        self.val_acc = MulticlassAccuracy(num_classes=12)
        # F1 score for train and validation
        self.train_f1 = MulticlassF1Score(num_classes=12)
        self.val_f1 = MulticlassF1Score(num_classes=12)
        # Recall for train y validation
        self.train_recall = MulticlassRecall(num_classes=12)
        self.val_recall = MulticlassRecall(num_classes=12)
        # Precision for train and validation
        self.train_precision = MulticlassPrecision(num_classes=12)
        self.val_precision = MulticlassPrecision(num_classes=12)


    def training_step(self, batch, batch_idx):
        self.model.train()
        x, y = batch
        x = x.float()

        if self.model_name == "inceptionV3":
            y_hat, y_hat_aux = self.model(x)
            loss1 = self.loss(y_hat, y)
            loss2 = self.loss(y_hat_aux, y)
            loss = loss1 + 0.4 * loss2
        else:
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
        
        y_hat = torch.softmax(y_hat, dim=1)
        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.log('train/train_loss', loss, on_step=False, on_epoch=True)
        self.log('train/train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train/train_f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train/train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train/train_recall', self.train_recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        x = x.float()
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)

        self.log('val/val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val/val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val/val_f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val/val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/val_recall', self.val_recall, on_step=False, on_epoch=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        x, y = batch
        x = x.float()
        y_hat = self.model(x)
        y_hat = torch.softmax(y_hat, dim=1)
        y_label = torch.argmax(y_hat, dim=1)
        return y, y_label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]