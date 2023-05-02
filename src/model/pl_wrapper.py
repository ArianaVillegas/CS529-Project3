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

        # Metrics to track during training and validation
        self.train_acc = MulticlassAccuracy(num_classes=12)
        self.val_acc = MulticlassAccuracy(num_classes=12)
        self.train_f1 = MulticlassF1Score(num_classes=12)
        self.val_f1 = MulticlassF1Score(num_classes=12)
        self.train_recall = MulticlassRecall(num_classes=12)
        self.val_recall = MulticlassRecall(num_classes=12)
        self.train_precision = MulticlassPrecision(num_classes=12)
        self.val_precision = MulticlassPrecision(num_classes=12)


    def training_step(self, batch, batch_idx):
        # Training of the model on the training set
        self.model.train()
        x, y = batch
        x = x.float()

        # For some models, like InceptionV3, there are auxiliary outputs
        # that need to be considered in the loss function
        if self.model_name == "inceptionV3":
            y_hat, y_hat_aux = self.model(x)
            loss1 = self.loss(y_hat, y)
            loss2 = self.loss(y_hat_aux, y)
            loss = loss1 + 0.4 * loss2
        else:
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
        
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        
        # Calculate and log accuracy, F1 score, precision, and recall
        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        
        # Logging the metrics
        self.log('train/train_loss', loss, on_step=False, on_epoch=True)
        self.log('train/train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train/train_f1', self.train_f1, on_step=False, on_epoch=True)
        self.log('train/train_precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train/train_recall', self.train_recall, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluation of the model on the validation set
        self.model.eval()
        x, y = batch
        x = x.float()
        y_hat = self.model(x)
        val_loss = self.loss(y_hat, y)
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat = torch.argmax(y_hat, dim=1)
        
        # Metrics computation
        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        
        # Logging the metrics
        self.log('val/val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val/val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val/val_f1', self.val_f1, on_step=False, on_epoch=True)
        self.log('val/val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val/val_recall', self.val_recall, on_step=False, on_epoch=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Prediction of the model on new data
        self.model.eval()
        x, y = batch
        x = x.float()
        y_hat = self.model(x)
        
        # Post-processing of the output
        y_hat = torch.softmax(y_hat, dim=1)
        y_label = torch.argmax(y_hat, dim=1)
        return y, y_label

    def configure_optimizers(self):
        # Definition of the optimizer and the learning rate scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]