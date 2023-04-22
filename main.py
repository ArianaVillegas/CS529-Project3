import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from ray import tune
import pandas as pd
import numpy as np
import Augmentor

from src.datalaoder import SeedDataset
from src.model import DummyCNN, PLWrapper
from src.utils import create_train_meta, create_test_meta


class _TuneReportCallback(TuneReportCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def augmentation(data_dir, tot_img):
    list_dir = os.listdir(data_dir)
    for idx, label in enumerate(list_dir):
        class_folder = os.path.join(data_dir, label)
        list_dir = os.listdir(class_folder)
        # Augmentation
        p = Augmentor.Pipeline(class_folder)
        p.random_distortion(probability=0.3, grid_width=16, grid_height=16, magnitude=8)
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.shear(probability=0.3, max_shear_left=10, max_shear_right=10)
        p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
        p.flip_random(probability=0.5)
        p.skew(probability=0.3)
        p.sample(tot_img - len(list_dir))
        
        
def get_model(config, model_name, window_size):
    in_size = [3, window_size, window_size]
    out_size = [1, 12]
    if model_name == "dummy":
        convs = [16, 32, 32, 64, 64]
        mlp = [1024, 64]
        model_ = DummyCNN(config, convs, mlp, in_size, out_size)
    elif model_name == "resnet":
        model_ = models.resnet50(pretrained=True)
        for param in model_.parameters():
            param.requires_grad = False
        num_ftrs = model_.fc.in_features
        model_.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, out_size[-1])
        )
    else:
        raise Exception(f"Model {model_name} not defined")
    return model_


def train(config, train_folder, val_prop, model_name, mode, window_size):
    # Read datasets
    meta = create_train_meta(train_folder)
    train_meta, val_meta = train_test_split(meta, test_size=val_prop, random_state=42, stratify=meta['idx'])
    train_set = SeedDataset(meta=train_meta, mode='train', window_size=window_size, augmented=True)
    val_set = SeedDataset(meta=val_meta, mode='val', window_size=window_size)
    
    # Create data loaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], num_workers=1)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], num_workers=1)
    
    loss = nn.CrossEntropyLoss()
    
    callbacks = [
        EarlyStopping(monitor="val/val_loss", mode="min", patience=50),
        ModelCheckpoint(dirpath=os.path.join(train_folder, "../../lightning_logs"), 
                        filename=f"{model_name}", save_top_k=1, monitor="val/val_loss")
    ]
    metrics = {"loss": "val/val_loss", "acc": "val/val_acc", "f1": "val/val_f1"}
    progress_bar = True
    if mode == "opt":
        callbacks += [_TuneReportCallback(metrics, on="validation_end")]
        progress_bar = False
    
    model_ = get_model(config, model_name, window_size)
    model = PLWrapper(config, model_, loss)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs=500,
        enable_progress_bar=progress_bar,
        callbacks=callbacks)
    
    trainer.fit(model, train_loader, val_loader)


def opt(train_folder, val_prop, model_name, mode, window_size):
    config = {
        "kernel_size": tune.choice([3, 5]),
        "lr": tune.loguniform(1e-4, 5e-3),
        "batch_size": tune.choice([8, 16, 32]),
    }

    trainable = tune.with_parameters(train,
                                    train_folder=train_folder,
                                    val_prop=val_prop,
                                    model_name=model_name,
                                    mode=mode,
                                    window_size=window_size)

    analysis = tune.run(trainable,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 0.5
                        },
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=10,
                        name="seeds")

    print(analysis.best_config)
    

def test(config, test_folder, model_name, window_size, classes):
    # Load test dataset
    meta = create_test_meta(test_folder)
    test_set = SeedDataset(meta=meta, mode='test', window_size=window_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], num_workers=1)
    
    # Build model
    model_ = get_model(config, model_name, window_size)
    loss = nn.CrossEntropyLoss()
    model = PLWrapper(config, model_, loss)
    
    # Load model checkpoint
    path = os.path.join(test_folder, f"../../lightning_logs/{model_name}.ckpt")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Predict test dataset
    model.eval()
    trainer = pl.Trainer()
    preds_ = trainer.predict(model, test_loader)
    names, preds = [], []
    for x, y in preds_:
        names.extend(list(x))
        preds.extend(np.array(y))
    pred_class = classes[preds]
    df = pd.DataFrame(data={'file': names, 'species': pred_class})
    df.to_csv(f'output_{model_name}.csv', index=False)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
            "Project 3 - Plant Seedlings Classification")
    parser.add_argument("--train-folder", type=str, default="data/train",
            help="The relative path to the training dataset folder")
    parser.add_argument("--test-folder", type=str, default="data/test",
            help="The relative path to the testting dataset folder")
    parser.add_argument("--val-prop", type=float, default=0.3,
            help="The validation proportion to split train and validation sets")
    parser.add_argument("--model", type=str, choices=['dummy', 'resnet'], default="dummy",
            help="Model name")
    parser.add_argument("--mode", type=str, choices=['opt', 'train', 'aug', 'test'], default="train",
            help="Execution mode: optmization (opt) | training (train) | augmentation (aug) | testing (test)")
    parser.add_argument("--window", type=int, default=224,
            help="Window size")
    parser.add_argument("--augmentation", type=int, default=1000,
            help="Data augmentation size per class")
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    args.train_folder = os.path.join(BASE_DIR, args.train_folder)
    args.test_folder = os.path.join(BASE_DIR, args.test_folder)
    classes = np.asarray(os.listdir(args.train_folder))
    
    if args.mode == 'opt':
        opt(args.train_folder, args.val_prop, args.model, args.mode, args.window)
    elif args.mode == 'aug':
        augmentation(args.train_folder, args.augmentation)
    elif args.mode == 'train':
        config = {
            "kernel_size": 3,
            "lr": 1e-3,
            "batch_size": 8,
        }
        train(config, args.train_folder, args.val_prop, args.model, args.mode, args.window)
    elif args.mode == 'test':
        config = {
            "kernel_size": 3,
            "lr": 5e-3,
            "batch_size": 32,
        }
        test(config, args.test_folder, args.model, args.window, classes)
    else:
        raise Exception("Execution mode not defined")