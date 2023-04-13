import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from ray import tune
import pandas as pd
import numpy as np
import Augmentor

from src.datalaoder import SeedDataset
from src.model import DummyCNN, PLWrapper


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
        
        
def get_model(model_name, window_size):
    in_size = [3, window_size, window_size]
    out_size = [1, 12]
    if model_name == "dummy":
        convs = [16, 32, 32, 64, 64]
        mlp = [1024, 64]
        model_ = DummyCNN(config, convs, mlp, in_size, out_size)
    elif model_name == "vgg":
        model_ = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        model_.classifier[-1] = nn.Linear(in_features=4096, out_features=out_size[-1], bias=True)
    else:
        raise Exception(f"Model {model_name} not defined")
    return model_


def train(config, train_folder, val_prop, model_name, mode, window_size):
    # Read datasets
    train_set = SeedDataset(data_dir=train_folder, mode='train', window_size=window_size)
    train_set, val_set = torch.utils.data.random_split(train_set, [1-val_prop, val_prop])
    
    # Create data loaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], num_workers=1)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], num_workers=1)
    
    loss = nn.CrossEntropyLoss()
    
    callbacks = [
        EarlyStopping(monitor="loss/val_loss", mode="min", patience=20),
        ModelCheckpoint(dirpath=os.path.join(train_folder, "../../lightning_logs"), 
                        filename=f"{model_name}", save_top_k=1, monitor="loss/val_loss")
    ]
    metrics = {"loss": "loss/val_loss", "acc": "acc/val_acc"}
    progress_bar = True
    if mode == "opt":
        callbacks += [_TuneReportCallback(metrics, on="validation_end")]
        progress_bar = False
    
    model_ = get_model(model_name, window_size)
    model = PLWrapper(config, model_, loss)
    trainer = pl.Trainer(
        max_epochs=250,
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
    test_set = SeedDataset(data_dir=test_folder, mode='test', window_size=window_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], num_workers=1)
    
    # Load model checkpoint
    path = os.path.join(test_folder, f"../../lightning_logs/{model_name}.ckpt")
    checkpoint = torch.load(path)
    model_ = get_model(model_name, window_size)
    
    loss = nn.CrossEntropyLoss()
    model = PLWrapper(config, model_, loss)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Predict test dataset
    model.eval()
    trainer = pl.Trainer()
    preds_ = trainer.predict(model, test_loader)
    names = []
    [names.extend(list(x)) for x, y in preds_]
    preds = []
    [preds.extend(np.array(y)) for x, y in preds_]
    pred_class = classes[preds]
    df = pd.DataFrame(data={'file': names, 'species': pred_class})
    df.to_csv('output.csv', index=False)


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
    parser.add_argument("--model", type=str, choices=['dummy', 'vgg'], default="dummy",
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
        # TODO improve to generalize for all CNN models
        config = {
            "kernel_size": 3,
            "lr": 1e-3,
            "batch_size": 16,
        }
        train(config, args.train_folder, args.val_prop, args.model, args.mode, args.window)
    elif args.mode == 'test':
        # TODO add testing and save to csv with submission format
        config = {
            "kernel_size": 3,
            "lr": 1e-3,
            "batch_size": 16,
        }
        test(config, args.test_folder, args.model, args.window, classes)
    else:
        raise Exception("Execution mode not defined")