import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from src.datalaoder import SeedDataset
from src.model import DummyCNN, PLWrapper


class _TuneReportCallback(TuneReportCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def train(config, train_folder, test_folder, val_prop, model_name, mode, window_size):
    # Read datasets
    train_set = SeedDataset(data_dir=train_folder, mode='train', window_size=window_size)
    train_set, val_set = torch.utils.data.random_split(train_set, [1-val_prop, val_prop])
    test_set = SeedDataset(data_dir=test_folder, mode='test', window_size=window_size)
    
    # Create data loaders
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], num_workers=4)
    val_dataloader = DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], num_workers=4)
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], num_workers=4)
    
    loss = nn.CrossEntropyLoss()
    
    in_size = [3, window_size, window_size]
    out_size = [1, 12]
    if model_name == "dummy":
        convs = [32, 64, 64, 128, 128]
        mlp = [1024, 256]
        model_ = DummyCNN(config, convs, mlp, in_size, out_size)
    
    callbacks = []
    metrics = {"loss": "loss/val_loss", "acc": "acc/val_acc"}
    progress_bar = True
    if mode == "opt":
        callbacks = [_TuneReportCallback(metrics, on="validation_end")]
        progress_bar = False
    
    model = PLWrapper(config, model_, loss)
    trainer = pl.Trainer(
        max_epochs=500,
        enable_progress_bar=progress_bar,
        callbacks=callbacks)
    
    trainer.fit(model, train_dataloader, val_dataloader)


def main(train_folder, test_folder, val_prop, model_name, mode, window_size):
    config = {
        "kernel_size": tune.choice([3, 5, 7]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32]),
    }

    trainable = tune.with_parameters(train,
                                    train_folder=train_folder,
                                    test_folder=test_folder,
                                    val_prop=val_prop,
                                    model_name=model_name,
                                    mode=mode,
                                    window_size=window_size)

    analysis = tune.run(trainable,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        metric="loss",
                        mode="min",
                        config=config,
                        num_samples=10,
                        name="tune")

    print(analysis.best_config)


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
    parser.add_argument("--model", type=str, default="dummy",
            help="Model name")
    parser.add_argument("--mode", type=str, default="train",
            help="Execution mode: optmization (opt) | training (train) | testing (test)")
    parser.add_argument("--window", type=int, default=224,
            help="Window size")
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    args.train_folder = os.path.join(BASE_DIR, args.train_folder)
    args.test_folder = os.path.join(BASE_DIR, args.test_folder)
    
    if args.mode == 'opt':
        main(args.train_folder, args.test_folder, args.val_prop, args.model, args.mode, args.window)
    elif args.mode == 'train':
        # TODO improve to generalize for all CNN models
        config = {
            "kernel_size": 3,
            "lr": 1e-3,
            "batch_size": 8,
        }
        train(config, args.train_folder, args.test_folder, args.val_prop, args.model, args.mode, args.window)
    elif args.mode == 'test':
        # TODO add testing and save to csv with submission format
        pass
    else:
        raise Exception("Execution mode not defined")