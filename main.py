import torch
from torch.utils.data import DataLoader

from src.datalaoder import PlantsDataset


def main(train_folder, test_folder, val_prop, batch_size):
    # Read datasets
    train_set = PlantsDataset(data_dir=train_folder, mode='train')
    train_set, val_set = torch.utils.data.random_split(train_set, [1-val_prop, val_prop])
    test_set = PlantsDataset(data_dir=test_folder, mode='test')
    
    # Create data loaders
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_set, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    

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
    parser.add_argument("--batch-size", type=int, default=8,
            help="Batch size")
    args = parser.parse_args()
    main(args.train_folder, args.test_folder, args.val_prop, args.batch_size)