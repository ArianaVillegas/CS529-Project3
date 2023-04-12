from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from PIL import Image
import numpy as np
import os

from torch.utils.data import Dataset


class SeedDataset(Dataset):
    def __init__(self, data_dir, mode='train', window_size=256):
        self.data_dir = data_dir
        self.mode = mode
        self.window_size = window_size
        self.data = []
        self.imgs = []
        self._load_dir()
        
    def _load_dir(self):
        list_dir = os.listdir(self.data_dir)
        for idx, label in enumerate(list_dir):
            if self.mode == 'train':
                class_folder = os.path.join(self.data_dir, label)
                list_dir = os.listdir(class_folder)
                # Interate over images
                for dir_path in [class_folder, os.path.join(class_folder, 'output')]:
                    for name in os.listdir(dir_path):
                        if name == 'output':
                            continue
                        self.data.append((os.path.join(dir_path, name), idx))
                        self.imgs.append(name)
            elif self.mode == 'test':
                self.data.append((os.path.join(self.data_dir, label), label))
                self.imgs.append(label)
        self.data = np.asarray(self.data)

    def __getitem__(self, item):
        image = Image.open(self.data[item][0])
        image = image.convert('RGB')
        image = image.resize((self.window_size, self.window_size))
        data = np.asarray(image)
        if self.mode == 'train':
            return [data, int(self.data[item][1])]
        return [data, self.data[item][1]]

    def __len__(self):
        return len(self.data)
    
    def get_img_names(self):
        return self.imgs