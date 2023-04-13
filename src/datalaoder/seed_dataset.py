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
        self._load_dir()
        
    def _load_dir(self):
        npz_file = os.path.join(self.data_dir, f'data_{self.window_size}.npz')
        if not Path(npz_file).is_file(): 
            labels = []
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
                            image = Image.open(os.path.join(dir_path, name))
                            image = image.convert('RGB')
                            image = image.resize((self.window_size, self.window_size))
                            data = np.asarray(image)
                            self.data.append(data)
                            labels.append(idx)
                elif self.mode == 'test':
                    image = Image.open(os.path.join(self.data_dir, label))
                    image = image.convert('RGB')
                    image = image.resize((self.window_size, self.window_size))
                    data = np.asarray(image, dtype=np.float32)
                    self.data.append(data)
                    labels.append(os.path.join(self.data_dir, label))
            self.data = np.asarray(self.data)
            np.savez(npz_file, data=self.data, labels=labels)

        self.data = np.load(npz_file)['data']
        labels = np.load(npz_file)['labels']
        self.data = list(zip(self.data, labels))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)