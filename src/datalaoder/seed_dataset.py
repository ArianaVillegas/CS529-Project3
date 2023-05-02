from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SeedDataset(Dataset):
    def __init__(self, meta, mode='train', window_size=224, augmented=False):
        """
        Initializes SeedDataset class with input metadata, mode, window size and augmentation flag.

        Args:
            meta (pandas.DataFrame): metadata for the dataset
            mode (str): dataset mode, train or val
            window_size (int): the target size of the image 
            augmented (bool): flag indicating if the dataset should be augmented

        Returns:
            None
        """
        self.meta = meta
        self.mode = mode
        self.window_size = window_size
        self.augmented = augmented

    def __getitem__(self, item):
        """
        Returns the item at a given index.

        Args:
            item (int): index of the item to retrieve

        Returns:
            tuple: A tuple containing the image tensor and the label.
        """
        image = Image.open(self.meta.iloc[item][0]).convert('RGB')
        if self.augmented and self.mode == 'train':
            self.transform = transforms.Compose([
                            transforms.RandomRotation(180),
                            transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
                            transforms.Resize((self.window_size, self.window_size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        else:
            self.transform = transforms.Compose([
                            transforms.Resize((self.window_size, self.window_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])  
        image = self.transform(image)
        return image, self.meta.iloc[item][2]

    def __len__(self):
        return len(self.meta.index)