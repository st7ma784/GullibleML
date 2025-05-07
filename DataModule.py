import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with filenames and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, root_dir, batch_size=32, num_workers=4):
        """
        Args:
            csv_file (str): Path to the CSV file with filenames and labels.
            root_dir (str): Directory with all the images.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),  # Resize to 800x800
            transforms.ToTensor(),         # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
        ])

    def setup(self, stage=None):
        """
        Called on every GPU separately - setup data for train, val, test splits.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ImageDataset(
                csv_file=self.csv_file,
                root_dir=self.root_dir,
                transform=self.transform
            )
            self.val_dataset = ImageDataset(
                csv_file=self.csv_file,
                root_dir=self.root_dir,
                transform=self.transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageDataset(
                csv_file=self.csv_file,
                root_dir=self.root_dir,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, prefetch_factor=3,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)