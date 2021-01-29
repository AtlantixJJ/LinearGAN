import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class NoiseDataset(Dataset):
  def __init__(self, epoch_size=1024, latent_size=512, fixed=False):
    super().__init__()
    self.latent_size = latent_size
    self.epoch_size = epoch_size
    self.fixed = fixed
    if fixed:
      self.z = torch.randn(epoch_size, latent_size)
  
  def __getitem__(self, idx):
    if self.fixed:
      return self.z[idx]
    else:
      return torch.randn(self.latent_size)
  
  def __len__(self):
    return self.epoch_size


class NoiseDataModule(pl.LightningDataModule):
  def __init__(self, train_size=1024, val_size=1024, latent_size=512, batch_size=64):
    super().__init__()
    self.batch_size = batch_size
    self.train_ds = NoiseDataset(train_size, latent_size)
    self.val_ds = NoiseDataset(val_size, latent_size, fixed=True)
    self.test_ds = NoiseDataset(epoch_size, latent_size)

  def train_dataloader(self):
    return DataLoader(self.train_ds, batch_size=self.batch_size)

  def val_dataloader(self):
    return DataLoader(self.val_ds, batch_size=self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_ds, batch_size=self.batch_size)