from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.trajair_dataset import TrajAirDataset
from utils.data import seq_collate


class TrajAirDataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        obs: int,
        preds: int,
        preds_step: int,
        delim: str,
        train_batch_size: int,
        val_batch_size: int,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super(TrajAirDataModule, self).__init__()
        self.root = root
        self.obs_len = obs
        self.pred_len = preds
        self.step = preds_step
        self.delim = delim
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = TrajAirDataset(
            self.root + "train", obs_len=self.obs_len, pred_len=self.pred_len, step=self.step, delim=self.delim

        )
        self.val_dataset = TrajAirDataset(
            self.root + "test", obs_len=self.obs_len, pred_len=self.pred_len, step=self.step, delim=self.delim
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            pin_memory=self.pin_memory,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            persistent_workers=self.persistent_workers,
            collate_fn=seq_collate
        )
    

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            pin_memory=self.pin_memory,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=seq_collate
        )
