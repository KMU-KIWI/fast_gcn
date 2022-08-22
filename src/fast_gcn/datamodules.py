from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

from typing import Union, List, Optional

from fast_gcn.data import NTU60, NTU120
from fast_gcn import transforms


class NTU60DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        eval_batch_size: int = 64,
        benchmark_type: str = "xsub",
        joint_type: Union[List[str], str] = "3d",
        max_bodies: int = 2,
        length: int = 30,
        length_threshold: int = 11,
        spread_threshold: float = 0.8,
        num_workers: int = 0,
        download: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["eval_batch_size", "num_workers", "download"])

        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.download = download

        self.train_transform = transforms.SampleFrames(length)
        self.test_transform = transforms.SampleFrames(length, 10)

    @property
    def num_features(self) -> int:
        if self.hparams.joint_type == "3d":
            return 3
        else:
            return 2

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = NTU60(
                root=self.hparams.data_dir,
                benchmark=self.hparams.benchmark_type,
                split="train",
                joint_type=self.hparams.joint_type,
                max_bodies=self.hparams.max_bodies,
                transform=self.train_transform,
                download=self.download,
            )
            self.val_set = NTU60(
                root=self.hparams.data_dir,
                benchmark=self.hparams.benchmark_type,
                split="val",
                joint_type=self.hparams.joint_type,
                max_bodies=self.hparams.max_bodies,
                transform=self.train_transform,
                download=self.download,
            )
        else:
            self.test_set = NTU60(
                root=self.hparams.data_dir,
                benchmark=self.hparams.benchmark_type,
                split="val",
                joint_type=self.hparams.joint_type,
                max_bodies=self.hparams.max_bodies,
                transform=self.test_transform,
                download=self.download,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_size=self.eval_batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_size=self.eval_batch_size,
        )


class NTU120DataModule(NTU60DataModule):
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = NTU120(
                root=self.hparams.data_dir,
                benchmark=self.benchmark,
                split="train",
                joint_type=self.joint_type,
                max_bodies=self.max_bodies,
                transform=self.train_transform,
                download=self.download,
            )
            self.val_set = NTU120(
                root=self.hparams.data_dir,
                benchmark=self.benchmark,
                split="val",
                joint_type=self.joint_type,
                max_bodies=self.max_bodies,
                transform=self.train_transform,
                download=self.download,
            )
        else:
            self.test_set = NTU120(
                root=self.hparams.data_dir,
                benchmark=self.hparams.benchmark_type,
                split="val",
                joint_type=self.hparams.joint_type,
                max_bodies=self.hparams.max_bodies,
                transform=self.test_transform,
                download=self.download,
            )


def add_data_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("NTU120")

    parser.add_argument("--data_dir", type=str, default=".")

    parser.add_argument("--benchmark_type", type=str, default="xsub")
    parser.add_argument("--joint_type", type=str, default="3d")
    parser.add_argument("--max_bodies", type=int, default=2)
    parser.add_argument("--length", type=int, default=30)

    parser.add_argument("--length_threshold", type=int, default=11)
    parser.add_argument("--spread_threshold", type=float, default=0.8)

    parser.add_argument("--download", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
