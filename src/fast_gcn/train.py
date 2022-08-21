import argparse

import torch
from thop import profile

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

import wandb

from fast_gcn.litmodules import LitSGN
import fast_gcn.datamodules as dm


def train(args):
    dict_args = vars(args)

    if args.dataset == "ntu60":
        data = dm.NTU60DataModule(**dict_args)
    elif args.dataset == "ntu120":
        data = dm.NTU120DataModule(**dict_args)

    model = LitSGN(num_features=data.num_features, **dict_args)

    wandb_logger = WandbLogger(
        name=f"{args.dataset}_{args.benchmark_type}_{args.joint_type}",
        project="small_sgn",
        save_dir=None,
        log_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        dirpath=None,
        filename="{epoch}-{val_acc:.6f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            RichModelSummary(),
            RichProgressBar(),
            checkpoint_callback,
            lr_monitor,
        ],
        logger=wandb_logger,
        min_epochs=16,
    )

    x = torch.randn(1, 30, 2, 25, 3)

    macs, params = profile(model.model, inputs=(x,))

    print(f"{macs / 1e9} GMACS  {params / 1e6}M params")

    trainer.fit(model, data)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="ntu60")

    LitSGN.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    dm.add_data_specific_args(parser)

    args = parser.parse_args()

    train(args)
