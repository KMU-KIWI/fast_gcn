import pytorch_lightning as pl

from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from fast_gcn.models import SGN


class LitSGN(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 60,
        length: int = 30,
        num_joints: int = 25,
        num_features: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = SGN(num_classes, length, num_joints, num_features)

        self.loss = CrossEntropyLoss(label_smoothing=0.1)

        self.train_metric = Accuracy()
        self.val_metric = Accuracy()
        self.test_metric = Accuracy()

        self.train_class_metric = Accuracy(average="none", num_classes=num_classes)
        self.val_class_metric = Accuracy(average="none", num_classes=num_classes)
        self.test_class_metric = Accuracy(average="none", num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
        )

        grad_batches = self.trainer.accumulate_grad_batches
        if grad_batches is None:
            grad_batches = 1
        steps_per_epoch = (
            len(self.trainer.datamodule.train_dataloader()) // grad_batches
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.hparams.pct_start,
            anneal_strategy=self.hparams.anneal_strategy,
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "name": "learning_rate",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def _compute_loss(self, x, labels):
        logits = self.model(x)
        loss = self.loss(logits, labels)
        return loss, logits

    def _step(self, batch, stage: str):
        x, edge_index, labels = batch
        loss, logits = self._compute_loss(x, labels)

        self.log(f"{stage}_loss", loss)

        preds = logits.argmax(dim=1)
        if stage == "train":
            self.train_metric.update(preds, labels.int())
            self.train_class_metric.update(preds, labels.int())
        elif stage == "val":
            self.val_metric.update(preds, labels.int())
            self.val_class_metric.update(preds, labels.int())
        elif stage == "test":
            self.test_metric.update(preds, labels.int())
            self.test_class_metric.update(preds, labels.int())

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, stage="test")

    def _log_metric(self, stage):
        if stage == "train":
            acc = self.train_metric.compute()
        elif stage == "val":
            acc = self.val_metric.compute()
        elif stage == "test":
            acc = self.test_metric.compute()

        self.log(f"{stage}_acc", acc, prog_bar=True)

        if stage == "train":
            class_acc = self.train_class_metric.compute()
        elif stage == "val":
            class_acc = self.val_class_metric.compute()
        elif stage == "test":
            class_acc = self.test_class_metric.compute()
        for i, acc in enumerate(class_acc.tolist()):
            self.log(f"{stage}_{i}_acc", acc)

        if stage == "train":
            self.train_metric.reset()
            self.train_class_metric.reset()
        elif stage == "val":
            self.val_metric.reset()
            self.val_class_metric.reset()
        elif stage == "test":
            self.test_metric.reset()
            self.test_class_metric.reset()

    def training_epoch_end(self, outputs):
        self._log_metric(stage="train")

    def validation_epoch_end(self, outputs):
        self._log_metric(stage="val")

    def test_epoch_end(self, outputs):
        self._log_metric(stage="test")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitClassifier")
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-8)
        parser.add_argument("--pct_start", type=float, default=0.0)
        parser.add_argument("--anneal_strategy", type=str, default="cos")
