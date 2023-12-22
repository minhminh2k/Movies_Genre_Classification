import gc
from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric

from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall



class Ml1mCombineLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = nn.BCELoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_metric_1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='macro')
        self.val_metric_1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='macro')
        self.test_metric_1 = MultilabelF1Score(num_labels=18, threshold=0.5, average='macro')
        
        self.train_metric_2 = MultilabelAccuracy(num_labels=18, threshold=0.5, average='macro')
        self.val_metric_2 = MultilabelAccuracy(num_labels=18, threshold=0.5, average='macro')
        self.test_metric_2 = MultilabelAccuracy(num_labels=18, threshold=0.5, average='macro')
        
        self.train_metric_3 = MultilabelAveragePrecision(num_labels=18, average='macro')
        self.val_metric_3 = MultilabelAveragePrecision(num_labels=18, average='macro')
        self.test_metric_3 = MultilabelAveragePrecision(num_labels=18, average='macro')

        self.train_metric_4 = MultilabelPrecision(num_labels=18, average='macro')
        self.val_metric_4 = MultilabelPrecision(num_labels=18, average='macro')
        self.test_metric_4 = MultilabelPrecision(num_labels=18, average='macro')
        
        self.train_metric_5 = MultilabelRecall(num_labels=18, average='macro')
        self.val_metric_5 = MultilabelRecall(num_labels=18, average='macro')
        self.test_metric_5 = MultilabelRecall(num_labels=18, average='macro')
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best_1 = MaxMetric()
        self.val_metric_best_2 = MaxMetric()
        self.val_metric_best_3 = MaxMetric()
        self.val_metric_best_4 = MaxMetric()
        self.val_metric_best_5 = MaxMetric()
        

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.net(x, y)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric_1.reset()
        self.val_metric_2.reset()
        self.val_metric_3.reset()
        self.val_metric_4.reset()
        self.val_metric_5.reset()
        self.val_metric_best_1.reset()
        self.val_metric_best_2.reset()
        self.val_metric_best_3.reset()
        self.val_metric_best_4.reset()
        self.val_metric_best_5.reset()
        
    def model_step(self, batch: Any):
        x, text, y = batch[0], batch[1], batch[2]
        
        preds = self.forward(text, x)
        loss = self.criterion(preds, y)
        
        # Code to try to fix CUDA out of memory issues
        del x
        gc.collect()
        torch.cuda.empty_cache()

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_metric_1(preds, targets)
        self.train_metric_2(preds, targets)
        self.train_metric_3(preds, targets.int())
        self.train_metric_4(preds, targets)
        self.train_metric_5(preds, targets)
        
    
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_metric_2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/AP", self.train_metric_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_metric_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_metric_5, on_step=False, on_epoch=True, prog_bar=True)
        
        

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_metric_1(preds, targets)
        self.val_metric_2(preds, targets)
        self.val_metric_3(preds, targets.int())
        self.val_metric_4(preds, targets)
        self.val_metric_5(preds, targets)

        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_metric_2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/AP", self.val_metric_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_metric_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_metric_5, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # get current val acc
        acc1 = self.val_metric_1.compute()
        acc2 = self.val_metric_2.compute()
        acc3 = self.val_metric_3.compute()
        acc4 = self.val_metric_4.compute()
        acc5 = self.val_metric_5.compute()
        
        # update best so far val acc
        self.val_metric_best_1(acc1)  
        self.val_metric_best_2(acc2)  
        self.val_metric_best_3(acc3)  
        self.val_metric_best_4(acc4)  
        self.val_metric_best_5(acc5)  
        
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_metric_best_1.compute(), prog_bar=True)
        self.log("val/acc_best", self.val_metric_best_2.compute(), prog_bar=True)
        self.log("val/ap_best", self.val_metric_best_3.compute(), prog_bar=True)
        self.log("val/precision_best", self.val_metric_best_4.compute(), prog_bar=True)
        self.log("val/recall_best", self.val_metric_best_5.compute(), prog_bar=True)
        
        

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metric_1(preds, targets)
        self.test_metric_2(preds, targets)
        self.test_metric_3(preds, targets.int())
        self.test_metric_4(preds, targets)
        self.test_metric_5(preds, targets)
        
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_metric_2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/AP", self.test_metric_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_metric_4, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_metric_5, on_step=False, on_epoch=True, prog_bar=True)

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf

    # find paths
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

    config_path = str(path / "configs")
    print(f"project-root: {path}")
    print(f"config path: {config_path}")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(f"config: \n {OmegaConf.to_yaml(cfg.model, resolve=True)}")

        model = hydra.utils.instantiate(cfg.model)
        batch = torch.rand(1, 3, 256, 256)
        output = model(batch)

        print(f"output shape: {output.shape}")  # [1, 1, 256, 256]

    main()