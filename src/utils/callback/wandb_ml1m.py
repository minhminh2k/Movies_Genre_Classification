import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

class WandbCallback_Ml1m(Callback):
    def __init__(
        self,
        image_id: str = "1377.jpg",
        data_path: str = "content/dataset",
        n_images_to_log: int = 5,
        img_size: int = 256,
    ):
        self.img_size = img_size
        self.n_images_to_log = n_images_to_log  # number of logged images when eval

        self.four_first_preds = []
        self.four_first_targets = []
        self.four_first_batch = []
        self.four_first_image = []
        self.show_pred = []
        self.show_target = []

        self.batch_size = 1
        self.num_samples = 8
        self.num_batch = 0

        image_path = os.path.join(data_path, "ml1m-images")
        image_path = os.path.join(image_path, image_id)

        self.sample_image = np.array(Image.open(image_path).convert("RGB"))
        self.sample_image_height, self.sample_image_width = (
            self.sample_image.shape[0],
            self.sample_image.shape[1],
        )

        self.transform = Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]

        pass

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass