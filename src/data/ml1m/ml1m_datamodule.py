from typing import Any, Dict, Optional, Tuple

import albumentations as A
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from data.ml1m.components.ml1m import Ml1mDataset
from data.ml1m.components.transform_ml1m import TransformMl1m
from data.ml1m.components.ml1m_combine import Ml1mCombineDataset
from data.ml1m.components.transform_ml1m_combine import TransformMl1mCombine


class Ml1mDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "content/dataset",
        data_type: str = "img",
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        transform_train: Optional[A.Compose] = None,
        transform_val: Optional[A.Compose] = None,
        n_length: int = 4,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        

    def setup(self, visualize_dist=False, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            if self.hparams.data_type == "img":
                dataset = Ml1mDataset(
                    data_dir=self.hparams.data_dir,
                )
                # Try catch block for stratified splits
                data_len = len(dataset)
                dframe = dataset.data
                '''
                train_len = int(data_len * self.hparams.train_val_test_split[0])
                val_len = int(data_len * self.hparams.train_val_test_split[1])
                test_len = data_len - train_len - val_len

                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=[train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(42),
                )

                print("Using random_split.")
                '''
            
                data_train_val = Subset(dataset, range(0, 3106))
                train_val_len = len(data_train_val)
                train_len = int(train_val_len * 0.9)
                val_len = train_val_len - train_len
                
                self.data_train, self.data_val = random_split(
                    dataset=data_train_val,
                    lengths=[train_len, val_len],
                    generator=torch.Generator().manual_seed(42),
                )
            
                self.data_test = Subset(dataset, range(3106, 3883))
                
                # create transform dataset from subset
                self.data_train = TransformMl1m(self.data_train, self.hparams.transform_train)
                self.data_val = TransformMl1m(self.data_val, self.hparams.transform_val)
                self.data_test = TransformMl1m(self.data_test, self.hparams.transform_val)
            
            elif self.hparams.data_type == "combine":
                dataset = Ml1mCombineDataset(
                    data_dir=self.hparams.data_dir,
                    n_length=self.hparams.n_length,
                )
                data_len = len(dataset)
                dframe = dataset.data
                data_train_val = Subset(dataset, range(0, 3106))
                train_val_len = len(data_train_val)
                train_len = int(train_val_len * 0.9)
                val_len = train_val_len - train_len
                
                self.data_train, self.data_val = random_split(
                    dataset=data_train_val,
                    lengths=[train_len, val_len],
                    generator=torch.Generator().manual_seed(42),
                )
            
                self.data_test = Subset(dataset, range(3106, 3883))
                
                # create transform dataset from subset
                self.data_train = TransformMl1mCombine(self.data_train, self.hparams.transform_train)
                self.data_val = TransformMl1mCombine(self.data_val, self.hparams.transform_val)
                self.data_test = TransformMl1mCombine(self.data_test, self.hparams.transform_val)
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf

    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs")
    output_path = path / "outputs"
    print(f"config_path: {config_path}")
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(OmegaConf.to_yaml(cfg.data, resolve=True))

        airbus = hydra.utils.instantiate(cfg.data)
        airbus.setup(
            visualize_dist=False
        )  # set visualize_dist to True to see distribution of train, val & test set

    main()