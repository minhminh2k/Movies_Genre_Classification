from typing import Any, Optional

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from data.ml1m.components.ml1m import Ml1mDataset

import torch


class TransformMl1m(Dataset):
    mean = None
    std = None

    def __init__(self, dataset: Ml1mDataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        # image, mask, label, file_id = self.dataset[index]  # (768, 768, 3), (768, 768)
        image, genre = self.dataset[index] # (445, 300, 3) (18, )
       
        if self.transform is not None:
            transformed = self.transform(image=image)
            # img_size set in hydra config
            image = transformed["image"]  # (3, img_size, img_size)
            genre = torch.from_numpy(genre)
            genre = genre.float()#.unsqueeze(0)  # (1, img_size, img_size)

        return image, genre 
    
    
if __name__ == "__main__":
    
    ml1m = TransformMl1m(Ml1mDataset())
    img, genre = ml1m[2]
    print(img.shape) # 3, 445, 300
    print(type(img)) # Tensor
    print(genre.shape) # [18]