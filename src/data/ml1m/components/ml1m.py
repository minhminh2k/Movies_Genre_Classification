import shutil
import zipfile
import os 
import numpy as np
import pandas as pd

from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import torch

class Ml1mDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "content/dataset",
    ) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        
        movies_train = pd.read_csv('content/dataset/movies_train.dat', engine='python',
                                sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
        movies_test = pd.read_csv('content/dataset/movies_test.dat', engine='python',
                                sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
        movies_train['genre'] = movies_train.genre.str.split('|')
        movies_test['genre'] = movies_test.genre.str.split('|')
        
        folder_img_path = 'content/dataset/ml1m-images'
        movies_train['id'] = movies_train.index
        movies_train.reset_index(inplace=True)
        movies_train['img_path'] = movies_train.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)
        
        movies_test['id'] = movies_test.index
        movies_test.reset_index(inplace=True)
        movies_test['img_path'] = movies_test.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)
        
        '''
        movies_train_path = movies_train[movies_train['img_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]
        movies_test_path = movies_test[movies_test['img_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]

        movies_combine_path = pd.concat([movies_train_path, movies_test_path], axis=0)
        '''
        
        movies_combine_path = pd.concat([movies_train, movies_test], axis=0)
        
        # label genre
        with open('content/dataset/genres.txt', 'r') as f:
            genre_all = f.readlines()
            genre_all = [x.replace('\n','') for x in genre_all]
        self.genre2idx = {genre:idx for idx, genre in enumerate(genre_all)} # Dictionary
        
        self.data = movies_combine_path
        
        self.len_data = len(movies_combine_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        genre = self.data.iloc[index].genre

        # preprocess img
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = np.random.rand(256,256,3)
  
        # img_tensor = torch.from_numpy(img).float()

        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        # genre_tensor = torch.from_numpy(genre_vector).float()

        return np.array(img, dtype=np.float32), np.array(genre_vector, dtype=np.float32)
    
        
if __name__ == "__main__":
    ml1m = Ml1mDataset()
    print(len(ml1m)) # 3256
    print(ml1m.data.iloc[2].title) # Batman Returns (1992)
    print(ml1m.data.iloc[2]['title']) # Batman Returns (1992)
    
    print(ml1m.data.iloc[2].img_path) # 1377.jpg
    # print(ml1m.data.index.id) # error
    img, genre = ml1m[2]
    print(img.shape) # 445, 300, 3
    # print(genre.shape) # (18,)
    
    abc = Subset(ml1m, range(3106, 3200))
    x, _ = abc[1]
    y, _ = abc[2]
    print(x.shape)
    print(y.shape)
    # print(len(abc)) # 350
    