import shutil
import zipfile
import os 
import numpy as np
import pandas as pd

from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import torch

from nltk import wordpunct_tokenize
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from transformers import AutoTokenizer
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

# Remove stopword from text
def remove_stop_words(text):
    text = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word.isalpha() and not word in stop_words]
    text = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in text]
    text = [word for word in text if not re.match(r'^[ivxlcdm]+$', word)]
    text = [lemmatizer.lemmatize(w) for w in text]

    return ' '.join(text)

def tokenize(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = wordpunct_tokenize(text)
    # tokens = tokens[:-1] # remove last token because it is the year which maybe is not useful
    return tokens

def create_vocab(data_train):
    df = data_train.copy()
    arr_title = df['title_new'].tolist()
    vocab = set()
    for title in arr_title:
        tokens = tokenize(title)
        vocab.update(tokens)
    vocab = list(vocab)
    pad_token = '<PAD>'
    unk_token = '<UNK>'
    vocab.append(pad_token)
    vocab.append(unk_token)
    return vocab


class Ml1mCombineToken(Dataset):
    def __init__(
        self,
        data_dir: str = "content/dataset",
        n_length: int = 4,
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
        
        movies_train['title_new'] = [remove_stop_words(x) for x in movies_train.title]
        movies_test['title_new'] = [remove_stop_words(x) for x in movies_test.title]
                
        movies_combine_path = pd.concat([movies_train, movies_test], axis=0)
        
        # label genre
        with open('content/dataset/genres.txt', 'r') as f:
            genre_all = f.readlines()
            genre_all = [x.replace('\n','') for x in genre_all]
        self.genre2idx = {genre:idx for idx, genre in enumerate(genre_all)} # Dictionary
        
        self.data = movies_combine_path
        
        self.data['title_tokens'] = [tokenize(x) for x in self.data.title_new]
        
        self.len_data = len(movies_combine_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        
        self.transform = Compose(
                [
                    A.Resize(256, 256),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        title = self.data.iloc[index].title
        genre = self.data.iloc[index].genre

        # preprocess img
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = np.random.rand(256,256,3)
        image = np.array(image, dtype=np.uint8)
        
        transformed = self.transform(image=image)
        image = transformed["image"]
        
        # preprocess text        
        title_vector = self.tokenizer(title, return_tensors='pt', padding='max_length')
            
        title_vector = {key: torch.squeeze(value, 0) for key, value in title_vector.items()}
        

        # preprocess label
        genre_vector = torch.zeros(len(self.genre2idx), dtype=torch.float)

        for g in genre:
            genre_vector[self.genre2idx[g]] = 1

        return image, title_vector, genre_vector
    
        
if __name__ == "__main__":
    ml1m = Ml1mCombineToken()
    img, title, genre = ml1m[0]
    print(img.shape)
    print(title)

    