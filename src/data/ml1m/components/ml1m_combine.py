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


class Ml1mCombineDataset(Dataset):
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
        
        '''
        movies_train_path = movies_train[movies_train['img_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]
        movies_test_path = movies_test[movies_test['img_path'].apply(lambda x: os.path.exists(x) if pd.notna(x) else False)]

        movies_combine_path = pd.concat([movies_train_path, movies_test_path], axis=0)
        '''
        
        
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
        
        # create vocab
        vocab = create_vocab(movies_train[0:2706])
        
        self.len_vocab = len(vocab)
        
        pad_token = '<PAD>'
        unk_token = '<UNK>'
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}

        # Create a binary vector for each word in each sentence
        MAX_LENGTH = n_length # Max length title 4
        vectors = []
        for title_tokens in self.data.title_tokens.tolist():
            if len(title_tokens) < MAX_LENGTH:
                num_pad = MAX_LENGTH - len(title_tokens)
                for idx in range(num_pad):
                    title_tokens.append(pad_token)
            else:
                title_tokens = title_tokens[:MAX_LENGTH]
            title_vectors = []
            for word in title_tokens:
                binary_vector = np.zeros(len(vocab))
                if word in vocab:
                    binary_vector[self.token2idx[word]] = 1
                else:
                    binary_vector[self.token2idx[unk_token]] = 1
                title_vectors.append(binary_vector)

            vectors.append(np.array(title_vectors))
        self.data['vectors'] = vectors # Binary vector
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        title = self.data.iloc[index].title
        genre = self.data.iloc[index].genre

        # preprocess img
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = np.random.rand(256,256,3)

        # preprocess text
        title_vector = self.data.iloc[index].vectors


        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))

        for g in genre:
            genre_vector[self.genre2idx[g]] = 1

        return np.array(img, dtype=np.float32), np.array(title_vector, dtype=np.float32), np.array(genre_vector, dtype=np.float32)
    
        
if __name__ == "__main__":
    ml1m = Ml1mCombineDataset(n_length=4)

    print(ml1m.data.iloc[0]) 
    print(ml1m.data.iloc[3106]) 
    print(ml1m.data.iloc[3882]) 
    
    print(ml1m.len_vocab)

    