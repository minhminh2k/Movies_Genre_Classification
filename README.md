
# Movies Genres Multi-Label Classification
Welcome to the MOVIES GENRE CLASSIFICATION repository. This project provides an easy-to-use framework of the classification model for Movies Genre.

## Members
- Dương Quang Minh - 21020219
- Lê Văn Bảo - 21020171
- Tống Minh Trí - 21020249

## Description
Final Project for Machine Learning Class INT3405E 21

This repo was made by UET-VNU students

Topic: Movies Genres Multi-Label Classification using poster or combining poster and movie title

Main Technologies: PyTorch, Lightning, Hydra, WanDB

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/minhminh2k/Movies_Genre_Classification.git
   ```
2. Change the directory:
   ```
   cd Movies_Genre_Classification
   ```
3. Create a conda environment and install dependencies:
   ```
   conda create -n movies python=3.8
   ```
4. Activate the conda environment:
   ```
   conda activate movies
   ```
5. Install necessary dependencies
   ```
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
   pip install -r requirements.txt
   ```
## Dataset
1. Download ML1M Dataset:
   ```
   gdown https://drive.google.com/uc?id=1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD 
   ```
2. Unzip Dataset
   ```
   unzip ml1m.zip
   ```
## How to run

#### Train model with default configuration

```
# before training, set env WANDB_API_KEY to log with wandb logger
export WANDB_API_KEY = ${WANDB_API_KEY}

# train
python src/train.py
```
#### Train with other models
- The configuration can be modified using [Hydra](https://hydra.cc/)
1. Customize dataloader for your dataset:
   ```bash
    File path: ./src/data/{your_datamodule}.py
   ```
2. Add your model architecure:
   ```bash
    File path: ./src/models/{your_module}.py
   ```
3. Edit visualization with Wandb:
   ```bash
    File path: ./src/utils/callbacks/wandb_callback_ml1m.py
   ```
4. Modify config file:  
   ```bash
   Dataset: ./configs/data/{your_dataset}.yaml
   ```
   ```bash
   Model: ./configs/model/{your_model}.yaml
   ```
   ```bash
   Training: ./configs/train.yaml
   ```
5. Training:  
    ```bash
   python src/train.py 
    ```
        
## Results: 
- Models using poster or combining poster and title as input: [Wandb](https://wandb.ai/minhqd9112003/movielens?workspace=user-minhqd9112003).
- Models using movie title as input, the result of BaseModel, the result of combined model using LSTM for text and CNN for poster: [Google Colab](https://colab.research.google.com/drive/1RMRjzu_gKKmLiTXc69HBZ7yJf4tJrq57?usp=sharing)


