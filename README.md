
# Movies Genre Classification
Welcome to the MOVIES GENRE CLASSIFICATION repository. This project provides an easy-to-use framework of the classification model for Movies Genre.
## Requirements
All the dependencies can be installed using the provided requirements.txt file.
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
   Training: ./configs/train.yaml
   ```
5. Training:  
    ```bash
   python src/train.py 
    ```
        



