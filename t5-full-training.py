import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration, T5Tokenizer

from detox.preprocessing import preprocess, expand
from detox.fine_tuning import run_finetuning
from detox.datasets import DetoxificationDataset

import argparse

#region Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--model-path', type=str)
parser.add_argument('--max-length', type=int, required=True)
parser.add_argument('--train-path', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--save-model-when', nargs='+', type=int)

args = vars(parser.parse_args())
#endregion

MODEL_NAME = args['model_name']
TRAIN_PATH = args['train_path']
MAX_LENGTH = args['max_length']
N_EPOCHS = args['epochs']
MODEL_PATH = args['model_path']

use_pretrained = args['model_path'] is None

labeled_df = preprocess(pd.read_csv(TRAIN_PATH, sep='\t', keep_default_na=False))
expanded_df = expand(labeled_df)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

train_dataset = DetoxificationDataset(tokenizer, df=expanded_df, max_length=MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

if use_pretrained:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME) 
else:
    model = torch.load(MODEL_PATH)

model_params = dict(
    LEARNING_RATE=1e-5,
    WEIGHT_DECAY=1e-3,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=model_params["LEARNING_RATE"],
    weight_decay=model_params['WEIGHT_DECAY']
)

run_finetuning(model, model_params,
    optimizer, train_loader, N_EPOCHS,
    save_model_when=args['save_model_when'])