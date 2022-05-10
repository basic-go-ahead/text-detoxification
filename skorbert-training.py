import numpy as np
import pandas as pd

from tqdm import tqdm

import cloudpickle, json
from pathlib import Path

from typing import Any, Dict

import torch

from transformers import AutoTokenizer

from detox.datasets import ScoredSentencePairDataset
from detox.scoring import calc_metrics
from detox.models import SkorBERT
from detox.losses import cross_entropy

import argparse


#region Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--train-path', type=str, required=True)
parser.add_argument('--split-path', type=str)
parser.add_argument('--model-params', type=json.loads, required=True)
parser.add_argument('--output-path', type=str, required=True)
parser.add_argument('--save-model-when', nargs='+', type=int)

args = vars(parser.parse_args())
#endregion

TRAIN_PATH = args['train_path']
SPLIT_PATH = args['split_path']
OUTPUT_PATH = Path(args['output_path'])
SAVE_MODEL_WHEN = args['save_model_when']

model_params = args['model_params']
BATCH_SIZE = model_params['batch_size']
MODEL_NAME = model_params['model_name']
MAX_LENGTH = model_params['max_length']
LEARNING_RATE = model_params['learning_rate']
WEIGHT_DECAY = model_params['weight_decay']
N_EPOCHS = model_params['n_epochs']
MAX_FAILS = model_params['max_fails']
MODEL_PATH = model_params['model_path'] if 'model_path' in model_params else None
CONTROL_BOTH = model_params['control_both'] if 'control_both' in model_params else False
SHUFFLE = model_params['shuffle'] if 'shuffle' in model_params else True


#region Data Preparing
train_pared_df = pd.read_csv(TRAIN_PATH, sep='\t')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if SPLIT_PATH is not None:
    with open(SPLIT_PATH, 'rb') as file:
        split_info = cloudpickle.load(file)


    train_dfs, valid_dfs = [], []

    for fold, d in split_info.items():
        df = train_pared_df.loc[d['train_indices']]
        train_dfs.append(df)
        
        df = train_pared_df.loc[d['valid_indices']]
        valid_dfs.append(df)

    train_loaders, valid_loaders = [], []


    for train_df, valid_df in zip(train_dfs, valid_dfs):
        train_dataset = ScoredSentencePairDataset(tokenizer,
            toxic_comments=train_df.toxic_comment,
            hypotheses=train_df.no_toxic,
            scores=train_df.score,
            max_length=MAX_LENGTH
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=SHUFFLE, batch_size=BATCH_SIZE)
        train_loaders.append(train_loader)

        valid_dataset = ScoredSentencePairDataset(tokenizer,
            toxic_comments=valid_df.toxic_comment,
            hypotheses=valid_df.no_toxic,
            scores=valid_df.score,
            max_length=MAX_LENGTH)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)
        valid_loaders.append(valid_loader)
else:
    train_dataset = ScoredSentencePairDataset(tokenizer,
        toxic_comments=train_pared_df.toxic_comment,
        hypotheses=train_pared_df.no_toxic,
        scores=train_pared_df.score,
        max_length=MAX_LENGTH
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
#endregion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = cross_entropy

#region Training Loop Functions
def train(current_epoch: int, model, loader, optimizer, params: Dict[str, Any]):
    model.train()
    total_loss = 0.

    with tqdm(loader) as progress_bar:
        for batch_index, batch in enumerate(progress_bar, 1):
            optimizer.zero_grad()   
            output = model(
                batch['input_ids'].to(device, dtype=torch.long),
                batch['attention_mask'].to(device, dtype=torch.long),
                batch['token_type_ids'].to(device, dtype=torch.long)
            ).squeeze()
            loss = criterion(output.to(device, dtype=torch.float32), batch['target'].to(device, dtype=torch.float32))
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_index % 10 == 0:
                progress_bar.set_description(f'Epoch {current_epoch:02d}: mean_train_loss = {total_loss / batch_index:.05f}')

    return total_loss / batch_index
                
                
def validate(current_epoch, model, loader):
    model.eval()
    total_loss = 0.
    
    with torch.no_grad(), tqdm(loader) as progress_bar:
        for batch_index, batch in enumerate(progress_bar, 1):
            output = model(
                batch['input_ids'].to(device, dtype=torch.long),
                batch['attention_mask'].to(device, dtype=torch.long),
                batch['token_type_ids'].to(device, dtype=torch.long)
            ).squeeze()
            try:
                loss = criterion(output, batch['target'].to(device, dtype=torch.float32))
            except:
                break
            total_loss += loss.item()
            
            if batch_index % 10 == 0:
                progress_bar.set_description(f'Epoch {current_epoch:02d}: mean_valid_loss = {total_loss / batch_index:.05f}')
        
        valid_loss = total_loss / batch_index
        progress_bar.set_description(f'Epoch {current_epoch:02d}: mean_valid_loss = {valid_loss:.05f}')
        
    return valid_loss
#endregion

#region Training Loop
if SPLIT_PATH is not None:
    for fold, (train_loader, valid_loader) in enumerate(zip(train_loaders, valid_loaders)):
        print()
        print('-'*15)
        print(f'FOLD #{fold}')
        print()

        if MODEL_PATH is not None:
            model = torch.load(MODEL_PATH.format(fold))
        else:
            model = SkorBERT(MODEL_NAME)
        model.to(device)

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

        fails = 0
        prev_valid_loss = float('inf')
        prev_train_loss = float('inf')

        for current_epoch in range(1, N_EPOCHS + 1):
            train_loss = train(current_epoch, model, train_loader, optimizer, None)
            valid_loss = validate(current_epoch, model, valid_loader)

            if valid_loss < prev_valid_loss or (abs(valid_loss - prev_valid_loss) < 1e-4 and (not CONTROL_BOTH or train_loss < prev_train_loss)):
                print('\033[34mSaving model...\033[0m')
                prev_valid_loss = valid_loss
                prev_train_loss = train_loss
                torch.save(model, OUTPUT_PATH / f'model-{fold}.dump')
                fails = 0
            else:
                model = torch.load(OUTPUT_PATH / f'./model-{fold}.dump')
                fails += 1

                if fails == MAX_FAILS:
                    break

                scheduler.step()

                print('\033[32mReducing LR to {0}\033[0m'.format(optimizer.param_groups[0]['lr']))
else:
    if MODEL_PATH is not None:
        model = torch.load(MODEL_PATH)
    else:
        model = SkorBERT(MODEL_NAME)
    model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    for current_epoch in range(1, N_EPOCHS + 1):
        train_loss = train(current_epoch, model, train_loader, optimizer, None)

        if current_epoch in SAVE_MODEL_WHEN:        
            torch.save(model, f'./model-{current_epoch:02d}.dump')

    torch.save(model, OUTPUT_PATH / f'model.dump')
#endregion

if SPLIT_PATH is not None:
    for fold in range(len(train_loaders)):
        model = torch.load(OUTPUT_PATH / f'./model-{fold}.dump')
        model.eval()

        valid_loader = valid_loaders[fold]
        pred_scores = []
        embeddings = []

        with torch.no_grad(), tqdm(valid_loader) as progress_bar:
            for batch_index, batch in enumerate(progress_bar, 1):
                p = model(
                    batch['input_ids'].to(device, dtype=torch.long),
                    batch['attention_mask'].to(device, dtype=torch.long),
                    batch['token_type_ids'].to(device, dtype=torch.long)
                ).squeeze()
                pred_scores.append(p.cpu().numpy())
                
                batch_embeddings = model._model(
                    batch['input_ids'].to(device, dtype=torch.long),
                    batch['attention_mask'].to(device, dtype=torch.long),
                    batch['token_type_ids'].to(device, dtype=torch.long)
                ).logits.cpu().numpy()
                
                embeddings.append(batch_embeddings)
                
        pred_scores = np.hstack(pred_scores)
        embeddings = np.vstack(embeddings)
        
        with open(OUTPUT_PATH / f'./embeddings-{fold}.pickle', 'wb') as file:
            cloudpickle.dump(embeddings, file)
            
        with open(OUTPUT_PATH / f'./scores-{fold}.pickle', 'wb') as file:
            cloudpickle.dump(pred_scores, file)
        
        valid_df = valid_dfs[fold]
        valid_df['pred_score'] = pred_scores
        less = valid_df[valid_df['score'] < 1.0]
        
        print(f'MODEL #{fold}')
        print('\033[34mFULL:\033[0m', calc_metrics(valid_df, 'score', 'pred_score'))
        print('\033[34mLESS:\033[0m', calc_metrics(less, 'score', 'pred_score'))
        print()