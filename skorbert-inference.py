import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from detox.datasets import ScoredSentencePairDataset
from detox.models import SkorBERT as SkorBERTRegressor
from detox.preprocessing import collect_answers

#region Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)
parser.add_argument('--folders', nargs='+', default=[], required=True)
parser.add_argument('--model-params', type=json.loads, required=True)

args = vars(parser.parse_args())
#endregion

INPUT_PATH = args['input_path']
OUTPUT_PATH = Path(args['output_path'])

model_params = args['model_params']
MODEL_NAME = model_params['model_name']
MAX_LENGTH = model_params['max_length']
BATCH_SIZE = model_params['batch_size']

ensemble = 'model_dir' in model_params 

if ensemble:
    MODEL_DIR = Path(model_params['model_dir'])
else:
    MODEL_PATH = model_params['model_path']

#region Data Preparing
answers = []

for folder in args['folders']:
    collect_answers(answers, folder)

answers = pd.concat(answers)
answers.drop_duplicates(inplace=True)
answers.sort_values(by='group', inplace=True)

test_df = pd.read_csv(INPUT_PATH, sep='\t')
test_df['group'] = test_df.index

paired_df = answers.merge(test_df, on='group')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = ScoredSentencePairDataset(tokenizer,
    toxic_comments=paired_df.toxic_comment,
    hypotheses=paired_df.no_toxic,
    scores=paired_df.index,
    max_length=MAX_LENGTH
)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
#endregion

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#region Inference
if ensemble:
    predictions = defaultdict(list)

    for fold in range(5):
        model = torch.load(
            MODEL_DIR / f'model-{fold}.dump',
            map_location=torch.device('cpu')
        )
        model.eval()
        with torch.no_grad(), tqdm(loader) as progress_bar:
            for batch_index, batch in enumerate(progress_bar, 1):
                p = model(
                    batch['input_ids'].to(device, dtype=torch.long),
                    batch['attention_mask'].to(device, dtype=torch.long),
                    batch['token_type_ids'].to(device, dtype=torch.long)
                ).squeeze()
                predictions[fold].append(p)


    fold_scores = np.empty((paired_df.shape[0], 5), dtype=np.float32)

    for fold in range(5):
        fold_scores[:, fold] = np.hstack([p.cpu().numpy() for p in predictions[fold]])


    paired_df['-mean_score'] = -fold_scores.mean(axis=1)
    paired_df.sort_values(by=['group', '-mean_score'], kind='mergesort', inplace=True)
    paired_df.to_csv(OUTPUT_PATH / 'scored.tsv', sep='\t', index=False)

    submission = paired_df.groupby('group').head(1)
    submission.sort_values(by='group')[['no_toxic']].to_csv(OUTPUT_PATH / 'answer.txt', sep='\t', index=False)
else:
    model = torch.load(
        MODEL_PATH,
        map_location=torch.device('cpu')
    )
    model.eval()

    predictions = []
    
    with torch.no_grad(), tqdm(loader) as progress_bar:
        for batch_index, batch in enumerate(progress_bar, 1):
            p = model(
                batch['input_ids'].to(device, dtype=torch.long),
                batch['attention_mask'].to(device, dtype=torch.long),
                batch['token_type_ids'].to(device, dtype=torch.long)
            ).squeeze()
            predictions.append(p)

    predictions = np.hstack([p.cpu().numpy() for p in predictions])

    paired_df['-mean_score'] = -predictions
    paired_df.sort_values(by=['group', '-mean_score'], kind='mergesort', inplace=True)
    paired_df.to_csv(OUTPUT_PATH / 'scored.tsv', sep='\t', index=False)

    submission = paired_df.groupby('group').head(1)
    submission.sort_values(by='group')[['no_toxic']].to_csv(OUTPUT_PATH / 'answer.txt', sep='\t', index=False)
#endregion
