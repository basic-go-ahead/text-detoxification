import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer

from detox.datasets import DetoxificationDataset
from detox.preprocessing import preprocess
from detox.scoring import BLEUScorer

#region Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--max-length', type=int, required=True)
parser.add_argument('--toxic-path', type=str, required=True)
parser.add_argument('--detoxic-path', type=str, required=True)
parser.add_argument('--compare-with', type=str)

args = vars(parser.parse_args())
#endregion

TOXIC_PATH = args['toxic_path']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#region Data Preparing
source_df = preprocess(pd.read_csv(TOXIC_PATH, sep='\t', keep_default_na=False))

tokenizer = T5Tokenizer.from_pretrained(args['model_name'])
dataset = DetoxificationDataset(tokenizer, df=source_df, max_length=args['max_length'])
loader = DataLoader(dataset, batch_size=20)
#endregion

def run_generator(model, tokenizer, loader, n_additional_tokens: int=0):
    model.eval()
    with torch.no_grad(), tqdm(loader) as progress_bar:
        for batch in progress_bar:
            source_ids = batch['source_ids'].to(model.device, dtype=torch.long)
            source_mask = batch['source_mask'].to(model.device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids = source_ids,
                attention_mask = source_mask, 
                eos_token_id=tokenizer.eos_token_id,
                max_length=source_mask.sum() + n_additional_tokens,
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
            )
            
            tokens = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            
            yield from tokens

#region Inference
model = torch.load(args['model_path'], map_location=device)

hypotheses = list(run_generator(model, tokenizer, loader))

pd.DataFrame({'no_toxic': hypotheses}).to_csv(args['detoxic_path'], sep='\t', index=False)
#endregion

if args['compare_with'] is not None:
    compare_df = preprocess(pd.read_csv(args['compare_with'], sep='\t', keep_default_na=False))
    scorer = BLEUScorer(compare_df)

    bleu = np.mean(scorer.get_scores(hypotheses))
    print('BLEU: {0:.5f}'.format(bleu))
    uncased_bleu = np.mean(scorer.get_scores(hypotheses, uncased=True))
    print('uncased BLEU: {0:.5f}'.format(uncased_bleu))
