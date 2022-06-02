from typing import Iterable, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class DetoxificationDataset(Dataset):
    def __init__(self, tokenizer, *, df: pd.DataFrame, max_length: int, verbose: bool=True):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._tokenized_comments = []
        self._tokenized_targets = []
        
        self._contains_targets = 'neutral_comment1' in df.columns
        
        tokenizer_params = dict(
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'   
        )
        
        for e in tqdm(df.itertuples(), disable=not verbose, total=len(df)):
            encoded_input = tokenizer(e.toxic_comment, **tokenizer_params)
            self._tokenized_comments.append(encoded_input)
            
            if self._contains_targets:
                encoded_output = tokenizer(e.neutral_comment1, **tokenizer_params)
                self._tokenized_targets.append(encoded_output)
            
    
    def __getitem__(self, index: int):
        if self._contains_targets:
            target_data = {
                'target_ids': self._tokenized_targets[index]['input_ids'].squeeze(),
                'target_mask': self._tokenized_targets[index]['attention_mask'].squeeze()
            }
        else:
            target_data = {}
        
        return {
            'source_ids': self._tokenized_comments[index]['input_ids'].squeeze(),
            'source_mask': self._tokenized_comments[index]['attention_mask'].squeeze(),
            **target_data
        }


    def __len__(self):
        return len(self._tokenized_comments)


class ScoredSentencePairDataset(Dataset):
    def __init__(self,
        tokenizer,
        *,
        toxic_comments: Iterable[str],
        hypotheses: Iterable[str],
        max_length: int,
        scores: Union[Iterable[int], None]=None,
        verbose: bool=True):
        self._tokenizer = tokenizer
        self._encoded_pairs = []

        self._contains_targets = scores is not None
        self._scores = list(scores) if self._contains_targets else None
        
        with tqdm(zip(toxic_comments, hypotheses), disable=not verbose) as progress_bar:
            for c, h in progress_bar:
                t = tokenizer(c, h, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
                self._encoded_pairs.append(t)

                
    def __len__(self):
        return len(self._scores)

    
    def __getitem__(self, index: int):
        if self._contains_targets:
            target_data = {'target': self._scores[index]}
        else:
            target_data = {}

        return {
            'input_ids': self._encoded_pairs[index]['input_ids'].squeeze(),
            'attention_mask': self._encoded_pairs[index]['attention_mask'].squeeze(),
            'token_type_ids': self._encoded_pairs[index]['token_type_ids'].squeeze(),
            **target_data
        }
