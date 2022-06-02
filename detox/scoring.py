from typing import Dict, Iterable

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse


class BLEUScorer:
    def __init__(self, df: pd.DataFrame):
        self._references = []
        self._uncased_references = []
        self._df = df
        
        for e in df.itertuples():
            e_references = self._collect_references(e, False)
            self._references.append(e_references)
            e_references = self._collect_references(e, True)
            self._uncased_references.append(e_references)
            
            
    def _collect_references(self, e, uncased: bool):
        action = lambda s: s.lower() if uncased else s
        e_references = [action(e.neutral_comment1).split()]
        if e.neutral_comment2:
            e_references.append(action(e.neutral_comment2).split())
        if e.neutral_comment3:
            e_references.append(action(e.neutral_comment3).split())
        return e_references
            
            
    def get_scores(self, hypothesis: Iterable[str], uncased: bool=False) -> Iterable[float]:
        scores = []
        references = self._uncased_references if uncased else self._references

        smoother = SmoothingFunction()
        
        for its_references, h in zip(references, hypothesis):
            v = sentence_bleu(
                its_references,
                (h.lower() if uncased else h).split(),
                smoothing_function=smoother.method1
            )
            scores.append(v)
            
        return scores


def calc_metrics(df: pd.DataFrame, true_scores_column: str, pred_scores_column: str) -> Dict[str, float]:   
    return {
        'mse': mse(df[true_scores_column], df[pred_scores_column]),
        'pearson': pearsonr(df[true_scores_column], df[pred_scores_column])
    }
