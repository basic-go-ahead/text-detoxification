import os
from typing import Iterable, Union

import emoji
import pandas as pd

from .scoring import BLEUScorer


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    remove_emoji = lambda s: emoji.replace_emoji(s, replace='')
    
    for c in ['toxic_comment', 'neutral_comment1', 'neutral_comment2', 'neutral_comment3']:
        if c in df.columns:
            df[c] = df[c].map(remove_emoji)

    return df


def expand(df: pd.DataFrame) -> pd.DataFrame:
    having2 = df.loc[df['neutral_comment2'] != '', ['toxic_comment', 'neutral_comment2']].rename(columns={'neutral_comment2': 'neutral_comment1'})
    having3 = df.loc[df['neutral_comment3'] != '', ['toxic_comment', 'neutral_comment3']].rename(columns={'neutral_comment3': 'neutral_comment1'})
    
    return pd.concat([df[['toxic_comment', 'neutral_comment1']], having2, having3]).drop_duplicates()


def _collect_answers(container: list,
    path_template: str,
    answer_ids: Iterable[int],
    scorer: Union[BLEUScorer, None]=None
):
    for e in answer_ids:
        df = pd.read_csv(path_template.format(e), sep='\t')
        df['group'] = df.index
        if scorer is not None:
            df['score'] = scorer.get_scores(df.no_toxic)
            df.drop(df[df['score'] == 1.].index, inplace=True)
        container.append(df)


def collect_answers(container: list,
    folder: str,
    scorer: Union[BLEUScorer, None]=None
):
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path, sep='\t')
            df['group'] = df.index
            if scorer is not None:
                df['score'] = scorer.get_scores(df.no_toxic)
                df.drop(df[df['score'] == 1.].index, inplace=True)
            container.append(df)
