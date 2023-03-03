# text-detoxification

This is the 1st place solution code for [the educational competition on text detoxification](https://codalab.lisn.upsaclay.fr/competitions/4768?secret_key=a714c571-6544-48fd-9bc9-661c229d7204) at [Ozon Masters NLP Course](https://ozonmasters.ru/nlp_pub).

This competition is inspired by the original one, [RUSSE 2022 Russian Text Detoxification Based on Parallel Corpora](https://codalab.lisn.upsaclay.fr/competitions/642), but differs from it by test data and the way of evaluation. Namely, submissions here are evaluated using [`nltk.translate.bleu_score.corpus_bleu`](https://www.nltk.org/_modules/nltk/translate/bleu_score.html) and the development dataset of the original competition serves as the test dataset in this competition.

# Task Formulation

Detoxification is a kind of text style transfer, which aims to paraphrase some text in the toxic style to the non-toxic one, preserving the meaning of the original content and maintaining natural language fluency.

# Solution Strategy

The solution consists of the following stages:
- Candidate Generation
- Scoring and Candidate Selection

## Candidate Generation

In this stage, we train [`sberbank-ai/ruT5-base`](https://huggingface.co/sberbank-ai/ruT5-base) and [`cointegrated/rut5-base-paraphraser`](https://huggingface.co/cointegrated/rut5-base-paraphraser) on the entire training dataset, periodically saving their snapshots. Now, for a given text fragment, each snapshot generates its candidates for both the training data and the test data.

The best model here reached 0.805.

## Scoring and Candidate Selection

Next, for a given pair of a toxic comment and its non-toxic candidate, we set a goal to predict the BLEU score between this candidate and the ground truth non-toxic comment. For this purpose, we label such pairs for the training dataset and train [`cointegrated/rubert-tiny-toxicity`](https://huggingface.co/cointegrated/rubert-tiny-toxicity) on these labeled pairs.

Finally, to produce a submission file, this model scores candidates for the test data. For each toxic comment in the test dataset, the ranks obtained are exploited to determine the best candidate.

This approach increased the BLEU score to 0.814.