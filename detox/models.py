from torch import nn
from transformers import AutoModelForSequenceClassification


class SkorBERT(nn.Module):
    def __init__(self, model_name: str):
        super(SkorBERT, self).__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._linear = nn.Linear(5, 1)
        self._activation = nn.Sigmoid()


    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self._model.forward(input_ids, attention_mask, token_type_ids)
        x = self._linear(x.logits)
        return self._activation(x)