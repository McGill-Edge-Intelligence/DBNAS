import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

'''
Pretrained teacher model: BERT
Load teacher model from huggingface library
'''

class BERT(nn.Module):
    """
    BERT teacher using BertForSequenceClassification for GLUE tasks.
    """
    def __init__(self):
        super().__init__()
        self.config = BertConfig.from_pretrained('textattack/bert-base-uncased-SST-2', output_hidden_states=True)
        self.bert_model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2', config=self.config)
        self.bert_model.requires_grad_(False)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        return self.bert_model(input_ids, attention_mask, token_type_ids)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    model = BERT()
    # print(model)
    # outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, token_type_ids=inputs.token_type_ids)
    outputs = model(**inputs)
    print(len(outputs))
    logits = outputs.logits
    print(logits)