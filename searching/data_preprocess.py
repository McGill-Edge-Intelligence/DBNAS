import pandas as pd
from hydra.experimental import compose, initialize
from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.legacy import data
from torchtext.data import get_tokenizer
import numpy as np
from sklearn import metrics
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

initialize(config_path="config")
cfg = compose(config_name="config")

seq_len= cfg.data.seq_len

config = BertConfig.from_pretrained('textattack/bert-base-uncased-SST-2', output_hidden_states=True)
bert_model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2', config=config)

'''
load data into a dataframe
'''
def read_data(data_path):
    return pd.read_csv(data_path,sep="\t")

'''
return the vocabulary dictionary using the train dataset
'''
def get_vocab(train_df):
    text_field = data.Field(sequential=True, tokenize='basic_english', lower=True)
    preprocessed_text = train_df['sentence'].apply(lambda x: text_field.preprocess(x))
    text_field.build_vocab(preprocessed_text,vectors="glove.6B.100d")
    return text_field.vocab

'''
convert a sentence to a list of index that has shape seq_len
'''
def sentence2idx(vocab, sentence):
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(sentence)
    indices = [0] * seq_len
    for idx, token in enumerate(tokens):
        indices[idx] = vocab[token]
    return torch.LongTensor(indices)


'''
get the accuracy of predictions
'''
def getSSTAccuracy(prediction, label):
    prediction = torch.sigmoid(prediction).cpu().detach().numpy() >= 0.5
    label = label.cpu().detach().numpy()
    return metrics.accuracy_score(label, prediction)


class SSTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, max_token_len: int = seq_len, teacher_tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')):
        super().__init__()
        self.data = data
        self.max_token_len = max_token_len
        self.teacher_tokenizer = teacher_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        sentence = data_row["sentence"]
        label = data_row["label"]
        label = np.atleast_1d(label)
 
        encoding = self.teacher_tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'   
        )
        
        #return the data as a dict
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            token_type_ids=encoding["token_type_ids"].flatten(),
            labels=torch.from_numpy(label).type(torch.FloatTensor)
        )
        

def get_dataloader(data_type, data_path, batch_size):
    data_df = read_data(data_path)
    data_dataset = SSTDataset(data_df)
    if data_type == 'train':
        return DataLoader(data_dataset, batch_size, shuffle=True, num_workers=2)
    else:
        return DataLoader(data_dataset, batch_size, shuffle=False, num_workers=2)
