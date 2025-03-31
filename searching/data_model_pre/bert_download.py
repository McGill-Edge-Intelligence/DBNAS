from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

tokenizer.save_pretrained('./bert_masked')
model.save_pretrained('./bert_masked')

# print(model)
