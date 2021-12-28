import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import os


class Reviews(Dataset):
    def __init__(self, tokenizer, split):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.split = split
        self.data = None
        self.addData()

    def addData(self):
        self.data = []
        if self.split == 'train':
            file_name_list = os.listdir('../train_data_all')
            for file_name in file_name_list:
                file_path = os.path.join('../train_data_all', file_name)
                df = pd.read_csv(file_path)
                for idx, row in df.iterrows():
                    comments_content = row['CommentsContent']
                    comments_stars = row['CommentsStars']
                    self.data.append({'txt': comments_content,
                                      'cls': comments_stars})

    def __getitem__(self, index):
        txt, cls = self.data[index]
        token = self.tokenizer.encode_plus(text=txt,
                                           add_special_tokens=False,
                                           padding='max_length',
                                           max_length=64,
                                           truncation=True,
                                           return_attention_mask=True)
        token_ids = torch.tensor(token['input_ids'])
        attn_mask = torch.tensor(token['attention_mask'])

        sample = {'input_tokens': token_ids,
                  'input_attn_mask': attn_mask,
                  'cls': cls}

        return sample
