import torch.nn as nn

import pickle
import pandas as pd
import torch
import numpy as np

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        # super().__init__(vocab_size, embed_size, padding_idx=0)

        rows = pd.read_csv('./table.csv', sep='\t')
        
        data_list = []

        with open('./Data/preprocessed/yna_min_uc3-min_sc0/dataset.pkl', 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                except EOFError:
                    break
                data_list.append(data)

        df = pd.Series(data_list[0])
        keys = df.smap.keys()

        table = pd.DataFrame()

        for k in keys:
            table = pd.concat([table, rows[rows['content_id'] == k]])
        table = table['vector'].reset_index(drop=True)
        table = table.str.split(',').str

        floatTable = pd.DataFrame()

        for i in range(128):
            floatTable = pd.concat([floatTable, table[i].astype(float)], axis=1)
        
        zero = torch.normal(0, 1, size=(1,128)).to(torch.device('cuda'))

        mask = torch.normal(0, 1, size=(1,128)).to(torch.device('cuda'))

        weight = torch.from_numpy(np.array(floatTable)).float().to(torch.device('cuda'))

        weight = torch.cat((zero, weight, mask), 0)

        super().__init__(vocab_size, embed_size, padding_idx=0, _weight=weight)

        self.weight.requires_grad=True
