import torch.nn as nn

import pickle
import pandas as pd
import torch
import numpy as np

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=128, **kwargs):

        args = kwargs['args']

        rows = pd.read_csv('./lookUpTable.csv', sep='\t')
        
        data_list = []

        with open('./Data/preprocessed/yna_min_uc{}-min_sc{}_{}/dataset.pkl'.format(args.min_uc, args.min_sc, args.dataset_datetime), 'rb') as f:
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
        
        zero = torch.normal(0, 1, size=(1,128))
        mask = torch.normal(0, 1, size=(1,128))
        weight = torch.from_numpy(np.array(floatTable)).float()
        weight = torch.cat((zero, weight, mask), 0).to(torch.device(args.device))

        super().__init__(vocab_size, embed_size, padding_idx=0, _weight=weight)