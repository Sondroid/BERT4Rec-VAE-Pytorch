from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
from matplotlib import pyplot as plt

class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        
        if sys._getframe(1).f_code.co_name == 'test':
            seqs, answers = batch
            scores = self.model(seqs)
            scores = scores[:, -1, :]
            labels = F.one_hot(answers.reshape(-1), num_classes = self.args.num_items + 1)
            
            # v = torch.argsort(scores, dim = 1, descending=True)[:,:5]
            # print(v[:20,:])
            # plt.hist(v.reshape(-1).cpu().numpy(), bins=1351)
            # plt.xlabel('news id', fontsize = 14)
            # plt.show()
        
        else:
            seqs, candidates, labels = batch
            scores = self.model(seqs)  # B x T x V
            scores = scores[:, -1, :]  # B x V
            # scores = scores.gather(1, candidates)  # B x C
            labels = candidates[:,0]
            labels = F.one_hot(labels.reshape(-1), num_classes = self.args.num_items + 1)
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        
        return metrics
