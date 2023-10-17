import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import math

from probe import OneWordPSDProbe
from constructLabel import ConstructLabelGaget, EsspLabelGaget

class probe(nn.Module):
    def __init__(self, args):
        super(probe, self).__init__()
        self.oneWordpsdProbe = OneWordPSDProbe(args={'probe': {'maximum_rank': args['probe']['maximum_rank']}, 'model': {'hidden_dim': args['model']['hidden_dim']}})
        self.constructLabel = ConstructLabelGaget(args=None)
    
    def forward(self, batch):
        norms = self.oneWordpsdProbe(batch).to("cuda:0" if torch.cuda.is_available() else "cpu")
        pseu_labels = self.constructLabel(norms).to("cuda:0" if torch.cuda.is_available() else "cpu")
        
        prob_loss = None
        Prob_loss_fct = MSELoss()
        prob_loss = Prob_loss_fct(norms, pseu_labels)
        
        return prob_loss

class essp(nn.Module):
    def __init__(self, args):
        super(essp, self).__init__()
        self.oneWordpsdProbe = OneWordPSDProbe(args={'probe': {'maximum_rank': args['probe']['maximum_rank']}, 'model': {'hidden_dim': args['model']['hidden_dim']}})
        self.constructLabel = EsspLabelGaget(args=None)
    
    def forward(self, batch):
        norms = self.oneWordpsdProbe(batch).to("cuda:0" if torch.cuda.is_available() else "cpu")
        pseu_labels = self.constructLabel(norms).to("cuda:0" if torch.cuda.is_available() else "cpu")
        prob_loss = None
        Prob_loss_fct = MSELoss()
        prob_loss = Prob_loss_fct(norms, pseu_labels)
        
        return prob_loss
    
class sp(nn.Module):
    def __init__(self, args):
        super(sp, self).__init__()
        self.oneWordpsdProbe = OneWordPSDProbe(args={'probe': {'maximum_rank': args['probe']['maximum_rank']}, 'model': {'hidden_dim': args['model']['hidden_dim']}})
    
    def forward(self, batch, labels):
        norms = self.oneWordpsdProbe(batch).to("cuda:0" if torch.cuda.is_available() else "cpu")
        prob_loss = None
        Prob_loss_fct = MSELoss()
        prob_loss = Prob_loss_fct(norms, labels)
        
        return prob_loss
