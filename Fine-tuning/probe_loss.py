import torch
from torch import nn, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
    
class buff_Loss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.superParameter1 = torch.tensor(10.0)
        self.superParameter2 = torch.tensor(5.0)
    
    def forward(self, norms, pseu_labels):
        batchlen, seqlen = norms.size()
        for i in range(0, batchlen):
            for j in range(0, seqlen):
                if(pseu_labels[i][j] == 1):
                    norms[i][j] = self.superParameter1 * norms[i][j]
                    pseu_labels[i][j] = self.superParameter1 * pseu_labels[i][j]
                if(pseu_labels[i][j] == 2):
                    norms[i][j] = self.superParameter2 * norms[i][j]
                    pseu_labels[i][j] = self.superParameter1 * pseu_labels[i][j]
        
        prob_loss = None
        Prob_loss_fct = MSELoss()
        prob_loss = Prob_loss_fct(norms, pseu_labels)
        
        return prob_loss
