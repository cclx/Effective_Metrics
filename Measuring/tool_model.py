import torch.nn as nn
import torch

def Compute_norms(batch):
    batchlen, seqlen, rank = batch.size()
    norms = torch.bmm(batch.view(batchlen* seqlen, 1, rank),
        batch.view(batchlen* seqlen, rank, 1))
    norms = norms.view(batchlen, seqlen)
    return norms
    
