import torch
from torch import nn

class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.superParameter = torch.tensor(10.0)

    def forward(self, loss, cola_loss, epoch):
        return loss + cola_loss * self.superParameter
    """
    def forward(self, loss, cola_loss, epoch):
        if(cola_loss.item() > 0.1):
            return loss + cola_loss * self.superParameter * torch.tensor(pow(2, -epoch))
        else:
            return loss
    """
