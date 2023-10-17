import torch
import time
import datetime
import random
import numpy as np

import torch.utils.checkpoint
from transformers import BertPreTrainedModel, BertModel

from draw import draw_probeTree
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from utils import format_time


if __name__ == '__main__':
    
    model = torch.load('psdProbe_save4.pt')
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        print(model.oneWordpsdProbe.proj[0])
        
        with open("proj7_4.csv", "w") as f:
            for i in range(len(model.oneWordpsdProbe.proj)):
                for j in range(len(model.oneWordpsdProbe.proj[i])):
                    f.write("%.4f " % (model.oneWordpsdProbe.proj[i][j]))
                f.write("\n")
    
    torch.save(model.to("cpu"), 'psdProbe_save4cpu.pt')
    
    print('Finish All')
