# %load dataPreproce.py
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from draw import draw_Evaluated_probe

def loadCSV():
    """

    :return:
    """
    df = pd.read_csv("./freeze_prob_cola-seed1~40.csv",
                     delimiter='\t', header=None, names=['key', 'state'])
    keys = df.key.values  
    states = df.state.values  
    return keys, states

def producere():
    keys, states = loadCSV()
    loss_totalstats = [[] for i in range(0, 10)]
    for i in range(0, 25):
        for j in range(0, 10):
            for g in range(2, 3):
                for k in range(0, 10):
                    if(keys[i*300+j*30+g*10+k]=='Valid. Matthews.'):
                        loss_totalstats[j].append(float(states[i*300+j*30+g*10+k]))
                        
    for i in range(len(loss_totalstats)):
        draw_Evaluated_probe(loss_totalstats[i], 'seed{:}'.format(i*4+1))


if __name__ == '__main__':
    print('Test begin')

    producere()

    print('Test Finish')

