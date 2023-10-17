# %load dataPreproce.py
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from draw import draw_Evaluated_probe, draw_Evaluated_normaltrain,  draw_Evaluated_overmeasure

def loadCSV(file_name):
    """

    :return:
    """
    df = pd.read_csv(file_name, delimiter='\t', header=None, names=['key', 'state'])
    keys = df.key.values  
    states = df.state.values  
    return keys, states

def producere(file_name, out_name):
    keys, states = loadCSV(file_name)
    loss_totalstats = [[] for i in range(0, 10)]
    for i in range(0, 25):
        for j in range(0, 10):
            for g in range(2, 3):
                for k in range(0, 10):
                    if(keys[i*300+j*30+g*10+k]=='Valid. Matthews.'):
                        loss_totalstats[j].append(float(states[i*300+j*30+g*10+k]))
                        
    for i in range(len(loss_totalstats)):
        draw_Evaluated_probe(loss_totalstats[i], out_name+'-'+'seed{:}'.format(i*4+1))

def noprobe_producere(file_name):
    keys, states = loadCSV(file_name)
    loss_stats = []
    for i in range(0, 10):
        for j in range(2, 3):
            for k in range(0, 7):
                if(keys[i*21+j*7+k]=='Valid. Matthews.'):
                    loss_stats.append(float(states[i*21+j*7+k]))
    
    draw_Evaluated_normaltrain(loss_stats)


def over_measure():
    prob_keys, prob_states = loadCSV("prob_cola-seed1~40.csv")
    normal_keys, normal_states = loadCSV("./cola-seed1~40.csv")
    loss_totalstats = [[] for i in range(0, 25)]
    for i in range(0, 25):
        for j in range(0, 10):
            for g in range(2, 3):
                for k in range(0, 10):
                    if(prob_keys[i*300+j*30+g*10+k]=='Valid. Matthews.'):
                        loss_totalstats[i].append(float(prob_states[i*300+j*30+g*10+k]))
    maxm = []
    minm = []
    avgm = []
    stdm = []
    for i in range(0, 25):
        stdm.append(np.std(loss_totalstats[i]))
        avgm.append(np.mean(loss_totalstats[i]))
        maxm.append(np.max(loss_totalstats[i]))
        minm.append(np.min(loss_totalstats[i]))
    
    print("avgm===============:")
    print(avgm)
    print("stdm===============:")
    print(stdm)
    print("maxm===============:")
    print(maxm)
    print("minm===============:")
    print(minm)
                       
    loss_stats = []
    for i in range(0, 10):
        for j in range(2, 3):
            for k in range(0, 7):
                if(normal_keys[i*21+j*7+k]=='Valid. Matthews.'):
                    loss_stats.append(float(normal_states[i*21+j*7+k]))
    
    print("avgm===============:")
    print(np.mean(loss_stats))
    print("stdm===============:")
    print(np.std(loss_stats))
    print("maxm===============:")
    print(np.max(loss_stats))
    print("minm===============:")
    print(np.min(loss_stats))
    """
    for i in range(0, 25, 5):
        name = "freeze-multi-layer-measure_print-l{tom}~{jerry}".format(tom=i, jerry=i+4)
        draw_Evaluated_overmeasure(loss_totalstats, loss_stats, i, name)
    """



if __name__ == '__main__':
    print('Test begin')
    
    #file_name = "./cola-seed1~40.csv"
    #out_name = "unfreeze"
    over_measure()
    #noprobe_producere(file_name)

    print('Test Finish')

