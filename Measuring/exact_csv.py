import time
import datetime
import random
import numpy as np
import pandas as pd


def exact_csv(file_name):
   
    data = list(pd.read_csv(file_name)['epoch\t1'])
    
    states = ['epoch\t1'.split('\t')]
    
    for i in range(len(data)):
        states.append(data[i].split('\t'))
    
    split_states = []
    
    for i in range(int(len(states)/5)):
        memo_state = []
        for j in range(5):
            memo_state.append(states[i*5+j])
        split_states.append(memo_state)
    
    return split_states

def exact_manycsv(name, variables):
    many_data = []
    name_prefix = ""
    name_suffix1 = "_l"
    name_suffix2 = ".csv"
    for i in range(len(variables)):
        file_name = name_prefix + name + name_suffix1 + str(variables[i]) + name_suffix2
        data = exact_csv(file_name)
        many_data.append(data)
    
    return many_data

def exact_split(name, variables):
    data = exact_manycsv(name, variables)
    split_data = []
    for l in range(len(data)):
        memo_state = []
        for i in range(len(data[l])):
            if(data[l][i][0][1]=='10'): 
                memo_state.append(data[l][i][2][1])
        
        split_data.append(memo_state)
        
    return split_data
    
  
        

if __name__ == '__main__':
    name = "essp"
    variables = [i for i in range(0, 25)]
    data = exact_split(name, variables)
    
    print(data)
    