import os
from collections import namedtuple, defaultdict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py
import pandas as pd

from transformers import BertTokenizer

from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

def findMax_len(sentences):
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)
    return max_len

def input_ids_attention_masks(sentences, max_Len):
    """

    :param sentences:
    :return:
    """
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_Len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids']) 
        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks

def loadCSV(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file]
    df = pd.DataFrame(lines, columns=['sentence'])
    sentences = df.sentence.values
    return sentences;


def load_conll_dataset(filepath):
    lines = (x for x in open(filepath))
    tree = []
    flag = 0
    conllx_lines = []
    for line in lines:
        if line.startswith('#'):
            continue
        if line.strip():
            conllx_lines.append(line.strip().split('\t'))
            flag = 1
        else:
            if(flag):
                tree.append(conllx_lines)
            conllx_lines = []
            flag = 0
    
    return tree;
    
def get_ordering_index(conllx_heads, i, head_indices=None):
    if conllx_heads:
        head_indices = []
        number_of_underscores = 0
        for elt in conllx_heads:
            if elt == '_':
                head_indices.append(0)
                number_of_underscores += 1
            else:
                head_indices.append(int(elt) + number_of_underscores)
    length = 0
    i_head = i+1
    while True:
        i_head = head_indices[i_head - 1]
        if i_head != 0:
            length += 1
        else:
            return length+1
    
def labels(tree, sent):
    sent = sent.split(" ")
    sentence_length = len(sent)
    depths = torch.zeros(sentence_length)
    conllx_heads = [tree[i][6] for i in range(sentence_length)]
    
    for i in range(sentence_length):
        depths[i] = get_ordering_index(conllx_heads, i)
    return depths

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0])
    else:
        return []

def constructData(mapping, label, max_len):
    true_labels = []
    for i in range(len(label)):
        true_label = []
        true_label.append(torch.tensor(2.0))
        length = len(mapping[i])
        if(len(label[i])-1!=mapping[i][length-1]):
            continue
        for j in range(length):
            true_label.append(torch.tensor(label[i][mapping[i][j]]*1.0))
        for k in range(max_len-length-1):
            true_label.append(torch.tensor(2.0))
        true_labels.append(torch.tensor(true_label))
        
    return true_labels

def homogeneousData(data, label):
    #input_ids, attention_masks = input_ids_attention_masks(data, max_len)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    
    mapping = []
    stoken = []
    
    for i in range(len(data)):
        sentence = data[i]
        tokens = tokenizer.tokenize(sentence)
        stoken.append(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        index = -1
        sent_map = []
        for j in range(len(tokens)):
            token = tokens[j]
            if(not token.startswith('##')):
                index+=1
            sent_map.append(index)
        
        mapping.append(sent_map)
    
    true_data = []
    for i in range(len(label)):
        length = len(mapping[i])
        if(len(label[i])-1!=mapping[i][length-1]):
            continue
        true_data.append(data[i])
    
    max_len = findMax_len(true_data)
    
    input_ids, attention_masks = input_ids_attention_masks(true_data, max_len)
    
    true_labels = constructData(mapping, label, max_len)
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    shape0 = attention_masks.shape[0]
    shape1 = attention_masks.shape[1]
    true_labels = torch.cat(true_labels, dim=0).view(shape0, shape1)

    return input_ids, attention_masks, true_labels;
    

def loader_trueData():
    sents_test = loadCSV("./data/en_ewt-ud-test.csv")
    tree_test = load_conll_dataset("./data/en_ewt-ud-test.conllu")
    dep_test = []
    for i in range(len(sents_test)):
        dep_test.append(labels(tree_test[i], sents_test[i]))

    sents_train = loadCSV("./data/en_ewt-ud-train.csv")
    tree_train = load_conll_dataset("./data/en_ewt-ud-train.conllu")
    dep_train = []
    

    for i in range(len(sents_train)):
        dep_train.append(labels(tree_train[i], sents_train[i]))

    sents_dev = loadCSV("./data/en_ewt-ud-dev.csv")
    tree_dev = load_conll_dataset("./data/en_ewt-ud-dev.conllu")
    dep_dev = []

    for i in range(len(sents_dev)):
        dep_dev.append(labels(tree_dev[i], sents_dev[i]))

    train_data = np.concatenate((sents_train, sents_dev))
    train_label = np.concatenate((dep_train, dep_dev))
    
    train_ids, train_masks, true_trainLabel = homogeneousData(train_data, train_label)
    
    print(train_ids.size, train_masks.size, true_trainLabel.size)
    
    test_data = sents_test
    test_label = dep_test
    
    test_ids, test_masks, true_testLabel = homogeneousData(test_data, test_label)
    
    
    train_dataset = TensorDataset(train_ids, train_masks, true_trainLabel)
    test_dataset = TensorDataset(test_ids, test_masks, true_testLabel)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    
    return train_dataloader, test_dataloader

loader_trueData()

