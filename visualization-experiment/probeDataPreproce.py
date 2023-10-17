import pandas as pd
import math
import torch
import numpy as np
import h5py
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def findMax_len(sentences):
    """
    :param sentences:
    :return:
    """
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    max_len = 0
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)  # 66
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

    # For every sentence... 
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`         
        #   (6) Create attention masks for [PAD] tokens.             
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_Len,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])  # append[[101, 2256,...., 102, 0, 0,...,0]]  1*64

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])  # append[[1, 1, 1, ...1, 0, 0..., 0 ]]   1*64

    return input_ids, attention_masks

def train_dev_Dataset(layer_features):
    # Combine the training inputs into a TensorDataset.  
    dataset = TensorDataset(layer_features)
    #print(type(dataset))
    #print(dataset)

    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    return train_dataset, val_dataset

def train_val_DataLoader(train_dataset, val_dataset):
    batch_size = 1
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader

def dataPrepare(filepath):
    hf = h5py.File(filepath, 'r') 
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    for index in sorted([int(x) for x in indices]):
        layer_features = hf[str(index)]
        single_layer_features_list.append(layer_features)
    
    single_layer_features = np.array(single_layer_features_list)
    single_layer_features = torch.tensor(single_layer_features)
    print(single_layer_features.shape)


    train_dataset, dev_dataset = train_dev_Dataset(single_layer_features)
 
    train_dataloader, validation_dataloader = train_val_DataLoader(train_dataset, dev_dataset)

    return train_dataloader, validation_dataloader


def load_dataCSV():
    """

    :return:
    """
    df_cola_train = pd.read_csv("./sentences_public/cola_public/raw/in_domain_train.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    df_cola_dev = pd.read_csv("./sentences_public/cola_public/raw/in_domain_dev.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    df_cola_test = pd.read_csv("./sentences_public/cola_public/raw/out_of_domain_dev.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    
    cola_trainSe = df_cola_train.sentence.values
    cola_trainLa = df_cola_train.label.values
    cola_devSe = df_cola_dev.sentence.values
    cola_devLa = df_cola_dev.label.values
    cola_testSe = df_cola_test.sentence.values
    cola_testLa = df_cola_test.label.values

    sentences_cola = []
    
    for i in range(len(cola_trainSe)):
        #print(cola_trainSe[i], cola_trainLa[i])
        if(int(cola_trainLa[i]) == 1):
            sentences_cola.append(cola_trainSe[i])
    
            
    for i in range(len(cola_devSe)):
        #print(cola_trainSe[i], cola_trainLa[i])
        if(int(cola_devLa[i]) == 1):
            sentences_cola.append(cola_devSe[i])
            
    for i in range(len(cola_testSe)):
        #print(cola_trainSe[i], cola_trainLa[i])
        if(int(cola_testLa[i]) == 1):
            sentences_cola.append(cola_testSe[i])
    
    
    sentences_cola = np.array(sentences_cola)
    
    print(sentences_cola.shape)
    
    df_mrpc_train = pd.read_csv("./sentences_public/MRPC_public/msr_paraphrase_train.tsv", delimiter='\t', 
                     quoting=3, header=None, names=['Quailty', 'ID1', 'ID2', 'String1', 'String2'])
    
    df_mrpc_test = pd.read_csv("./sentences_public/MRPC_public/msr_paraphrase_test.tsv", delimiter='\t', 
                     quoting=3, header=None, names=['Quailty', 'ID1', 'ID2', 'String1', 'String2'])
    
    sentences_mrpc = np.hstack((df_mrpc_train.String1.values, df_mrpc_train.String2.values, 
                                df_mrpc_test.String1.values, df_mrpc_test.String2.values))
    
    print(sentences_mrpc.shape)
    
    
    df_rte_train = pd.read_csv("./sentences_public/RTE_public/train.tsv", delimiter='\t', 
                               quoting=3, header=None, names=['index', 'sentence1', 'sentence2', 'label'])

    df_rte_dev = pd.read_csv("./sentences_public/RTE_public/dev.tsv", delimiter='\t', 
                              quoting=3, header=None, names=['index', 'sentence1', 'sentence2', 'label'])
    
    df_rte_test = pd.read_csv("./sentences_public/RTE_public/test.tsv", delimiter='\t', 
                           quoting=3, header=None, names=['index', 'sentence1', 'sentence2'])
    
    """
    
    rte_trainSe1 = df_rte_train.sentence1.values
    rte_trainSe2 = df_rte_train.sentence2.values
    rte_testSe1 = df_rte_test.sentence1.values
    rte_testSe2 = df_rte_test.sentence2.values
    
    print(rte_trainSe1[1], rte_trainSe2[1])
    print(rte_testSe1[1], rte_testSe2[1])
    
    #print(df_rte_train[1].sentence1.values, df_rte_train[1].sentence2.values)
    #print(df_rte_dev[1].sentence1.values, df_rte_dev[1].sentence2.values)
    #print(df_rte_test[1].sentence1.values, df_rte_test[1].sentence2.values)
    """
    
    sentences_rte = np.hstack((df_rte_train[1:].sentence1.values, df_rte_train[1:].sentence2.values, 
                               df_rte_dev[1:].sentence1.values, df_rte_dev[1:].sentence2.values,
                               df_rte_test[1:].sentence1.values, df_rte_test[1:].sentence2.values))
    
    print(sentences_rte.shape)
    
    df_stsb_train = pd.read_csv("./sentences_public/STS-B_public/sts-train.tsv", delimiter='\t', 
                               quoting=3, header=None, names=['genre', 'filename', 'year', 'old_index', 'score', 'sentence1', 'sentence2'])

    df_stsb_dev = pd.read_csv("./sentences_public/STS-B_public/sts-dev.tsv", delimiter='\t', 
                              quoting=3, header=None, names=['genre', 'filename', 'year', 'old_index', 'score', 'sentence1', 'sentence2'])
    
    df_stsb_test = pd.read_csv("./sentences_public/STS-B_public/sts-test.tsv", delimiter='\t', 
                           quoting=3, header=None, names=['genre', 'filename', 'year', 'old_index', 'score', 'sentence1', 'sentence2'])
    
    sentences_stsb = np.hstack((df_stsb_train.sentence1.values, df_stsb_train.sentence2.values, 
                                df_stsb_dev.sentence1.values, df_stsb_dev.sentence2.values,
                                df_stsb_test.sentence1.values, df_stsb_test.sentence2.values))
    
    print(sentences_stsb.shape)
    
    
    
    sentences = np.hstack((sentences_cola, sentences_mrpc, sentences_rte, sentences_stsb))
    
    print(sentences.shape)
    
    return sentences


if __name__ == '__main__':
    print('Test begin')

    _ = load_dataCSV()
    
    max_len = findMax_len(_)

    print('Test Finish')
