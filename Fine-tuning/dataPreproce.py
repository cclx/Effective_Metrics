# %load dataPreproce.py
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def loadCSV():
    """

    :return:
    """
    df = pd.read_csv("./cola_public/raw/in_domain_train.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values  # 8551 numpy.ndarray
    labels = df.label.values  # 8551 numpy.ndarray
    return sentences, labels

def loadCSV2():
    """
    :return:
    """
    df = pd.read_csv("./cola_public/raw/in_domain_dev.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values  # 8551 numpy.ndarray
    labels = df.label.values  # 8551 numpy.ndarray
    df2 = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv",
                     delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    out_sentences = df2.sentence.values  # 8551 numpy.ndarray
    out_labels = df2.label.values
    return sentences, labels, out_sentences, out_labels


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

    print('Max sentence length: ', max_len)  # 47
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


def train_val_Dataset(input_ids, attention_masks, labels):
    # Combine the training inputs into a TensorDataset.  
    dataset = TensorDataset(input_ids, attention_masks, labels)
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

def test_Dataset(input_ids, attention_masks, labels):
    # Combine the training inputs into a TensorDataset.  
    dataset = TensorDataset(input_ids, attention_masks, labels)
    #print(type(dataset))
    #print(dataset)
    
    return dataset


def train_val_DataLoader(train_dataset, val_dataset):

    batch_size = 32
 
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

def test_DataLoader(test_dataset):
  
    batch_size = 32
  
    test_dataloader = DataLoader(
        test_dataset,  # The training samples.
        sampler=RandomSampler(test_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    return test_dataloader

def True_dataPrepare():
    # 从 CSV 文件读出 sentences 和 labels
    train_sentences, train_labels = loadCSV()
    dev_sentences, dev_labels, out_dev_sentences, out_dev_labels = loadCSV2()
    print(len(train_sentences), len(train_labels))
    """

    pos_train_sentences, pos_train_labels, neg_train_sentences, neg_train_labels = [], [], [], []
    neg_dev_sentences, neg_dev_labels, pos_dev_sentences, pos_dev_labels = [], [], [], []
    for i in range(len(train_labels)):
        #print(cola_trainSe[i], cola_trainLa[i])
        if(int(train_labels[i]) == 1):
            pos_train_sentences.append(train_sentences[i])
            pos_train_labels.append(train_labels[i])
    
    for i in range(len(dev_labels)):
        #print(cola_trainSe[i], cola_trainLa[i])
        if(int(dev_labels[i]) == 1):
            pos_dev_sentences.append(dev_sentences[i])
            pos_dev_labels.append(dev_labels[i])
            
    train_sentences, train_labels = [], []
    dev_sentences, dev_labels = [], []
    
    train_sentences, train_labels = pos_train_sentences, pos_train_labels
    dev_sentences, dev_labels = pos_dev_sentences, pos_dev_labels
    print(len(train_sentences), len(train_labels))
    """
   
   
    maxLen = findMax_len(train_sentences)
    maxLen = max(maxLen, findMax_len(dev_sentences))
    #maxLen = max(maxLen, findMax_len(out_dev_sentences))
    #
    train_input_ids, train_attention_masks = input_ids_attention_masks(train_sentences, maxLen)
    dev_input_ids, dev_attention_masks = input_ids_attention_masks(dev_sentences, maxLen)
    out_dev_input_ids, out_dev_attention_masks = input_ids_attention_masks(out_dev_sentences, maxLen)
  
    train_input_ids = torch.cat(train_input_ids, dim=0)  # torch.Size([8551, 64])
    train_attention_masks = torch.cat(train_attention_masks, dim=0)  # torch.Size([8551, 64])
    dev_input_ids = torch.cat(dev_input_ids, dim=0)  # torch.Size([8551, 64])
    dev_attention_masks = torch.cat(dev_attention_masks, dim=0)  # torch.Size([8551, 64])
    train_labels = torch.tensor(train_labels)  # torch.Size([8551])
    dev_labels = torch.tensor(dev_labels)  # torch.Size([8551])
    
    out_dev_input_ids = torch.cat(out_dev_input_ids, dim=0)  # torch.Size([8551, 64])
    out_dev_attention_masks = torch.cat(out_dev_attention_masks, dim=0)  # torch.Size([8551, 64])
    train_labels = torch.tensor(train_labels)  # torch.Size([8551])
    dev_labels = torch.tensor(dev_labels)
    out_dev_labels = torch.tensor(out_dev_labels)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
    out_dev_dataset = TensorDataset(out_dev_input_ids, out_dev_attention_masks, out_dev_labels) 
   
    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(dev_dataset)))
    print('{:>5,} validation samples'.format(len(out_dev_dataset)))

    # 制作 DataLoader
    train_dataloader, validation_dataloader = train_val_DataLoader(train_dataset, dev_dataset)

    return train_dataloader, validation_dataloader


def dataPrepare():

    sentences, labels = loadCSV()
   
    maxLen = findMax_len(sentences)
    #
    input_ids, attention_masks = input_ids_attention_masks(sentences, maxLen)

    input_ids = torch.cat(input_ids, dim=0)  # torch.Size([8551, 64])
    attention_masks = torch.cat(attention_masks, dim=0)  # torch.Size([8551, 64])
    labels = torch.tensor(labels)  # torch.Size([8551])

    train_dataset, val_dataset = train_val_Dataset(input_ids, attention_masks, labels)

    train_dataloader, validation_dataloader = train_val_DataLoader(train_dataset, val_dataset)

    return train_dataloader, validation_dataloader

def dataPrepare2():
  
    sentences, labels = loadCSV2()

    maxLen = findMax_len(sentences)
    #
    input_ids, attention_masks = input_ids_attention_masks(sentences, maxLen)

    input_ids = torch.cat(input_ids, dim=0)  # torch.Size([8551, 64])
    attention_masks = torch.cat(attention_masks, dim=0)  # torch.Size([8551, 64])
    labels = torch.tensor(labels)  # torch.Size([8551])

    test_dataset = test_Dataset(input_ids, attention_masks, labels)

    test_dataloader = test_DataLoader(test_dataset)

    return test_dataloader


if __name__ == '__main__':
    print('Test begin')

    trainLoader, valLoader = dataPrepare()

    print('Test Finish')

