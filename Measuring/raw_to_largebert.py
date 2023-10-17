# -*- coding: utf-8 -*-

import pandas as pd
import math
import torch
import h5py
import numpy as np
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.utils.checkpoint
import random
from transformers import BertPreTrainedModel, BertModel

from probeDataPreproce import findMax_len, input_ids_attention_masks, load_dataCSV

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert = BertModel.from_pretrained("bert-large-uncased")
bert.to(device)
bert.eval()

test_sentences = load_dataCSV()

maxLen = findMax_len(test_sentences)

test_input_ids, test_attention_masks = input_ids_attention_masks(test_sentences, maxLen)
test_input_ids = torch.cat(test_input_ids, dim=0)  # torch.Size([8551, 64])
test_attention_masks = torch.cat(test_attention_masks, dim=0)  # torch.Size([8551, 64])

test_dataset = TensorDataset(test_input_ids, test_attention_masks)

print('{:>5,} samples'.format(len(test_dataset)))

batch_size = 32
test_dataloader = DataLoader(
    test_dataset,  # The training samples.
    sampler=RandomSampler(test_dataset),  # Select batches randomly
    batch_size=batch_size  # Trains with this batch size.
)

layerIndex = 24

with h5py.File("probe_trainData{:}".format(layerIndex), 'w') as fout:
    for step, batch in enumerate(test_dataloader):
        #if(step>=2):
        #   break
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        
        with torch.no_grad():
            outputs = bert(
                b_input_ids,
                attention_mask=b_input_mask,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
            )
            
            output_hidden_states = outputs.hidden_states[layerIndex]
        #print(output_hidden_states.shape)   
        dset = fout.create_dataset(str(step), output_hidden_states.shape)
        dset[:, :, :] = np.vstack([np.array(output_hidden_states.cpu())])

print("Finish!")
