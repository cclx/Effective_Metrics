import pandas as pd
import math
import torch
import h5py
import numpy as np
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.utils.checkpoint
from transformers import BertPreTrainedModel, BertModel

from probeDataPreproce import findMax_len, input_ids_attention_masks, load_dataCSV


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

with h5py.File("probe_trainData_", 'w') as fout:
    for step, batch in enumerate(test_dataloader):
        print(batch[0].shape)
        batch_te = torch.tensor(batch) 
            
        dset = fout.create_dataset(str(step), batch_te.shpae)
        dset[:, :, :] = np.vstack([np.array(batch_te)])

print("Finish!")
