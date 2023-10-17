# %load probe_test.py
# %load test.py
# %load fixed_train.py
# %load train.py
import torch
import time
import datetime
import random
import numpy as np

import torch.utils.checkpoint
from transformers import BertPreTrainedModel, BertModel


from tool_model import Compute_norms
from draw import draw_probeTree
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from transformers import BertTokenizer
from utils import format_time


# ---------------------------------------- Testing ----------------------------------------------------
def test(text, layer, name):

    print('Testing...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    sentence = text
    
    print("sentence: ", sentence)
    
    #print("length of sentence: ", len(sentence))
    
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    
    length = len(tokenizer.encode(sentence, add_special_tokens=True))
    
    print("length of sentence_ids: ", length)
    
    input_ids = []
    attention_masks = []
    
    encoded_dict = tokenizer.encode_plus(
        sentence,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=length,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    
    input_ids.append(encoded_dict['input_ids'])
    
    attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    
    attention_masks = torch.cat(attention_masks, dim=0)
    
    tokenizer_sentence = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("tokenizer_sentence: ", tokenizer_sentence)
    
    print("length of tokenizer_sentence: ", len(tokenizer_sentence))
    
    
    
    dataset = TensorDataset(input_ids, attention_masks)
    
    batch_size = 1
    
    dataloader = DataLoader(
        dataset,  
        sampler=RandomSampler(dataset),  
        batch_size=batch_size
    )
    
    bert = BertModel.from_pretrained("bert-large-uncased")
    bert.to(device)
    bert.eval()
    
    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
    
    normalize_normal = []
    layer_Index = layer
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
        
        output_hidden_states = outputs.hidden_states[layer_Index]
        
    #print(b_input_ids)
      

    model = torch.load('psdProbe_savel{}.pt'.format(layer), map_location=device)
    model.eval()
    model.to(device)
    
    normalize_norms = Compute_norms(output_hidden_states)
    norms = model.oneWordpsdProbe(output_hidden_states).to("cuda:0" if torch.cuda.is_available() else "cpu")
    pseu_labels = model.constructLabel(norms).to("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    for i in range(len(pseu_labels)):

        for j in range(len(pseu_labels[i])):
            print(tokenizer_sentence[j], ": ", pseu_labels[i][j], normalize_norms[i][j], norms[i][j])
            print(tokenizer_sentence[j], normalize_norms[i][j]/pseu_labels[i][j], normalize_norms[i][j]/norms[i][j])
            print("========")
    """
    tokenizer_sentence = tokenizer_sentence
    #pseu_label = np.array(pseu_labels[0])
    norm = norms[0].detach().cpu().numpy()
    pseu_label = pseu_labels[0].detach().cpu().numpy()
    draw_probeTree(tokenizer_sentence, pseu_label, norm, name)
    print('Finish All')



    print("")
    print("Testing complete!")



if __name__ == '__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    text = "the so out of be that backpack backpack took the would lighter. bottle the I water"
    #loader = text.split(" ")
    #random.shuffle(loader)
    #text = ' '.join(loader)
    #print("text")
    
    for i in range(0, 25):
        name = "probeTree_l{tom}-{jack}".format(tom = i, jack = 3)
        test(text, i, name)
