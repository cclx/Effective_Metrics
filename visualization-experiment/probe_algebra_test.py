# %load probe_test.py
# %load test.py
# %load fixed_train.py
# %load train.py
import torch
import time
import datetime
import random
import numpy as np
import math
import sys

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.utils.checkpoint
from transformers import BertPreTrainedModel, BertModel


from tool_model import Compute_norms
from draw import draw_blueprint, draw_varprint
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from transformers import BertTokenizer
from utils import format_time

from probeDataPreproce import load_dataCSV, findMax_len


print('Testing...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
    
bert = BertModel.from_pretrained("bert-large-uncased")
bert.to(device)
bert.eval()

print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)


# ---------------------------------------- Testing ----------------------------------------------------
def test(text, layer):

    sentence = text
    
    print("sentence: ", sentence)
    
    #print("length of sentence: ", len(sentence))
    
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
    
    prob_loss = None 
    Prob_loss_fct = MSELoss()
    prob_loss = Prob_loss_fct(norms, pseu_labels)
    
    print("layer:", layer, "prob_loss:", prob_loss)
    """
    for i in range(len(pseu_labels)):
        for j in range(len(pseu_labels[i])):
            print(tokenizer_sentence[j], ": ", pseu_labels[i][j], normalize_norms[i][j], norms[i][j])
            print(tokenizer_sentence[j], "的方向估计: ", normalize_norms[i][j]/pseu_labels[i][j], normalize_norms[i][j]/norms[i][j])
            print("========")
    """
    tokenizer_sentence = tokenizer_sentence
    #pseu_label = np.array(pseu_labels[0])
    norm = norms[0].detach().cpu().numpy()
    pseu_label = pseu_labels[0].detach().cpu().numpy()
    summ = [0.0 for i in range(0, int(max(pseu_label)))]
    abssumm = [0.0 for i in range(0, int(max(pseu_label)))]
    cout = [0.0 for i in range(0, int(max(pseu_label)))]
    for i in range(len(pseu_label)):
        summ[int(pseu_label[i])-1] += norm[i] - pseu_label[i]
        abssumm[int(pseu_label[i])-1] += abs(norm[i] - pseu_label[i])
        cout[int(pseu_label[i])-1] += 1
     
    average = [summ[i]/cout[i] for i in range(0, int(max(pseu_label)))]
    absaverage = [abs(abssumm[i]/cout[i]) for i in range(0, int(max(pseu_label)))]
    
    max_depth = max(pseu_label)
    
    ave_var = [0.0 for i in range(0, int(max(pseu_label)))]
    absave_var = [0.0 for i in range(0, int(max(pseu_label)))]
    
    for i in range(len(pseu_label)):
        ave_var[int(pseu_label[i])-1] += math.pow((norm[i] - pseu_label[i] - average[int(pseu_label[i])-1]), 2)
        absave_var[int(pseu_label[i])-1] += math.pow((abs(norm[i] - pseu_label[i]) - absaverage[int(pseu_label[i])-1]), 2)
    
    for i in range(len(cout)):
        if(cout[i] == 1.0):
            ave_var[i] = 0.0
            absave_var[i] = 0.0
        else:
            ave_var[i] = math.pow(ave_var[i]/(cout[i]-1), 0.5)
            absave_var[i] = math.pow(absave_var[i]/(cout[i]-1), 0.5)
    
    
    print("average: ", average)
    
    
    #draw_probeTree(tokenizer_sentence, pseu_label, norm, name)
    #print('Finish All')
    
    print(ave_var)
    print(absave_var)

    #print("")
    #print("Testing complete!")
    return average, absaverage, ave_var, absave_var


def var_test():
    
    sentences = load_dataCSV()
    max_len = findMax_len(sentences)
    total_var = []
    total_absvar = []
    total_max_depth = []
    
    sample_number = 1000
    
    index = [i for i in range(len(sentences))]
    
    sample = np.random.choice(index, sample_number, replace = False)
    
    for i in range(5, 10):
        var = [0.0 for j in range(max_len)]
        absvar = [0.0 for j in range(max_len)]
        cout = [0.0 for j in range(max_len)]
        max_depth = 0.0
        for k in range(0, sample_number):
            text = sentences[sample[k]]
            average, absaverage, ave_var, absave_var = test(text, i)
            if(len(ave_var)>max_depth):
                max_depth = len(ave_var)
            for j in range(len(ave_var)):
                var[j] += ave_var[j]
                cout[j] += 1
                absvar[j] += absave_var[j]
                
        for j in range(max_depth):
            var[j] = var[j] / cout[j]
            absvar[j] = absvar[j] / cout[j]
        
        total_var.append(var)
        total_absvar.append(absvar)
        total_max_depth.append(max_depth)
    
    print("total_max_depth: ", total_max_depth)
    
    with open("total_max_depthl5~9.csv", "w") as f:
        
        for j in range(len(total_max_depth)):
            f.write("%s\n" % (total_max_depth[j]))
        f.close()

    
    for i in range(5, 10, 5):
        
        with open("probe_multi-layer_varprint-l{tom}~{jerry}.csv".format(tom=i, jerry=i+4), "w") as f:
            for j in range(5):
                for k in range(len(total_var[i+j])):
                    f.write("%s\t" % "{0:.5f}".format((total_var[i+j][k])))
                    f.write("\n")
            f.close()
        name = "probe_multi-layer_varprint-l{tom}~{jerry}".format(tom=i, jerry=i+4)
        draw_varprint(total_var, i, name, total_max_depth)
        #draw_blueprint(y, i, name, text)
        
        with open("probe_multi-layer_absvarprint-l{tom}~{jerry}.csv".format(tom=i, jerry=i+4), "w") as f:
            for j in range(5):
                for k in range(len(total_absvar[i+j])):
                    f.write("%s\t" % "{0:.5f}".format((total_absvar[i+j][k])))
                    f.write("\n")
            f.close()
        absname = "probe_multi-layer_absvarprint-l{tom}~{jerry}".format(tom=i, jerry=i+4)
        
        draw_varprint(total_absvar, i, absname, total_max_depth)
        #draw_blueprint(absy, i, absname, text)
    
    print("Finish!")


if __name__ == '__main__':
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    """
    var_test()
    """
    arg = sys.argv
    text = str(arg[1])
    """
    loader = text.split(" ")
    random.shuffle(loader)
    text = ' '.join(loader)
    """
    print(text)
    
    for i in range(0, 25, 5):
        y = []
        absy = []
        for j in range(i, i+5):
            average, absaverage, ave_var, absave_var = test(text, j)
            y.append(average)
            absy.append(absaverage)
        
        name = str(arg[2])+"-l{tom}~{jerry}".format(tom=i, jerry=i+4)
        draw_blueprint(y, i, name, text)
    
