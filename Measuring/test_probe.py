# %load train.py
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import probeDataPreproce
from probe_trainModel import probe
from utils import format_time, flat_accuracy


def test(model, train_dataloader, validation_dataloader, device, Layer):

    training_stats = []        
    total_t0 = time.time()    
    
    t0 = time.time()
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            
        layer_features = batch[0][0].to(device)
            
        with torch.no_grad():
            loss = model(layer_features)
            
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.5f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    

    print("")
    print("Running Validation...")

    t0 = time.time()

    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        layer_features = batch[0][0].to(device)
        with torch.no_grad():
            loss = model(layer_features)

        total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.5f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    
    return training_stats




if __name__ == '__main__':
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    with open("ssp-layer1.csv", "w") as f:
        for i in range(1, 2):
            train_dataloader, validation_dataloader = probeDataPreproce.dataPrepare("probe_trainData{:}".format(i))
            model = torch.load('ssp_savel{:}.pt'.format(i), map_location = device)
            model.eval()
            model.to(device)
            training_stats = test(model, train_dataloader, validation_dataloader, device, i)
            for index, key in enumerate(training_stats[len(training_stats)-1]):
                f.write("%s\t%s\n" % (key, training_stats[len(training_stats)-1][key]))

    print('Finish All')
