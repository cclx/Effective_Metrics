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

def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device, layer_index):

    training_stats = []         
    total_t0 = time.time()      

    for epoch_i in range(epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
           
            layer_features = batch[0][0].to(device)
            model.zero_grad()

            loss = model(layer_features)

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))



        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

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

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    
    with open("ssp_l{:}.csv".format(layer_index), "w") as f:
        for i in range(len(training_stats)):
            for index, key in enumerate(training_stats[i]):
                f.write("%s\t%s\n" % (key, training_stats[i][key]))

    torch.save(model, 'ssp_savel{:}.pt'.format(layer_index))
    print("Model saved !")


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    epochs = 10

    l = 21
    r = 22
   
    for i in range(l, r):
        train_dataloader, validation_dataloader = probeDataPreproce.dataPrepare("probe_trainData{:}".format(i))
        model = probe(args={'probe': {'maximum_rank': int(1024/2)}, 'model': {'hidden_dim': 1024}})
        #model = essp(args={'probe': {'maximum_rank': int(1024/2)}, 'model': {'hidden_dim': 1024}})
        model.to(device)
        total_steps = len(train_dataloader) * epochs
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device, i)

    print('Finish All')
