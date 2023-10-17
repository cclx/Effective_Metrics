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


# ---------------------------------------- Train ----------------------------------------------------
def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device):

    training_stats = []        
    total_t0 = time.time()      

    # For each epoch...
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        
        t0 = time.time()
        
        total_train_loss = 0
        
        model.train()

        # For each batch of training data...  
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
           
            layer_features = batch[0][0].to(device)
            #print(layer_features.shape)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments arge given and what flags are set.
            # For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model outputs prior to activation.
            # [ optimizer.zero_grad() feature,label.cuda y = model(x) loss = criterion(y,label) ]
            loss = model(layer_features)


            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            layer_features = batch[0][0].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            with torch.no_grad():
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                loss = model(layer_features)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
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

    torch.save(model, 'psdProbe_savel22.pt')
    print("Model saved !")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataloader, validation_dataloader = probeDataPreproce.dataPrepare("probe_trainData22")


    model = probe(args={'probe': {'maximum_rank': int(1024/2)}, 'model': {'hidden_dim': 1024}})

    model.to(device)

    epochs = 10
    total_steps = len(train_dataloader) * epochs

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device)

    print('Finish All')
