# %load train.py
import torch
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import numpy as np

from fixed_model import fixedBertClassification
from dataPreproce import True_dataPrepare
from utils import format_time, flat_accuracy, flat_matthews


# ---------------------------------------- Train ----------------------------------------------------
def train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device, index, seed):

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
        total_cola_loss = 0
        total_prob_loss = 0
        total_eval_matthews = 0
        model.train()

        # For each batch of training data... 
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments arge given and what flags are set.
            # For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model outputs prior to activation.
            # [ optimizer.zero_grad() feature,label.cuda y = model(x) loss = criterion(y,label) ]
            #loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            outputs, prob_loss, cola_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, epoch = epoch_i)
            loss = outputs.loss
            logits = outputs.logits
            total_train_loss += loss.item()
            total_cola_loss += cola_loss.item()
            total_prob_loss += prob_loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_cola_loss = total_cola_loss / len(train_dataloader)
        avg_prob_loss = total_prob_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.5f}".format(avg_train_loss))
        print("  Average cola loss: {0:.5f}".format(avg_cola_loss))
        print("  Average prob loss: {0:.5f}".format(avg_prob_loss))
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
        total_eval_cola_loss = 0
        total_eval_prob_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            with torch.no_grad():
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs, prob_loss, cola_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
            total_eval_cola_loss += cola_loss.item()
            total_eval_prob_loss += prob_loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_matthews += flat_matthews(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_matthews = total_eval_matthews / len(validation_dataloader)
        print("  Matthews: {0:.5f}".format(avg_val_matthews))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.5f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'Layer': index,
                'epoch': epoch_i + 1,
                'Training Loss': "{0:.5f}".format(avg_val_loss),
                'Training Prob Loss': "{0:.5f}".format(avg_prob_loss),
                'Training Cola Loss': "{0:.5f}".format(avg_cola_loss),
                'Valid. Loss': "{0:.5f}".format(avg_val_loss),
                'Valid. Matthews.': "{0:.5f}".format(avg_val_matthews),
                'Training Time': training_time,
                'Validation Time': validation_time,
                'Seed': seed
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    
    return training_stats
    #torch.save(model, 'bertClass-large_savel{jack}_{tom}.pt'.format(tom=seed, jack = index))
    #print("Model saved !")


if __name__ == '__main__':
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataloader, validation_dataloader = True_dataPrepare()

    epochs = 3
    total_steps = len(train_dataloader) * epochs
    """
    for i in range(0, 25):
        model = fixedBertClassification(layer_index = i)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        warmup_ratio = 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_ratio*total_steps,
                                                    num_training_steps=total_steps)
        train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device, i, seed_val)
    """

    total_stats = []
    for i in range(0, 25):
        model_stats = []
        for j in range(1, 41, 4):
            seed_val = j
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            torch.cuda.manual_seed_all(seed_val)
            
            model = fixedBertClassification(layer_index = i)
            model.to(device)
            optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
            warmup_ratio = 0.1
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_ratio*total_steps, num_training_steps=total_steps)
            train_stats = train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs, device, i, j)
            model_stats.append(train_stats)
        total_stats.append(model_stats)
    
    with open("prob_cola-seed1~40.csv", "w") as f:
        for i in range(len(total_stats)):
            for j in range(len(total_stats[i])):
                for k in range(len(total_stats[i][j])):
                    for index, key in enumerate(total_stats[i][j][k]):
                        f.write("%s\t%s\n" % (key, total_stats[i][j][k][key]))          
    
    print('Finish All')
