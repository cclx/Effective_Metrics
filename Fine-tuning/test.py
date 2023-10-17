# %load fixed_train.py
# %load train.py
import torch
import time
import datetime
import random
import numpy as np

from dataPreproce import dataPrepare2
from utils import format_time, flat_accuracy, flat_matthews


# ---------------------------------------- Testing ----------------------------------------------------
def test(model, test_dataloader, epochs, device):

    testing_stats = []        
    total_t0 = time.time()     

    # For each epoch...
    for epoch_i in range(epochs):

        # ========================================
        #                    Testing
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Testing...')

        # 单次epoch的时间
        t0 = time.time()
        # 单次epoch的loss
        total_test_loss = 0
        total_cola_loss = 0
        total_prob_loss = 0
        total_eval_matthews = 0
       
        # For each batch of testing data...  
        for step, batch in enumerate(test_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments arge given and what flags are set.
                # For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model outputs prior to activation.
                # [ optimizer.zero_grad() feature,label.cuda y = model(x) loss = criterion(y,label) ]
                #loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                total_test_loss += loss.item()
                
                # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_matthews += flat_matthews(logits, label_ids)
                
            

        avg_test_loss = total_test_loss / len(test_dataloader)

        testing_time = format_time(time.time() - t0)
        

        print("")
        print("  Average testing loss: {0:.2f}".format(avg_test_loss))
        print("  Testing epcoh took: {:}".format(testing_time))
        
        avg_test_matthews = total_eval_matthews / len(test_dataloader)
        print("  Matthews: {0:.2f}".format(avg_test_matthews))

        # Record all statistics from this epoch.
        testing_stats.append(
            {
                'epoch': epoch_i + 1,
                'testing Loss': avg_test_loss,
                'Test. Accur.': avg_test_matthews,
                'Testing Time': testing_time,
            }
        )

    print("")
    print("Testing complete!")

    print("Total testing took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    test_dataloader = dataPrepare2()


    model = torch.load('bertClass_save_notfixed_large.pt')
    model.eval()
    model.to(device)

    epochs = 1
    total_steps = len(test_dataloader) * epochs

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    test(model, test_dataloader, epochs, device)

    print('Finish All')
