# %load model.py
from transformers import BertForSequenceClassification, AdamW, BertConfig


def bertLinearClassification():
    # the pretrained BERT model with a single linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased",    # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,    # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,    # Whether the model returns attentions weights.
        output_hidden_states=False,    # Whether the model returns all hidden-states.
    )

    return model


def showModelParams(model):
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    return


if __name__ == '__main__':

    print('Test begin')

    model = bertLinearClassification()
    # Tell pytorch to run this model on the GPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    showModelParams(model)

    print('Test Finish')

