import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, DistilBertForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification
import tqdm
from tqdm import tqdm
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET = 'Dataset/OffenseEval'
# MAX_LEN = 60
# N_EPOCHS = 5
# NUM_CLASSES = 4
# BATCH_SIZE=32
#
# DATASET = 'Dataset/Emotion'
# MAX_LEN = 60
# N_EPOCHS = 5
# NUM_CLASSES = 4
# BATCH_SIZE=32

DATASET = 'Dataset/Sarcasm'
MAX_LEN = 60
N_EPOCHS = 5
NUM_CLASSES = 4
BATCH_SIZE=32


#%%

df = pd.read_csv(DATASET + '/train.csv')
df.dropna(inplace=True)

df.text = df.text.apply(lambda x: "[CLS] " + x + ' [SEP]')

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(' '.join(x.split()[:MAX_LEN])))

token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

token_ids_matrix = np.array(token_ids.apply(lambda x: x[:MAX_LEN] + [0] * max(0, MAX_LEN - len(x))).tolist()).astype('int64')

attention_matix = (token_ids_matrix != 0).astype('float')

train_dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(attention_matix), torch.tensor(np.array(df.label)))

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


#%%
df = pd.read_csv(DATASET + '/test.csv')

df.text = df.text.apply(lambda x: "[CLS] " + x + ' [SEP]')

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(' '.join(x.split()[:MAX_LEN])))

token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

token_ids_matrix = np.array(token_ids.apply(lambda x: x[:MAX_LEN] + [0] * max(0, MAX_LEN - len(x))).tolist()).astype('int64')

attention_matix = (token_ids_matrix != 0).astype('float')

test_dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(attention_matix), torch.tensor(np.array(df.label)))

test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

#%%

model = AlbertForSequenceClassification.from_pretrained(
    "albert-base-v2", num_labels=NUM_CLASSES)
model = model.cuda()

#%%
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


optimizer = AdamW(model.parameters(), lr=2e-5)

#%%
train_loss_set = []

# trange is a tqdm wrapper around the normal python range
for ep in range(N_EPOCHS):
    print('EPOCH', ep)

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    train_loss = 0
    train_acc = 0
    train_total, train_steps = 0, 0

    # Train the data for one epoch
    p_bar =tqdm(train_data_loader)
    for step, batch in enumerate(p_bar):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

        train_acc += logits.argmax(1).eq(b_labels).long().sum().item()
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        train_loss += loss.item()
        train_total += b_input_ids.size(0)
        train_steps += 1


        p_bar.set_description('Loss {:.4f} Acc {:.4f}'.format(train_loss / train_steps, train_acc / train_total))

    print('\nTrain Loss {:.4f} Acc {:.4f}'.format(train_loss / train_steps, train_acc / train_total))



    # Training

    with torch.no_grad():
        # Set our model to testing mode (as opposed to evaluation mode)
        model.eval()

        # Tracking variables
        test_loss = 0
        test_acc = 0
        test_total, test_steps = 0, 0


        # test the data for one epoch
        p_bar = tqdm(test_data_loader)
        for step, batch in enumerate(p_bar):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            # Forward pass
            loss, logits = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            test_acc += logits.argmax(1).eq(b_labels).long().sum().item()


            # Update tracking variables
            test_loss += loss.item()
            test_total += b_input_ids.size(0)
            test_steps += 1

            p_bar.set_description('Loss {:.4f} Acc {:.4f}'.format(test_loss / test_steps, test_acc / test_total))

        print("\nTest Loss {:.4f} Acc {:.4f}".format(test_loss / test_steps, test_acc / test_total))
