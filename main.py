import os
import random
import datetime, time
from torch.utils import tensorboard
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard.summary import hparams
from transformers import BertTokenizer, BertModel, DistilBertForSequenceClassification, AlbertTokenizer, \
    AlbertForSequenceClassification, AlbertModel
import tqdm
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

# %%
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='OffenseEval')
parser.add_argument('--max_len', default=60, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--batch_size', default=64, type=int)
#

# parser.add_argument('--dataset', default='Emotion')
# parser.add_argument('--max_len', default=60, type=int)
# parser.add_argument('--epochs', default=6, type=int)
# parser.add_argument('--num_classes', default=4, type=int)
# parser.add_argument('--batch_size', default=64, type=int)

# parser.add_argument('--dataset', default='Sarcasm')
# parser.add_argument('--max_len', default=60, type=int)
# parser.add_argument('--epochs', default=6, type=int)
# parser.add_argument('--num_classes', default=2, type=int)
# parser.add_argument('--batch_size', default=64, type=int)


parser.add_argument('--device', default='cuda')
parser.add_argument('--weight', action='store_false')
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--grad_clip', default=5., type=float)
parser.add_argument('--seed', default=1, type=int)
args = parser.parse_known_args()
print(args)

# %%
MAX_LEN = args[0].max_len
N_EPOCHS = args[0].epochs
NUM_CLASSES = args[0].num_classes
BATCH_SIZE = args[0].batch_size
EPSILON = 10e-13
LR = args[0].lr
DATASET = 'Dataset/' + args[0].dataset
WEIGHT = args[0].weight
GRAD_CLIP = args[0].grad_clip
device = torch.device(args[0].device) if torch.cuda.is_available() else torch.device('cpu')

SEED = args[0].seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# %%

time_string = (datetime.datetime.utcnow() + datetime.timedelta(seconds=12600)).replace(
    microsecond=0).isoformat('_')
hyper_parameters = {'batch': BATCH_SIZE, 'lr': LR, 'weight': WEIGHT}
hyper_parameters_string = '--'.join(
    [k + '=' + str(v) for k, v in vars(args[0]).items() if k in list(hyper_parameters.keys())])
tb = tensorboard.SummaryWriter(log_dir=f'runs/{args[0].dataset}/{hyper_parameters_string}/{time_string}')

# %%

df = pd.read_csv(DATASET + '/train.csv')
df.dropna(inplace=True)
label_counts = df.label.value_counts(normalize=True)

df.text = df.text.apply(lambda x: "[CLS] " + x + ' [SEP]')

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(' '.join(x.split()[:MAX_LEN])))

token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

token_ids_matrix = np.array(token_ids.apply(lambda x: x[:MAX_LEN] + [0] * max(0, MAX_LEN - len(x))).tolist()).astype(
    'int64')

attention_matix = (token_ids_matrix != 0).astype('float')

train_dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(attention_matix),
                              torch.tensor(np.array(df.label)))

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%
df = pd.read_csv(DATASET + '/test.csv')

df.text = df.text.apply(lambda x: "[CLS] " + x + ' [SEP]')

tokenized_texts = df.text.apply(lambda x: tokenizer.tokenize(' '.join(x.split()[:MAX_LEN])))

token_ids = tokenized_texts.apply(tokenizer.convert_tokens_to_ids)

token_ids_matrix = np.array(token_ids.apply(lambda x: x[:MAX_LEN] + [0] * max(0, MAX_LEN - len(x))).tolist()).astype(
    'int64')

attention_matix = (token_ids_matrix != 0).astype('float')

test_dataset = TensorDataset(torch.tensor(token_ids_matrix), torch.tensor(attention_matix),
                             torch.tensor(np.array(df.label)))

test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# %%
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AlbertModel.from_pretrained(
            "albert-base-v2", num_labels=NUM_CLASSES)
        self.ff = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, NUM_CLASSES if NUM_CLASSES > 2 else 1)
        )

    def forward(self, batch_input, batch_mask):
        xx = self.transformer(batch_input, batch_mask)
        xx = self.ff(xx[0][:, 0, :]).squeeze()
        return xx


model = MyModel()
model = model.cuda()
# %%
param_optimizer = list(model.named_parameters())

# no_decay = ['bias', 'gamma', 'beta']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
#      'weight_decay_rate': 0.0}
# ]


optimizer = AdamW(model.parameters(), lr=LR)


# %%
def calc_confusion_matrix(y, y_hat):
    with torch.no_grad():
        if NUM_CLASSES > 2:
            pred = y_hat.argmax(-1)
        else:
            pred = y_hat.gt(0)

    m = confusion_matrix(y.cpu(), pred.cpu(), range(NUM_CLASSES))
    return m


def calc_metrics(m):
    p = (m.diagonal() / m.sum(0).clip(EPSILON)).mean()
    r = (m.diagonal() / m.sum(1).clip(EPSILON)).mean()
    f1 = ((2 * p * r) / (p + r)).mean()
    accu = m.diagonal().sum() / m.sum()
    return p, r, f1, accu


train_weights = (1 / label_counts)
train_weights = train_weights.to_numpy() / train_weights.sum()
train_weights = torch.Tensor(train_weights).to(device)

if WEIGHT:
    if NUM_CLASSES > 2:
        criterion = nn.CrossEntropyLoss(reduction='sum', weight=train_weights)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=train_weights[1])
else:
    if NUM_CLASSES > 2:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

metrics_all = []

for i_epoch in range(1, N_EPOCHS + 1):

    model.train()

    # Tracking variables
    train_loss = 0
    train_confusion_table = np.zeros((NUM_CLASSES, NUM_CLASSES))
    train_total = 0
    metrics_train = ()

    p_bar = tqdm(train_data_loader)
    for step, batch in enumerate(p_bar):
        batch = tuple(t.to(device) for t in batch)
        batch_input, batch_mask, batch_y = batch
        optimizer.zero_grad()

        logits = model(batch_input, batch_mask)

        loss = criterion(logits, batch_y if NUM_CLASSES > 2 else batch_y.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        assert grad_norm >= 0, 'encounter nan in gradients.'
        optimizer.step()

        train_confusion_table += calc_confusion_matrix(batch_y, logits)
        train_total += batch_input.size(0)
        train_loss += loss.item()

        metrics_train = (
            train_loss / train_total,
            *calc_metrics(train_confusion_table),
        )

        p_bar.set_description(

            '[ EP {0:02d} ]'
            '[ TRN LS: {1:.4f} PR: {2:.4f} F1: {4:.4f} AC: {5:.4f}]'
                .format(i_epoch, *metrics_train))

    for k, v in zip(['train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_accuracy'], metrics_train):
        tb.add_scalar(k, v * 100, global_step=i_epoch)

    with torch.no_grad():
        # Set our model to testing mode (as opposed to evaluation mode)
        model.eval()

        # Tracking variables
        # test_loss = 0
        test_confusion_table = np.zeros((NUM_CLASSES, NUM_CLASSES))
        test_total = 0

        # test the data for one epoch
        p_bar = tqdm(test_data_loader)
        for step, batch in enumerate(p_bar):
            batch = tuple(t.to(device) for t in batch)
            batch_input, batch_mask, batch_y = batch

            logits = model(batch_input, batch_mask)

            test_confusion_table += calc_confusion_matrix(batch_y, logits)
            test_total += batch_input.size(0)
            # test_loss += loss.item()

            metrics_test = (
                # test_loss / test_total,
                *calc_metrics(test_confusion_table),
            )

            p_bar.set_description(
                '[ EP {0:02d} ]'
                '[ TST PR: {1:.4f} F1: {3:.4f} AC: {4:.4f}]'
                    .format(i_epoch, *metrics_test))

    for k, v in zip(['test_precision', 'test_recall', 'test_f1', 'test_accuracy'], metrics_test):
        tb.add_scalar(k, v * 100, global_step=i_epoch)

    metrics_all.append((*metrics_train, *metrics_test))

metrics = np.array(metrics_all).T
best_index = metrics[-2].argmax()
best_metrics = metrics.T[best_index]


a = (hyper_parameters,
               dict(zip(['z_precision', 'z_recall', 'z_f1', 'z_accuracy', 'z_ep'], best_metrics[-4:])))

for j in hparams(*a):
    tb.file_writer.add_summary(j)

for k, v in a[1].items():
    tb.add_scalar(k, v)

tb.close()

# %%
