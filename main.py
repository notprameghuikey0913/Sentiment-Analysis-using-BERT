!pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
import torch
!pip install transformers
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
import random


#Exploratory Data Analysis and Preprocessing

df = pd.read_csv("/smile-annotations-final.csv",
                 names = ['id', 'text', 'category'])
df.set_index('id', inplace = True)
# df.head()

# df.category.value_counts() # counts up how many times each unique instance occurs inside the data

df = df[~df.category.str.contains('\|')] # remove categories that have multiple emotions

df = df[df.category != 'nocode'] # remove categories that have "nocode" as the label

# df.category.value_counts()

possible_labels = df.category.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
# label_dict

df['labels'] = df.category.apply(lambda x: label_dict[x]) # convert emotions from 'string' to 'numerical' categories
# df.head()


# Training/Validation Split

X_train, X_test, y_train, y_test = train_test_split(
    df.index.values,
    df.labels.values,
    test_size = 0.15,
    random_state = 17,
    stratify = df.labels.values
)

df["data_type"] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_test, 'data_type'] = 'test'
# df.groupby(['category', 'labels', 'data_type']).count() # check the quantity spread out in train/test split



#  Loading Tokenizer and Encoding the Data

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', # using all lowercase data
    do_lower_case = True # convert everything to lower case because we're using
                         # uncased model of BERT
)

# convert all sentences from tweets from language to encoded form (tokens)
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].text.values,
    add_special_tokens = True, #BERT's way of knowing whena sent ends and a new one begins
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 256,
    return_tensors = 'pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    df[df.data_type == 'test'].text.values,
    add_special_tokens = True, #BERT's way of knowing when a sen ends and a new one begins
    return_attention_mask = True,
    pad_to_max_length = True,
    max_length = 256,
    return_tensors = 'pt'
)

#encoded_data_train returns a dictionary

#input ids basically represent each word as a number
input_ids_train = torch.tensor(encoded_data_train['input_ids'])
attention_masks_train = torch.tensor(encoded_data_train['attention_mask'])
labels_train = torch.tensor(df[df.data_type == 'train'].labels.values)

input_ids_test = torch.tensor(encoded_data_test['input_ids'])
attention_masks_test = torch.tensor(encoded_data_test['attention_mask'])
labels_test = torch.tensor(df[df.data_type == 'test'].labels.values)

# creating train and test datasets of tensor values of the encoded data from
# the previous cell

dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train,
                              labels_train)

dataset_test = TensorDataset(input_ids_test,
                             attention_masks_test,
                             labels_test)

# len(dataset_train)
# len(dataset_test)

# Setting up BERT Pretrained Model

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(label_dict),
    output_attentions = False,
    output_hidden_states = False
)

# Creating Data Loaders

# offer a way to iterate through your dataset in batches
batch_size = 4

dataloader_train = DataLoader(
    dataset_train,
    sampler = RandomSampler(dataset_train),
    batch_size = batch_size
)

dataloader_val = DataLoader(
    dataset_test,
    sampler = RandomSampler(dataset_test),
    batch_size = 32
)

# Setting Up Optimizer and Scheduler

optimizer = AdamW(
    model.parameters(),
    lr = 1e-5, # 2e-5 5e-5
    eps = 1e-8
)
epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = 0,
    num_training_steps = len(dataloader_train)*epochs
)

# Defining Performance Metrics

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis =1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels):
    labels_dict_inverse = {v: k for k, v in  label_dict.items()}

    preds_flat = np.argmax(preds, axis =1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {labels_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}')


# Creating the Training Loop

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# checkwhat device you have and set it properly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) # transfer the model to the device
# print(device)

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in tqdm(dataloader_val):

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)):

    model.train() # send model to training mode

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train,
                       desc = 'Epoch {:1d}'.format(epoch),
                       leave = False, # let it override itself after each epoch
                       disable = False
                       ) # how many batches have been trained and how many to go
    for batch in progress_bar:

        model.zero_grad() # initialize grads to 0

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids':      batch[0],
            'attention_mask': batch[1],
            'labels':         batch[2]

        }

        outputs = model(**inputs) # unpacks the dictionary straight into the input

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward() #backprop of bert

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


    #torch.save(model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_values = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_values)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')


# Loading and Evaluating the Model

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)
# print(device)

# _, predictions, true_vals = evaluate(dataloader_train)

# accuracy_per_class(predictions, true_vals)
