#  the book "NLP transfomers" chapter 2. 
import os
from transformers import pipeline
# classifier = pipeline("text-classification")

import datasets
print(datasets.__file__)
print(datasets.__path__)
from datasets import load_dataset
emotions  = load_dataset("emotion")
print(emotions)

# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 16000
#     })
#     validation: Dataset({
#         features: ['text', 'label'],
#         num_rows: 2000
#     })
#     test: Dataset({
#         features: ['text', 'label'],
#         num_rows: 2000
#     })
# })

train_ds = emotions["train"]
print(train_ds)
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 16000
# })

print(len(train_ds))
# 16000

print(train_ds.features)
# {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], id=None)}

print(train_ds[:5])

print(train_ds["text"][:5])

import pandas as pd
emotions.set_format(type='pandas')
df = emotions["train"][:]
print(df.head())

def lable_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(lable_int2str)
print(df.head())

import matplotlib.pyplot as plt
df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
input_ids = [token2idx[token] for token in tokenized_text]
categorical_df = pd.DataFrame({"Name": ["Bumblebee", "Optimus Prime", "Megation"], "Label ID": [0, 1, 2]})
print(categorical_df)
import torch
import torch.nn.functional as F
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings.shape)

print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")

from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

from transformers import DistilBertTokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokenizer.convert_tokens_to_string(tokens))

print(tokenizer.vocab_size)
print(tokenizer.model_max_length)


def tokenize(batch):
    _tokenized = tokenizer(batch["text"].values.tolist(), padding=True, truncation=True)
    _tokenized["text"] = batch["text"]
    _tokenized["label"] = batch["label"]
    return _tokenized

print(tokenize(emotions["train"][:2]))
emotions_encoded = emotions.map(tokenize, batched = True, batch_size=None)
print(f'emotions_encoded[train] column names: {emotions_encoded["train"].column_names}') 
print(f'emotions[train] column names: {emotions["train"].column_names}') 

from transformers import AutoModel
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

text = "this is a test"
inputs = tokenizer(text, return_tensors = "pt")
print(f"Input tensor shape： {inputs['input_ids'].size()}")

inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

print(outputs.last_hidden_state.size())

print(outputs.last_hidden_state[:,0].size())
# torch.Size([1,768])

def extract_hidden_state(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
    '''
    batch.items() = dict_items([('input_ids', tensor([[  101,  1045,  2134,  ...,     0,     0,     0],
        [  101,  1045,  2064,  ...,     0,     0,     0],
        [  101, 10047,  9775,  ...,     0,     0,     0],
        ...,
        [  101,  1045,  2131,  ...,     0,     0,     0],
        [  101,  1045,  2572,  ...,     0,     0,     0],
        [  101,  1045,  2318,  ...,     0,     0,     0]])), ('attention_mask', tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]]))])
    '''

    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] model
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotions_encoded.set_format("torch", columns = ["input_ids", "attention_mask"])
emotions_hidden = emotions_encoded.map(extract_hidden_state, batched=True)
print(f'emotions_hidden[train] column names: {emotions_hidden["train"].column_names}')
# ['input_ids', 'attention_mask', 'hidden_state']
import numpy as np
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
print(X_train.shape, X_valid.shape)

from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
# scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()

fig, axes = plt.subplots(2,3,figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]

labels = emotions["train"].features["label"].names
for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap, gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]),axes[i].set_yticks([])
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
# We increase max_iter to guarantee convergence
lr_clf = LogisticRegression(max_iter = 3000)
lr_clf.fit(X_train, y_train) # hidden_state vs label
lr_clf.score(X_valid, y_valid) # hidden_state vs label

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax= plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax =ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)

from transformers import AutoModelForSequenceClassification
num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels = num_labels)
         .to(device))

from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):  
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

from transformers import Trainer, TrainingArguments
batch_size = 64
logging_steps = len(emotions_encoded["train"])
model_name = f"{model_ckpt}-finetuned-emotion"

training_args = TrainingArguments(output_dir = "wtf",
                                  num_train_epochs=2,  
                                  learning_rate = 2e-5, 
                                  per_device_train_batch_size=batch_size, 
                                  per_device_eval_batch_size=batch_size, 
                                  weight_decay=0.01, 
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False, 
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")

from transformers import Trainer
trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train() # ValueError: The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask
preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)
y_preds = np.argmax(preds_output.predictions, axis = 1) 
plot_confusion_matrix(y_preds, y_valid, labels)

