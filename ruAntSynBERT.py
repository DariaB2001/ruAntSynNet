"""
Код для дообучения нейросетевой модели rubert-base-cased для классификации триплетов вида: слово1 | слово2 | контекст.
"""

print('Импорт библиотек...')
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch
import random
import os
import csv
import pandas as pd
from pprint import pprint


def set_seed(seed_value=42):
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  random.seed(seed_value)
  os.environ['PYTHONHASHSEED'] = str(seed_value)

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


set_seed()

# Загрузка датасета
print('Загрузка обучающего датасета...')
path_to_train_dataset = 'YOUR_PATH'  # путь к обучающему датасету
dataset_train_val = []
with open(path_to_train_dataset, encoding='utf-8') as f1:
  reader = csv.reader(f1)
  for row in reader:
    if len(row) == 2:
      dataset_train_val.append([row[0], int(row[1])])

df_train_val = pd.DataFrame(dataset_train_val, columns=['text', 'label'])
print(df_train_val.head())

print('Загрузка тестового датасета...')
path_to_test_dataset = 'YOUR_PATH'  # путь к тестовому датасету
dataset_test_raw = []
with open(path_to_test_dataset, encoding='utf-8') as f2:
  reader = csv.reader(f2)
  for row in reader:
    if len(row) == 2:
      dataset_test_raw.append([row[0], int(row[1])])

test_df = pd.DataFrame(dataset_test_raw, columns=['text', 'label'])
print(test_df.head())

# Отделяем от обучающей выборки валидационную в размере 20%
# от общего объёма данных, после чего превращаем все три датафрейма в объекты класса Dataset.
train_df, val_df = train_test_split(df_train_val, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df)

# Имея три датасета, создаём объект класса DatasetDict.
dataset = DatasetDict({
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
})
print(dataset)
pprint(dataset["train"][1000])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Используемое устройство: {device}")

# Загрузка токенизатора и модели
print('Загрузка токенизатора и модели...')
model_name = 'DeepPavlov/rubert-base-cased'

access_token = 'YOUR_ACCESS_TOKEN'  # Ваш access token с Huggingface
tokenizer = BertTokenizer.from_pretrained(model_name, token=access_token)
model = BertForSequenceClassification.from_pretrained(model_name, token=access_token, num_labels=2)
model.to(device)  # переместим модель на GPU (если доступен)

# Токенизация датасета
print('Токенизация датасета...')


def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length',
                     max_length=512, truncation=True, return_tensors='pt')


tokenized_datasets = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Обучение модели
print('Обучение модели...')
training_args = TrainingArguments(
    output_dir='results-bert-topic-cls',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',

    eval_strategy='epoch',  # Оценка качества модели - в конце каждой эпохи
    logging_steps=10,
    ## ----
    report_to="tensorboard",
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


train_dataset = tokenized_datasets['train']

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=tokenized_datasets['val'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

output_dir = 'OUTPUT_DIR'  # путь к папке, куда сохранится дообученная модель

print('Сохранение модели...')
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Оценка качества на тестовой выборке
print('Оценка качества модели на тестовой выборке...')
predictions = trainer.predict(tokenized_datasets['test'])
preds = np.argmax(predictions.predictions, axis=-1)

label_map ={
    0: 'Synonyms',
    1: 'Antonyms'
}

cm = confusion_matrix(predictions.label_ids, preds)

# Названия классов
labels = [label_map[i] for i in range(len(label_map))]

# Отрисуем confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Предсказанные метки классов')
plt.ylabel('Истинные метки классов')
plt.title('Confusion Matrix')
plt.show()

true = np.array(test_df['label'])  # истинные метки
print(classification_report(true, preds))
print(f'roc_auc_score: {round(roc_auc_score(true, preds), 2)}')
