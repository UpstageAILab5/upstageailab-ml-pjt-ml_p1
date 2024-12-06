import pandas as pd
import numpy as np

from tokenizers import Tokenizer
from transformers import AutoTokenizer,ElectraModel, ElectraTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding # GPU 있다면 사용가능
import torch
from sklearn.model_selection import train_test_split
import re
from datasets import Dataset
from ReviewDataset import ReviewDataset


# 데이터 불러오기

file_path = '/data/ephemeral/home/data_pred/labeled_review_data_20241205_030506.csv'
data = pd.read_csv(file_path)

data = data.dropna(subset=['predicted_label','Review_Text'])
dataframe = data[['predicted_label','Review_Text']]
dataframe.rename(columns={'predicted_label': 'Label'}, inplace=True)


# 모델 불러오기

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("./kcElectra_k_up")
tokenizer = AutoTokenizer.from_pretrained("./kcElectra_k_up", use_fast=True)


def preprocess_text(text):
    # 특수문자 제거
    # text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 데이터 전처리 및 토큰화

def tokenize_data(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


# 데이터 전처리 적용

dataframe['Review_Text'] = dataframe['Review_Text'].astype(str).apply(preprocess_text)

def tokenize(batch) :
    temp = tokenizer(batch['Review_Text'], padding=True, truncation=True, max_length=400, return_tensors="pt")
    return temp

dataset = Dataset.from_pandas(dataframe)
dataset = dataset.map(tokenize, batched=True)


# 데이터셋 객체 생성

dataset = ReviewDataset(dataset, tokenizer)


# test data 불러오기

train_dataset = pd.read_csv('train_dataset.csv')
eval_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = eval_dataset.map(tokenize, batched=True)


# 학습 인자 설정
training_args = TrainingArguments(
        output_dir='./results/fin',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        seed=42,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 평가 메트릭 정의
def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = np.mean(labels == preds)
        return {'accuracy': accuracy}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer 초기화 및 학습 시작
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )


# 모델 학습

trainer.train()


# 학습된 모델 저장

model.save_pretrained('./kcElectra_k_up')
tokenizer.save_pretrained('./kcElectra_k_up')
