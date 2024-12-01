import mlflow
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import numpy as np
from transformers import pipeline
import argparse
import os
import time
from datetime import datetime

os.environ["NO_PROXY"] = "*"

# MLflow 서버 연결 설정
os.environ["MLFLOW_TRACKING_URI"] = "http://10.196.197.32:30162"

def train_model(input_train_path: str, input_test_path: str, model_output_dir: str, train_output_dir: str):
    """
    감성 분석을 위한 TinyBERT 모델 학습 함수
    """
    # MLflow 실험 설정
    mlflow.set_experiment("sentiment_analysis")
    
    # MLflow autolog 활성화
    mlflow.pytorch.autolog()
    
    # 실행 이름에 타임스탬프 추가
    run_name = f"tinybert_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # 토크나이저 및 평가 메트릭 설정
        tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", use_fast=True)
        accuracy = evaluate.load('accuracy')
        
        def tokenize(batch):
            return tokenizer(batch['review'], padding=True, truncation=True, max_length=512)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=-1)
            results = accuracy.compute(predictions=predictions, references=labels)
            
            # 추가 메트릭 계산 (예: 클래스별 정확도)
            class_accuracies = {
                f"class_{label}_accuracy": np.mean(predictions[labels == label] == label)
                for label in np.unique(labels)
            }
            results.update(class_accuracies)
            
            return results

        # 데이터 로드 및 전처리
        train_df = pd.read_csv(input_train_path)
        test_df = pd.read_csv(input_test_path)
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        id2label = {0: 'negative', 1: 'positive'}
        label2id = {'negative': 0, 'positive': 1}
        
        train_dataset = train_dataset.map(tokenize, batched=True, batch_size=None)
        test_dataset = test_dataset.map(tokenize, batched=True, batch_size=None)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D",
            num_labels=2,
            label2id=label2id,
            id2label=id2label
        )
        
        # 학습 파라미터 설정
        training_args = TrainingArguments(
            output_dir=train_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy='epoch',
            logging_dir=f"{train_output_dir}/logs",
            logging_steps=100,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy"
        )
        
        # Trainer 설정
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )
        
        # 모델 학습 수행
        trainer.train()
        
        # 최종 평가
        trainer.evaluate()
        
        # 모델 저장
        trainer.save_model(model_output_dir)
    
    print("Model training and evaluation completed. Model saved to", model_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 학습 스크립트')
    parser.add_argument('--input_train_path', required=True, help='학습 데이터 CSV 파일 경로')
    parser.add_argument('--input_test_path', required=True, help='테스트 데이터 CSV 파일 경로')
    parser.add_argument('--model_output_dir', required=True, help='학습된 모델을 저장할 경로')
    parser.add_argument('--train_output_dir', required=True, help='학습 중간 결과물을 저장할 경로')
    
    args = parser.parse_args()
    
    train_model(
        input_train_path=args.input_train_path,
        input_test_path=args.input_test_path,
        model_output_dir=args.model_output_dir,
        train_output_dir=args.train_output_dir
    )