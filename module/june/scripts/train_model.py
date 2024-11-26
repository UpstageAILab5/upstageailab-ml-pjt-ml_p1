# 필요한 라이브러리 임포트
import pandas as pd  # 데이터 처리를 위한 pandas
from datasets import Dataset  # Hugging Face의 데이터셋 처리
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments  # 트랜스포머 모델 관련
import evaluate  # 모델 평가 메트릭
import numpy as np  # 수치 연산
from transformers import pipeline  # 추론 파이프라인

def train_model(input_train_path: str, input_test_path: str, model_output_dir: str, train_output_dir: str):
    """
    감성 분석을 위한 TinyBERT 모델 학습 함수
    
    Args:
        input_train_path: 학습 데이터 CSV 파일 경로
        input_test_path: 테스트 데이터 CSV 파일 경로  
        model_output_dir: 학습된 모델을 저장할 경로
        train_output_dir: 학습 중간 결과물을 저장할 경로
    """
    # TinyBERT 토크나이저 초기화 - 빠른 토크나이저 사용
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", use_fast=True)
    
    # 모델 성능 평가를 위한 정확도 메트릭 로드
    accuracy = evaluate.load('accuracy')
    
    def tokenize(batch):
        """
        입력 텍스트를 토큰화하는 함수
        최대 길이 300으로 패딩/자르기 수행
        """
        return tokenizer(batch['review'], padding=True, truncation=True, max_length=300)
    
    def compute_metrics(eval_pred):
        """
        모델 예측 결과의 정확도를 계산하는 함수
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)  # 가장 높은 확률의 클래스 선택
        return accuracy.compute(predictions=predictions, references=labels)
    
    # CSV 파일에서 학습/테스트 데이터 로드
    train_df = pd.read_csv(input_train_path)
    test_df = pd.read_csv(input_test_path)
    
    # pandas DataFrame을 HuggingFace Dataset 형식으로 변환
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 레이블 인코딩을 위한 매핑 정의
    id2label = {0: 'negative', 1: 'positive'}  # 숫자 -> 텍스트
    label2id = {'negative': 0, 'positive': 1}  # 텍스트 -> 숫자
    
    # 데이터셋의 텍스트를 토큰화
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=None)
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=None)
    
    # TinyBERT 모델 초기화 - 이진 분류(긍정/부정)를 위한 설정
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D",
        num_labels=2,  # 이진 분류이므로 2개 레이블
        label2id=label2id,
        id2label=id2label
    )
    
    # 학습 파라미터 설정
    args = TrainingArguments(
        output_dir=train_output_dir,  # 학습 결과물 저장 경로
        overwrite_output_dir=True,  # 기존 결과물 덮어쓰기
        num_train_epochs=3,  # 전체 데이터셋 3회 반복 학습
        learning_rate=2e-5,  # 미세조정을 위한 적절한 학습률
        per_device_train_batch_size=32,  # GPU당 학습 배치 크기
        per_device_eval_batch_size=32,  # GPU당 평가 배치 크기
        evaluation_strategy='epoch'  # 매 에포크마다 평가 수행
    )
    
    # 학습을 관리할 Trainer 객체 초기화
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # 모델 학습 수행 및 평가
    trainer.train()  # 학습 실행
    trainer.evaluate()  # 최종 평가
    trainer.save_model(model_output_dir)  # 학습된 모델 저장
    
    print("Model training and evaluation completed. Model saved to", model_output_dir)
