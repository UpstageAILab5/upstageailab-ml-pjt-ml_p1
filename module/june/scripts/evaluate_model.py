# 필요한 라이브러리 임포트
import evaluate  # Hugging Face의 평가 메트릭을 사용하기 위한 라이브러리
import numpy as np  # 배열 및 행렬 연산을 위한 수치 계산 라이브러리 
import pandas as pd  # 데이터프레임 처리를 위한 데이터 분석 라이브러리
from transformers import pipeline  # Hugging Face의 사전 학습 모델을 쉽게 사용하기 위한 파이프라인
import mlflow  # MLflow를 사용하여 모델 평가 메트릭을 기록하는 라이브러리

def evaluate_model(model_dir: str, test_data_path: str):
    """
    학습된 감성 분석 모델의 성능을 평가하는 함수
    
    Args:
        model_dir: 파인튜닝된 TinyBERT 모델이 저장된 디렉토리 경로
        test_data_path: 모델 성능 평가에 사용할 테스트 데이터셋의 CSV 파일 경로
    """

     # MLflow 실험 시작
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("sentiment_analysis")
    
    with mlflow.start_run():
        mlflow.autolog()  # 자동 로깅 활성화
        
        accuracy = evaluate.load('accuracy')

        classifier = pipeline(
            'text-classification',
            model=model_dir,
            tokenizer=model_dir,
            truncation=True,
            max_length=512,
            padding=True
        )
        
        test_df = pd.read_csv(test_data_path)
        predictions = classifier(test_df['review'].tolist())
        pred_labels = [1 if pred['label'] == 'positive' else 0 for pred in predictions]
        
        acc_score = accuracy.compute(predictions=pred_labels, references=test_df['label'].tolist())
        
        # MLflow에 메트릭 기록
        mlflow.log_metric("accuracy", acc_score['accuracy'])
        mlflow.log_param("model_dir", model_dir)
        mlflow.log_param("test_data_path", test_data_path)
        
        print(f"모델 평가 결과:")
        print(f"정확도: {acc_score['accuracy']:.4f}")