# 필요한 라이브러리 임포트
import evaluate  # Hugging Face의 평가 메트릭을 사용하기 위한 라이브러리
import numpy as np  # 배열 및 행렬 연산을 위한 수치 계산 라이브러리 
import pandas as pd  # 데이터프레임 처리를 위한 데이터 분석 라이브러리
from transformers import pipeline  # Hugging Face의 사전 학습 모델을 쉽게 사용하기 위한 파이프라인

def evaluate_model(model_dir: str, test_data_path: str):
    """
    학습된 감성 분석 모델의 성능을 평가하는 함수
    
    Args:
        model_dir: 파인튜닝된 TinyBERT 모델이 저장된 디렉토리 경로
        test_data_path: 모델 성능 평가에 사용할 테스트 데이터셋의 CSV 파일 경로
    """
    # Hugging Face의 accuracy 메트릭을 로드
    # 이진 분류 문제의 정확도를 계산하는데 사용됨
    accuracy = evaluate.load('accuracy')

    classifier = pipeline(
        'text-classification',
        model=model_dir,
        tokenizer=model_dir,
        truncation=True,
        max_length=512,
        padding=True  # 필요 시 패딩을 추가
    )
    
    # CSV 형식의 테스트 데이터를 판다스 데이터프레임으로 로드
    test_df = pd.read_csv(test_data_path)
    
    # 테스트 데이터의 리뷰 텍스트에 대해 모델 예측 수행
    predictions = classifier(test_df['review'].tolist())
    # 예측 결과를 이진 레이블로 변환 (positive -> 1, negative -> 0)
    pred_labels = [1 if pred['label'] == 'positive' else 0 for pred in predictions]
    
    # 예측값과 실제값을 비교하여 정확도 계산
    # predictions: 모델이 예측한 레이블
    # references: 실제 정답 레이블
    acc_score = accuracy.compute(predictions=pred_labels, references=test_df['label'].tolist())
    
    # 평가 결과 출력
    # 소수점 4자리까지 정확도 표시
    print(f"모델 평가 결과:")
    print(f"정확도: {acc_score['accuracy']:.4f}")
