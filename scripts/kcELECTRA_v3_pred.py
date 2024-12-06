# pip install datasets transformers torch evaluate transformers[torch] accelerate
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import mlflow
from transformers import ElectraForSequenceClassification, ElectraTokenizer
import torch
import datetime
import os

# MLflow 서버 연결 설정
os.environ["MLFLOW_TRACKING_URI"] = "http://10.196.197.32:30164"

def get_next_run_number():
    # 현재 실험의 모든 실행을 가져옴
    experiment = mlflow.get_experiment_by_name("kcELECTRA_v3_pred")
    if experiment is None:
        return 1
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # 기존 run_name에서 가장 큰 번호 찾기
    max_run_number = 0
    for run_name in runs['tags.mlflow.runName']:
        if run_name and 'kcELECTRA_v3_pred_' in run_name:
            try:
                number = int(run_name.split('_')[-1])
                max_run_number = max(max_run_number, number)
            except ValueError:
                continue
    
    return max_run_number + 1
# 결과 예측
def predict_sentiment(input_path: str, output_path: str):
    mlflow.set_experiment("kcELECTRA_v3_pred")
    
    # 다음 실행 번호 가져오기
    run_number = get_next_run_number()
    run_name = f"kcELECTRA_v3_pred_{run_number}"
    
    mlflow.start_run(run_name=run_name)
    
    try:
        # 파라미터 기록
        mlflow.log_params({
            "model_name": "monologg/koelectra-base-v3-discriminator",
            "max_length": 300,
            "threshold": 0.5,
            "input_path": input_path,
            "output_path": output_path,
            "run_number": run_number  # 실행 번호도 파라미터로 기록
        })

        df = pd.read_csv(input_path)
        
        # 감성 분석용 모델 로드
        model_name = "monologg/koelectra-base-v3-discriminator"
        tokenizer = ElectraTokenizer.from_pretrained(model_name)

        # 저장된 모델 로드
        saved_model_path = "/data/ephemeral/home/scripts/saved_model"  # 저장된 모델 경로
        model = ElectraForSequenceClassification.from_pretrained(saved_model_path)
            
        def predict_single_text(text):
            # 텍스트 전처리 및 토큰화
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=300,
                padding=True
            )
            
            # 예측
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                positive_prob = predictions[0][1].item()
                
            # 결과 반환 (0: 부정, 1: 긍정)
            return 1 if positive_prob > 0.5 else 0, positive_prob
            
        # DataFrame에 적용
        results = []
        for text in df['review_content']:
            label, prob = predict_single_text(text)
            results.append({
                'text': text,
                'predicted_label': label,
                'probability': prob  # probability 추가
            })
        
        # 결과를 DataFrame에 추가
        df['label'] = [r['predicted_label'] for r in results]
        
        df.to_csv(output_path, index=False)
        print(f"CSV 파일이 저장되었습니다: {output_path}")

        # 모델 예측 결과를 MLflow에 아티팩트로 저장
        mlflow.log_artifact(output_path, "predictions")
        
        # 예측 결과 메트릭 기록 부분 수정
        positive_count = sum(1 for r in results if r['predicted_label'] == 1)
        negative_count = sum(1 for r in results if r['predicted_label'] == 0)

        # 예측 확률 분포 계산
        probabilities = [r['probability'] for r in results]
        avg_confidence = sum(probabilities) / len(probabilities)
        high_confidence_preds = sum(1 for p in probabilities if p > 0.9 or p < 0.1)

        mlflow.log_metrics({
            "positive_predictions": positive_count,
            "negative_predictions": negative_count,
            "total_predictions": len(results),
            "average_confidence": avg_confidence,
            "high_confidence_predictions": high_confidence_preds,
            "high_confidence_ratio": high_confidence_preds / len(results)
        })

        # 모델 버전 관리를 위한 태그 추가
        mlflow.set_tag("model_version", "v1")
        mlflow.set_tag("model_status", "production")
        mlflow.set_tag("deployment_timestamp", datetime.datetime.now().isoformat())
                
    finally:
        # MLflow 실험 종료
        mlflow.end_run()

if __name__ == "__main__":
    # 커맨드 라인 인자 파서 생성
    parser = argparse.ArgumentParser(description='첫번째 모델 스크립트')
    parser.add_argument('--input_path', required=True, help='CSV 파일 입력 경로')
    parser.add_argument('--output_path', required=True, help='1차 모델 예측 파일 출력 경로')

    # 인자 파싱
    args = parser.parse_args()

    # prepare_data 함수 실행
    predict_sentiment(
        input_path=args.input_path,
        output_path=args.output_path,
    )