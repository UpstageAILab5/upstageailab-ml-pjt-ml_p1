# 데이터 전처리를 위한 필수 라이브러리 임포트
import pandas as pd  # 데이터프레임 처리를 위한 pandas 라이브러리
from datasets import Dataset  # Hugging Face의 데이터셋 처리 라이브러리

def prepare_data(input_path: str, train_output_path: str, test_output_path: str):
    """
    감성 분석을 위한 데이터 전처리 함수
    
    Args:
        input_path: 원본 CSV 파일 경로
        train_output_path: 전처리된 학습 데이터를 저장할 경로 
        test_output_path: 전처리된 테스트 데이터를 저장할 경로
    """
    # 1. 데이터 로드
    # CSV 파일을 pandas DataFrame으로 읽어와 메모리에 로드
    df = pd.read_csv(input_path)
    
    # 2. 데이터셋 변환
    # pandas DataFrame을 Hugging Face에서 사용하는 Dataset 형식으로 변환
    # 이는 효율적인 데이터 처리와 모델 학습을 위해 필요
    dataset = Dataset.from_pandas(df)
    
    # 3. 데이터 분할
    # 전체 데이터를 학습용(70%)과 테스트용(30%) 데이터로 분할
    dataset = dataset.train_test_split(test_size=0.3)
    
    # 4. 레이블 인코딩
    # 텍스트 형태의 감성 레이블을 숫자로 변환하기 위한 매핑 정의
    # negative는 0, positive는 1로 인코딩
    label2id = {'negative': 0, 'positive': 1}
    
    # 5. 레이블 변환
    # 데이터셋의 'sentiment' 열을 숫자 레이블로 변환하는 매핑 함수 적용
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})
    
    # 6. 데이터 저장
    # 전처리된 학습 데이터와 테스트 데이터를 CSV 파일로 저장
    pd.DataFrame(dataset['train']).to_csv(train_output_path, index=False)
    pd.DataFrame(dataset['test']).to_csv(test_output_path, index=False)
    
    # 7. 처리 완료 메시지 출력
    print("학습 및 테스트 데이터 저장 완료:")
    print(f"학습 데이터: {train_output_path}")
    print(f"테스트 데이터: {test_output_path}")