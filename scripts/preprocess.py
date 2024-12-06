# 데이터 전처리를 위한 필수 라이브러리 임포트
import pandas as pd  
from datasets import Dataset 
import argparse

def prepare_data(input_path: str, output_path: str):
    """
    감성 분석을 위한 데이터 전처리 함수
    
    Args:
        input_path: 원본 CSV 파일 경로
        output_path: 전처리된 데이터를 저장할 경로 
    """
    # 1. 데이터 로드
    # CSV 파일을 pandas DataFrame으로 읽어와 메모리에 로드
    df = pd.read_csv(input_path)
    
    # 2. 컬럼명 영문으로 변경
    df = df.rename(columns={
        '상호명': 'store_name',
        '주소': 'address', 
        '작성자': 'writer',
        '평점들': 'ratings',
        '방문일': 'visit_date',
        '리뷰내용':'review_content',
        '태그들': 'tags'
    })

    # 3. 리뷰 내용이 null인 행 제거
    df = df.dropna(subset=['review_content'])
    df.to_csv(output_path, index=False)

    # 4. 처리 완료 메시지 출력
    print("전처리 데이터 저장 완료:")
    print(f"전처리 데이터: {output_path}")

if __name__ == "__main__":

    # 커맨드 라인 인자 파서 생성
    parser = argparse.ArgumentParser(description='데이터 전처리 스크립트')
    parser.add_argument('--input_path', required=True, help='입력 CSV 파일 경로')
    parser.add_argument('--output_path', required=True, help='학습 데이터 출력 경로')

    # 인자 파싱
    args = parser.parse_args()

    # prepare_data 함수 실행
    prepare_data(
        input_path=args.input_path,
        output_path=args.output_path
    )