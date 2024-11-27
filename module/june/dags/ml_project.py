# 필요한 라이브러리 임포트
import datetime
from airflow import DAG  # Airflow DAG 객체 임포트
from airflow.operators.python import PythonOperator  # Python 작업을 실행하기 위한 Operator
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.dates import days_ago  # DAG 시작 날짜 설정을 위한 유틸리티
import os
import sys

# scripts 디렉토리 경로를 Python 경로에 추가
# 이를 통해 scripts 폴더 내의 Python 모듈을 임포트할 수 있음
sys.path.append('/opt/airflow/scripts')

# ML 파이프라인의 각 단계를 처리할 모듈들 임포트
# from preprocess import prepare_data  # 데이터 전처리 모듈
# from train_model import train_model  # 모델 학습 모듈
from fetch_files import fetch_model  # 모델 파일 가져오기 모듈
from evaluate_model import evaluate_model  # 모델 평가 모듈
from select_best_model import select_best_model  # 최적 모델 선택 모듈

# DAG의 기본 설정값 정의
# 이 설정들은 DAG 내의 모든 태스크에 공통적으로 적용됨
default_args = {
    'owner': 'airflow',  # DAG 소유자 지정
    'start_date': days_ago(1),  # DAG 시작 날짜 (어제부터)
    'retries': 3,  # 태스크 실패시 최대 재시도 횟수
    'retry_delay': datetime.timedelta(minutes=1),  # 재시도 간 대기 시간
}

# DAG 정의
with DAG(
    'ml_project',  # DAG의 고유 식별자
    default_args=default_args,  # 위에서 정의한 기본 설정 적용
    description='감성 분석을 위한 ML 파이프라인',  # DAG 설명
    schedule_interval='@daily',  # 실행 주기 (매일)
    catchup=False,  # 과거 실행 건너뛰기
    tags=['ml', 'sentiment_analysis']  # DAG 분류를 위한 태그
) as dag:

    # 1단계: 데이터 전처리 태스크
    # CSV 파일을 읽어서 학습용/테스트용 데이터셋으로 분할
    prepare_data_task = SSHOperator(
        task_id='prepare_data',  # 태스크 식별자
        ssh_conn_id='upstage_remote_server_ssh',  # Airflow에 설정된 SSH 연결 ID
        command='''
            /opt/conda/bin/python /data/ephemeral/home/scripts/preprocess.py \
            --input_path /data/ephemeral/home/data/tinybert_model.csv \
            --train_output_path /data/ephemeral/home/data/tinybert_train.csv \
            --test_output_path /data/ephemeral/home/data/tinybert_test.csv
        '''
    )

    setup_mlflow_task = SSHOperator(
        task_id='setup_mlflow',  # 태스크 식별자
        ssh_conn_id='upstage_remote_server_ssh',  # Airflow에 설정된 SSH 연결 ID
        command='''
            /opt/conda/bin/python /data/ephemeral/home/scripts/setup_mlflow.py 
        '''
    )

    # 2단계: 모델 학습 태스크
    # 전처리된 데이터를 사용하여 TinyBERT 모델 학습
    train_model_task = SSHOperator(
        task_id='train_model',
        ssh_conn_id='upstage_remote_server_ssh',  # Airflow에 설정된 SSH 연결 ID
        command='''
            /opt/conda/bin/python /data/ephemeral/home/scripts/train_model.py \
            --input_train_path /data/ephemeral/home/data/tinybert_train.csv \
            --input_test_path /data/ephemeral/home/data/tinybert_test.csv \
            --model_output_dir /data/ephemeral/home/models/tinybert-sentiment-analysis \
            --train_output_dir /data/ephemeral/home/train_dir
        ''',
        cmd_timeout=10800
    )

    fetch_tinybert_model_task = PythonOperator(
        task_id='fetch_tinybert_model',
        python_callable=fetch_model,
        op_kwargs={
            'remote_path': '/data/ephemeral/home/models/tinybert-sentiment-analysis',
            'local_path': '/opt/airflow/models/tinybert-sentiment-analysis'
        }
    )

    
    # fetch_train_dir_task = PythonOperator(
    #     task_id='fetch_train_dir',
    #     python_callable=fetch_model,
    #     op_kwargs={
    #         'remote_path': '/data/ephemeral/home/train_dir',
    #         'local_path': '/opt/airflow/train_dir'
    #     }
    # )

    # 3단계: 모델 평가 태스크
    # 학습된 모델의 성능을 테스트 데이터셋으로 평가
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        op_kwargs={
            'model_dir': '/opt/airflow/models/tinybert-sentiment-analysis',
            'test_data_path': '/opt/airflow/data/tinybert_test.csv',
        },
    )

    # 4단계: 최적 모델 선택 태스크
    # 여러 모델 중 가장 성능이 좋은 모델 선택
    select_best_model_task = PythonOperator(
        task_id='select_best_model',
        python_callable=select_best_model,
        op_kwargs={
            'models_dir': '/opt/airflow/models',
        },
    )

    # 태스크 간의 실행 순서 정의
    prepare_data_task >> setup_mlflow_task >> train_model_task >> fetch_tinybert_model_task >> evaluate_model_task >> select_best_model_task