# Apache Airflow 2.8.1 버전을 기반으로 하는 도커 이미지 생성
# Python 3.10 버전 사용
FROM apache/airflow:2.8.1-python3.10

# Airflow가 사용할 홈 디렉토리 환경변수 설정 
# 모든 Airflow 관련 파일들이 이 디렉토리 아래에 저장됨
ENV AIRFLOW_HOME=/opt/airflow

# root 사용자로 전환하여 시스템 레벨의 설치 작업 수행
USER root

# 필수 시스템 패키지 설치
# gcc: C 컴파일러 - 일부 Python 패키지 빌드에 필요
# libc-dev: C 표준 라이브러리 개발 파일
# python3-tk: Python Tkinter GUI 라이브러리 
# tk-dev: Tk GUI 툴킷 개발 파일
# supervisor: 프로세스 관리자
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    python3-tk \
    tk-dev \
    supervisor && \
    rm -rf /var/lib/apt/lists/*  # 설치 후 캐시 제거로 이미지 크기 최적화

# airflow 사용자로 전환하여 Python 패키지 관련 작업 수행
USER airflow
# pip 패키지 관리자 최신 버전으로 업그레이드
RUN pip install --upgrade pip

# requirements.txt 파일을 컨테이너에 복사하고 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 다시 root 사용자로 전환하여 시스템 설정 작업 수행
USER root

# 주석 처리된 스크립트 복사 및 실행 권한 설정 (필요시 주석 해제)
# COPY scripts/ /opt/airflow/scripts/
# RUN chmod +x /opt/airflow/scripts/*.py

# Airflow 작업 디렉토리 생성 및 작업 디렉토리로 설정
RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME

# Supervisor 설정 파일 복사 및 권한 설정
# Supervisor는 Airflow의 여러 컴포넌트(웹서버, 스케줄러 등)를 관리
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
RUN chown airflow: /etc/supervisor/conf.d/supervisord.conf

# Airflow DAG 파일들을 컨테이너의 dags 디렉토리로 복사
COPY dags/ $AIRFLOW_HOME/dags/

# 컨테이너가 사용할 포트 설정 (Airflow 웹 서버용)
EXPOSE 8080

# 컨테이너 시작 시 Supervisor 실행
# Supervisor가 설정 파일에 정의된 대로 Airflow 프로세스들을 관리
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
