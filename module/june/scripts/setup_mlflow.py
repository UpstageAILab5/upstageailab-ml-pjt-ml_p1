import cmd
import subprocess
import time
import mlflow
import os

# MLflow 서버 설정
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:30162"  # 원격 서버 주소

def setup_mlflow_server():
    """
    MLflow 서버를 설정하고 실행하는 함수
    """
    try:
        # MLflow 서버 실행을 위한 기본 디렉토리 설정
        mlflow_dir = "/data/ephemeral/home/mlruns"
        if not os.path.exists(mlflow_dir):
            os.makedirs(mlflow_dir)
            
        print("MLflow 서버를 시작합니다...")
        print("MLflow UI는 http://0.0.0.0:30162 에서 접근할 수 있습니다")
        
        # MLflow 서버 실행 명령어
        os.system(f"mlflow server \
            --backend-store-uri {mlflow_dir} \
            --default-artifact-root {mlflow_dir} \
            --host 0.0.0.0 \
            --port 30162")
            
    # nohup을 사용하여 백그라운드로 실행
        with open("/data/ephemeral/home/mlflow.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=log_file,
                stderr=log_file,
                preexec_fn=os.setsid  # 새로운 프로세스 그룹 생성
            )
        
        # 서버가 시작될 때까지 잠시 대기
        time.sleep(5)
        
        # 프로세스 ID 저장
        with open("/data/ephemeral/home/mlflow.pid", "w") as pid_file:
            pid_file.write(str(process.pid))
            
        print(f"MLflow 서버가 백그라운드에서 시작되었습니다. (PID: {process.pid})")
            
    except Exception as e:
        print(f"MLflow 서버 설정 중 오류가 발생했습니다: {str(e)}")
        
        
if __name__ == "__main__":
    setup_mlflow_server()
