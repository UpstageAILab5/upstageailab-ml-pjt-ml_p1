import cmd
import subprocess
import time
import mlflow
import os

# MLflow 서버 설정
os.environ["NO_PROXY"] = "*"
os.environ["MLFLOW_TRACKING_URI"] = "http://0.0.0.0:30164"  # 원격 서버 주소

def cleanup_port():
    """포트를 사용하는 프로세스 정리"""
    try:
        # 포트를 사용하는 프로세스 찾기
        cmd = "lsof -ti:30164"
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if process.stdout:
            pids = process.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(f"kill -9 {pid}", shell=True)
                    print(f"프로세스 {pid} 종료됨")
                except:
                    pass
    except:
        pass

def setup_mlflow_server():
    try:

        # 기존 프로세스 정리
        cleanup_port()
        
        # MLflow 프로세스도 정리
        subprocess.run("pkill -f 'mlflow server'", shell=True)
        time.sleep(2)  # 프로세스가 완전히 종료되길 기다림
        
        mlflow_dir = "/data/ephemeral/home/mlruns"
        os.makedirs(mlflow_dir, exist_ok=True)
            
        print("MLflow 서버를 시작합니다...")
        
        # conda 환경을 포함한 전체 명령어 구성
        cmd = f"""
        source /opt/conda/etc/profile.d/conda.sh && \
        conda activate myenv && \
        mlflow server \
            --backend-store-uri sqlite:///{mlflow_dir}/mlflow.db \
            --default-artifact-root {mlflow_dir} \
            --host 0.0.0.0 \
            --port 30164 \

        """
            
        # subprocess.Popen을 사용하여 백그라운드로 실행
        with open("/data/ephemeral/home/mlflow.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                shell=True,
                executable='/bin/bash',  # bash 셸 명시적 지정
                stdout=log_file,
                stderr=log_file,
                preexec_fn=os.setsid
            )
        
        time.sleep(5)
        
        # 프로세스 ID 저장
        with open("/data/ephemeral/home/mlflow.pid", "w") as pid_file:
            pid_file.write(str(process.pid))
            
        print(f"MLflow 서버가 백그라운드에서 시작되었습니다. (PID: {process.pid})")
            
    except Exception as e:
        print(f"MLflow 서버 설정 중 오류가 발생했습니다: {str(e)}")
        
if __name__ == "__main__":
    setup_mlflow_server()