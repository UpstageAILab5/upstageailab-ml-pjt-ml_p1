import shutil
from airflow.providers.ssh.hooks.ssh import SSHHook
import os

def fetch_model(remote_path, local_path):
    """원격 서버에서 모델 파일을 가져오는 함수"""
    
    # Airflow SSH connection 사용
    ssh_hook = SSHHook(ssh_conn_id='upstage_remote_server_ssh')
    
    try:
        # SSH 연결
        ssh_client = ssh_hook.get_conn()
        sftp = ssh_client.open_sftp()

        # 로컬 디렉토리가 이미 존재하면 내용을 삭제
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        # 로컬 디렉토리가 없으면 생성
        os.makedirs(local_path, exist_ok=True)
        
        # 원격 디렉토리의 모든 파일을 가져오기
        for f in sftp.listdir(remote_path):
            sftp.get(
                os.path.join(remote_path, f),
                os.path.join(local_path, f)
            )
            
    finally:
        sftp.close()
        ssh_client.close()