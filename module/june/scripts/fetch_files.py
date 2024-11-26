from airflow.providers.ssh.hooks.ssh import SSHHook
import os
import stat
import shutil  # 디렉토리 삭제를 위해 추가

def fetch_model(remote_path, local_path):
    ssh_hook = SSHHook(ssh_conn_id='upstage_remote_server_ssh')
    
    def copy_recursive(sftp, remote_dir, local_dir):
        """재귀적으로 디렉토리와 파일을 복사하는 함수"""
        # 기존 디렉토리가 있으면 삭제
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        
        os.makedirs(local_dir, exist_ok=True)
        
        for f in sftp.listdir(remote_dir):
            remote_path = os.path.join(remote_dir, f)
            local_path = os.path.join(local_dir, f)
            
            try:
                if stat.S_ISDIR(sftp.stat(remote_path).st_mode):
                    copy_recursive(sftp, remote_path, local_path)
                else:
                    sftp.get(remote_path, local_path)
            except Exception as e:
                print(f"Error processing {remote_path}: {str(e)}")
    
    try:
        ssh_client = ssh_hook.get_conn()
        sftp = ssh_client.open_sftp()
        
        # 재귀적으로 복사 실행
        copy_recursive(sftp, remote_path, local_path)
            
    finally:
        sftp.close()
        ssh_client.close()