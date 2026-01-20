import os
import time
import tempfile
import subprocess
import sys
from typing import Tuple

def format_time(seconds: float) -> str:
    """将秒数转换为 HH:MM:SS 格式"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class Logger:
    def __init__(self, model_name: str, task_name: str):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("model_test", model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"{task_name}_{self.timestamp}.log")
        self.file_handle = None

    def __enter__(self):
        self.file_handle = open(self.log_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()

    def write_header(self, columns: list):
        if self.file_handle:
            self.file_handle.write("\t".join(columns) + "\n")
            self.file_handle.flush()

    def log_result(self, data: dict, columns: list):
        if self.file_handle:
            row = [str(data.get(col, "")) for col in columns]
            self.file_handle.write("\t".join(row) + "\n")
            self.file_handle.flush()
    
    def log_summary(self, summary: str):
        if self.file_handle:
            self.file_handle.write("\n" + summary + "\n")
            self.file_handle.flush()
        print(summary)

    def get_log_path(self):
        return os.path.abspath(self.log_path)

class CodeExecutor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str) -> Tuple[str, str]:
        """
        执行 Python 代码并返回 (status, output/error)
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(code)
            
            # 执行
            result = subprocess.run(
                [sys.executable, tmp_file_path],
                capture_output=True,
                timeout=self.timeout
            )
            
            # 清理
            try:
                os.remove(tmp_file_path)
            except:
                pass

            stderr = result.stderr.decode('utf-8', errors='ignore')
            stdout = result.stdout.decode('utf-8', errors='ignore')
            
            if result.returncode == 0:
                return "PASSED", ""
            else:
                error_msg = stderr.strip() if stderr else (stdout.strip() if stdout else "Unknown Error")
                return "FAILED", error_msg.replace("\n", " | ")
                
        except subprocess.TimeoutExpired:
            return "TIMEOUT", "执行超时"
        except Exception as e:
            return "EXECUTION_ERROR", str(e)
