# logger_util.py
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def create_log_file(prefix="debate_log"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.txt"
    path = os.path.join(LOG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Debate log started: {datetime.now().isoformat()}\n\n")
    return path

class FileLogger:
    def __init__(self, path):
        self.path = path

    def log(self, text):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {text}\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def info(self, text):
        self.log(text)
