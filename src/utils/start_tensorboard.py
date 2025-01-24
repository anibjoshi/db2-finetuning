import subprocess
from pathlib import Path
from utils.config import TENSORBOARD_DIR

def start_tensorboard(port: int = 6006):
    """Start Tensorboard server.
    
    Args:
        port: Port number for Tensorboard server
    """
    cmd = f"tensorboard --logdir={TENSORBOARD_DIR} --port={port}"
    subprocess.Popen(cmd, shell=True)
    print(f"Tensorboard started at http://localhost:{port}") 