
import shutil
import subprocess
import os
import sys

def test_empty_env():
    ffmpeg_path = shutil.which('ffmpeg')
    command = [ffmpeg_path, '-version']
    
    print(f"Running: {command}")
    print("Testing with env={}")
    
    try:
        proc = subprocess.Popen(
            command,
            env={},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Popen with empty env success!")
    except OSError as e:
        print(f"Popen with empty env failed: {e}")

    print("\nTesting with env=None (inherit)")
    try:
        proc = subprocess.Popen(
            command,
            env=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Popen with inherited env success!")
    except OSError as e:
        print(f"Popen with inherited env failed: {e}")

if __name__ == "__main__":
    test_empty_env()
