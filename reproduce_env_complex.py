
import shutil
import subprocess
import os
import sys

def test_empty_env_complex():
    ffmpeg_path = shutil.which('ffmpeg')
    width, height = 100, 100
    fps = 10.0
    tmp_name = "debug_env_output.mp4"
    
    command = [
        ffmpeg_path,
        '-v', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', f'{fps}',
        '-i', '-',
        '-an',
        '-vcodec', 'h264',
        '-pix_fmt', 'yuv420p',
        '-vb', '20',
        '-qp', '20',
        '-y', tmp_name
    ]
    
    print(f"Running complex command with env={{}}")
    
    try:
        proc = subprocess.Popen(
            command,
            env={},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Popen success!")
        out, err = proc.communicate(input=b'')
        print(f"Return code: {proc.returncode}")
        print(f"Stderr: {err.decode('utf-8', errors='ignore')}")
    except OSError as e:
        print(f"Popen failed: {e}")

if __name__ == "__main__":
    test_empty_env_complex()
