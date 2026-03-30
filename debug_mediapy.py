
import shutil
import subprocess
import os
import sys

def debug_ffmpeg():
    print(f"Python: {sys.executable}")
    ffmpeg_path = shutil.which('ffmpeg')
    print(f"shutil.which('ffmpeg'): {ffmpeg_path}")
    
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found in PATH")
        return

    # Simulate mediapy command construction
    width, height = 100, 100
    fps = 10.0
    tmp_name = "debug_output.mp4"
    
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
        '-y', tmp_name
    ]
    
    print(f"Command: {command}")
    
    try:
        print("Attempting subprocess.Popen...")
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Popen success!")
        out, err = proc.communicate(input=b'')
        print(f"Return code: {proc.returncode}")
        print(f"Stderr: {err.decode('utf-8')}")
    except Exception as e:
        print(f"Popen failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ffmpeg()
