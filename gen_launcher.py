
import os
import sys
from pathlib import Path

def find_latest_subdir(parent):
    if not os.path.exists(parent):
        return None
    subdirs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
    if not subdirs:
        return None
    # Sort by version number logic roughly (alpha sort works for standard versions)
    subdirs.sort(key=lambda s: [int(p) for p in s.split('.') if p.isdigit()], reverse=True)
    return os.path.join(parent, subdirs[0])

def generate_launcher():
    vs_root = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    vcvars = os.path.join(vs_root, r"VC\Auxiliary\Build\vcvars64.bat")
    
    # 1. MSVC Libs
    msvc_root = os.path.join(vs_root, r"VC\Tools\MSVC")
    latest_msvc = find_latest_subdir(msvc_root)
    msvc_lib = os.path.join(latest_msvc, r"lib\x64") if latest_msvc else ""

    # 2. Windows SDK Libs
    sdk_root = r"C:\Program Files (x86)\Windows Kits\10\Lib"
    latest_sdk_dir = find_latest_subdir(sdk_root)
    
    ucrt_lib = ""
    um_lib = ""
    if latest_sdk_dir:
        ucrt_lib = os.path.join(latest_sdk_dir, r"ucrt\x64")
        um_lib = os.path.join(latest_sdk_dir, r"um\x64")

    # 3. Python Libs
    python_libs = r"C:\Users\FinnK\AppData\Local\Programs\Python\Python38\libs"

    print(f"Detected MSVC Lib: {msvc_lib}")
    print(f"Detected UCRT Lib: {ucrt_lib}")
    print(f"Detected UM Lib: {um_lib}")

    # Generate the Batch File
    batch_content = [
        "@echo off",
        f'call "{vcvars}" > nul',
        "setlocal enabledelayedexpansion",
        "",
        ":: Appending detected paths explicitly",
    ]
    
    if msvc_lib:
        batch_content.append(f'set "LIB=%LIB%;{msvc_lib}"')
    if ucrt_lib:
        batch_content.append(f'set "LIB=%LIB%;{ucrt_lib}"')
    if um_lib:
        batch_content.append(f'set "LIB=%LIB%;{um_lib}"')
        
    batch_content.append(f'set "LIB=%LIB%;{python_libs}"')
    
    batch_content.append("")
    batch_content.append("echo [Launcher] Environment set. Libs added.")
    batch_content.append("echo [Launcher] Running command: %*")
    batch_content.append("%*")
    
    with open("run_compiler_safe.bat", "w") as f:
        f.write("\n".join(batch_content))
    
    print("Successfully generated run_compiler_safe.bat")

if __name__ == "__main__":
    generate_launcher()
