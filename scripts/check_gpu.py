"""
Diagnose why PyTorch cannot use the GPU (the CUDA/PyTorch initialization
problem seen on SOL). Run it INSIDE a GPU job, not on the login node:

    srun <your usual GPU flags> --pty python scripts/check_gpu.py
    # or submit scripts/screen_sol.sbatch with the command replaced by this script

It prints facts and the most likely cause. It changes nothing.
"""

import os
import shutil
import subprocess
import sys


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"(failed: {error})"


print("python          :", sys.version.split()[0], sys.executable)
print("host            :", os.uname().nodename)
print("SLURM_JOB_ID    :", os.environ.get("SLURM_JOB_ID", "(not in a Slurm job)"))
print("SLURM_JOB_GPUS  :", os.environ.get("SLURM_JOB_GPUS", os.environ.get("SLURM_GPUS", "(unset)")))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
print("CUDA_HOME       :", os.environ.get("CUDA_HOME", "(unset)"))

print("\n-- nvidia-smi --")
if shutil.which("nvidia-smi"):
    print(run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]) or "(no GPU listed)")
else:
    print("nvidia-smi not found (login node, or no GPU allocated)")

print("\n-- torch --")
try:
    import torch
except ImportError as error:
    print("torch is not installed in this environment:", error)
    sys.exit(1)

print("torch version     :", torch.__version__)
print("torch built for CUDA:", torch.version.cuda)
print("cuda available    :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device            :", torch.cuda.get_device_name(0))
    try:
        a = torch.randn(1024, 1024, device="cuda")
        print("matmul on GPU     : OK", float((a @ a).sum().item()) != 0.0)
    except Exception as error:  # noqa: BLE001
        print("matmul on GPU FAILED:", error)
    print("\nGPU works. If model loading still fails, the problem is elsewhere (memory, model access).")
    sys.exit(0)

print("\n-- likely cause --")
if "+cpu" in torch.__version__ or torch.version.cuda is None:
    print("This is a CPU-ONLY torch build. Reinstall a CUDA build, e.g.\n"
          "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
          "(pick the cuXXX that matches the driver CUDA version shown by nvidia-smi).")
elif not shutil.which("nvidia-smi"):
    print("No GPU is visible to this process: you are probably on the login node, or the job did\n"
          "not request a GPU. Request one in the job (see scripts/screen_sol.sbatch).")
elif os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
    print("CUDA_VISIBLE_DEVICES is set to an empty string, which hides every GPU. Unset it.")
else:
    print("A GPU is listed but torch cannot initialize it. Typical causes:\n"
          " - torch was built for a newer CUDA than the node's driver supports (compare\n"
          "   'torch built for CUDA' above with the driver version; install an older cuXXX build)\n"
          " - a conflicting CUDA module is loaded (try `module purge`, then activate your env)\n"
          " - a broken or mismatched install (create a fresh env and reinstall torch)")
sys.exit(1)
