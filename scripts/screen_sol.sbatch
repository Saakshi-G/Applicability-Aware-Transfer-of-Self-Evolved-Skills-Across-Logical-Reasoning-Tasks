#!/bin/bash
# SOL GPU job that screens one model. Submit from the repo root:
#
#   mkdir -p logs
#   sbatch scripts/screen_sol.sbatch Qwen/Qwen2.5-7B-Instruct
#
# Resource flags follow ASU Research Computing's public-user example
# (-p htc -q public, -G for the GPU). Not tested on SOL by me.
# If you submit under a class account, your QoS/account may differ:
# run `myaccounts` on SOL and ask Research Computing which -q / -A to use.

#SBATCH --job-name=screen-model
#SBATCH --output=logs/screen_%j.out
#SBATCH --error=logs/screen_%j.err
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -t 0-2                     # 2 hours is plenty for 250 questions on a 7B model
#SBATCH -G a100:1                  # one A100
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --export=NONE              # start from a clean environment

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"

# --- environment ---
module load mamba/latest
source activate skills             # the mamba env you create once (see the steps I sent)

# Keep downloaded weights out of your home directory.
export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_DISABLE_XET=1
export BBH_DIR="$PWD/external/Big-Bench-Hard/bbh"
export LLM_BACKEND=hf
export HF_BATCH_SIZE=8
export MAX_NEW_TOKENS=1536

# Some models (e.g. Llama) need a Hugging Face access token: run `huggingface-cli login` once.

python scripts/check_gpu.py || exit 1
python experiments/screen_model.py --backend hf --model "$MODEL"
