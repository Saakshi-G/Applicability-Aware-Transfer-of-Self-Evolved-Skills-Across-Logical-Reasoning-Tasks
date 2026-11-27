#!/bin/bash
# usage: sbatch scripts/evolve_sol2.sbatch <model> <tag> [extra run_skill_evolution.py args]
#SBATCH --job-name=evolve2
#SBATCH --output=logs/evolve2_%j.out
#SBATCH --error=logs/evolve2_%j.err
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -t 0-3
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --export=NONE

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
TAG="${2:-run}"
SLUG=$(echo "$MODEL" | tr '/' '_')_${TAG}

module load mamba/latest
source activate skills
export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_DISABLE_XET=1
export BBH_DIR="$PWD/external/Big-Bench-Hard/bbh"
export LLM_BACKEND=hf
export HF_MODEL="$MODEL"
export HF_BATCH_SIZE=16
export MAX_NEW_TOKENS=1536

python scripts/check_gpu.py || exit 1
python -u experiments/run_skill_evolution.py --eval-test \
    --out "skills/library_${SLUG}.json" \
    --log "skills/evolution_log_${SLUG}.jsonl" "${@:3}"
