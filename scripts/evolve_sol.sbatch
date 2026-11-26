#!/bin/bash
# SOL GPU job: evolve skills on Logical Deduction with one model, then check the
# evolved library on the Logical Deduction TEST split (in-domain check only).
#
#   mkdir -p logs
#   sbatch scripts/evolve_sol.sbatch Qwen/Qwen2.5-3B-Instruct
#
# Outputs (named per model, so nothing is overwritten):
#   skills/library_<model>.json         the evolved skill library
#   skills/evolution_log_<model>.jsonl  one record per round
#   logs/evolve_<jobid>.out / .err      printed progress and any errors

#SBATCH --job-name=evolve
#SBATCH --output=logs/evolve_%j.out
#SBATCH --error=logs/evolve_%j.err
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -t 0-3
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --export=NONE

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
SLUG=$(echo "$MODEL" | tr '/' '_')

module load mamba/latest
source activate skills

export HF_HOME="/scratch/$USER/hf_cache"
export HF_HUB_DISABLE_XET=1
export BBH_DIR="$PWD/external/Big-Bench-Hard/bbh"
export LLM_BACKEND=hf
export HF_MODEL="$MODEL"        # run_skill_evolution.py calls get_llm() and reads this
export HF_BATCH_SIZE=16
export MAX_NEW_TOKENS=1536

python scripts/check_gpu.py || exit 1
python experiments/run_skill_evolution.py --eval-test \
    --out "skills/library_${SLUG}.json" \
    --log "skills/evolution_log_${SLUG}.jsonl"
