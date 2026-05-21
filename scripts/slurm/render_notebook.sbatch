#!/usr/bin/env bash
#SBATCH --job-name=spider-et-nb
#SBATCH --partition=maxcpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=data/spider_et_variation_fit_cache/logs/notebook_%j.out
#SBATCH --error=data/spider_et_variation_fit_cache/logs/notebook_%j.err

set -euo pipefail

cd "${SPIDER_ET_REPO:?SPIDER_ET_REPO is required}"

export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYDEVD_DISABLE_FILE_VALIDATION=1

"${JUPYTER_EXE:-jupyter}" nbconvert \
  --to notebook \
  --execute "5_Analysis_Boostfactor_Determination_Spider_ET.ipynb" \
  --output "5_Analysis_Boostfactor_Determination_Spider_ET_executed.ipynb" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=julia-1.12-_spider-bead-pull_-1.12
