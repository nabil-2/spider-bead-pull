#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIDER_ET_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

JULIA_EXE="${JULIA_EXE:-/software/julia/julia-1.12.6/bin/julia}"
JUPYTER_EXE="${JUPYTER_EXE:-jupyter}"
SPIDER_ET_CACHE_DIR="${SPIDER_ET_CACHE_DIR:-data/spider_et_variation_fit_cache}"
SPIDER_ET_DATA_FILE="${SPIDER_ET_DATA_FILE:-data/ET_results_variations.npz}"
SPIDER_ET_RESULTS_FILE="${SPIDER_ET_RESULTS_FILE:-data/5_result_bf_determination_spider_ET_variations.jld2}"
SPIDER_ET_N_MC="${SPIDER_ET_N_MC:-100}"
SPIDER_ET_SEED="${SPIDER_ET_SEED:-904230}"
SPIDER_ET_CONSTRAINT="${SPIDER_ET_CONSTRAINT:-EPYC&9554}"

mkdir -p "${SPIDER_ET_REPO}/${SPIDER_ET_CACHE_DIR}/logs"

export SPIDER_ET_REPO
export JULIA_EXE
export JUPYTER_EXE
export SPIDER_ET_CACHE_DIR
export SPIDER_ET_DATA_FILE
export SPIDER_ET_RESULTS_FILE
export SPIDER_ET_N_MC
export SPIDER_ET_SEED
export SPIDER_ET_FORCE="${SPIDER_ET_FORCE:-}"

export_arg="ALL,SPIDER_ET_REPO,JULIA_EXE,JUPYTER_EXE,SPIDER_ET_CACHE_DIR,SPIDER_ET_DATA_FILE,SPIDER_ET_RESULTS_FILE,SPIDER_ET_N_MC,SPIDER_ET_SEED,SPIDER_ET_FORCE"

echo "Submitting Spider ET production jobs from ${SPIDER_ET_REPO}"
echo "Constraint: ${SPIDER_ET_CONSTRAINT}"
echo "Cache dir:  ${SPIDER_ET_CACHE_DIR}"
echo "N_mc:       ${SPIDER_ET_N_MC}"

full_job="$(
  sbatch --parsable \
    --chdir="${SPIDER_ET_REPO}" \
    --constraint="${SPIDER_ET_CONSTRAINT}" \
    --export="${export_arg}" \
    "${SCRIPT_DIR}/full_reference.sbatch"
)"

config_job="$(
  sbatch --parsable \
    --chdir="${SPIDER_ET_REPO}" \
    --constraint="${SPIDER_ET_CONSTRAINT}" \
    --export="${export_arg}" \
    "${SCRIPT_DIR}/config_array.sbatch"
)"

assemble_job="$(
  sbatch --parsable \
    --chdir="${SPIDER_ET_REPO}" \
    --dependency="afterok:${full_job}:${config_job}" \
    --export="${export_arg}" \
    "${SCRIPT_DIR}/assemble.sbatch"
)"

render_job="$(
  sbatch --parsable \
    --chdir="${SPIDER_ET_REPO}" \
    --dependency="afterok:${assemble_job}" \
    --export="${export_arg}" \
    "${SCRIPT_DIR}/render_notebook.sbatch"
)"

job_file="${SPIDER_ET_REPO}/${SPIDER_ET_CACHE_DIR}/production_jobs.env"
cat > "${job_file}" <<EOF
FULL_JOB_ID=${full_job}
CONFIG_ARRAY_JOB_ID=${config_job}
ASSEMBLE_JOB_ID=${assemble_job}
RENDER_JOB_ID=${render_job}
SPIDER_ET_CONSTRAINT=${SPIDER_ET_CONSTRAINT}
SPIDER_ET_N_MC=${SPIDER_ET_N_MC}
SUBMITTED_AT=$(date --iso-8601=seconds)
EOF

echo "Submitted jobs:"
echo "  full:     ${full_job}"
echo "  configs:  ${config_job}"
echo "  assemble: ${assemble_job}"
echo "  render:   ${render_job}"
echo "Wrote ${job_file}"
echo "Monitor with: squeue -j ${full_job},${config_job},${assemble_job},${render_job}"
echo "If the first two jobs stay pending too long, cancel these ids and rerun with SPIDER_ET_CONSTRAINT='EPYC&9534' or 'EPYC&75F3'."
