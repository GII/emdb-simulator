#!/bin/bash
#SBATCH --job-name=emdb_train_baseline
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium

set -euo pipefail
VARIANT=cpu
HOST_WORKSPACE="${HOST_WORKSPACE:-$HOME/emdb_simulator_develop}"
source "$HOST_WORKSPACE/hpc/cesga/_common.sh"

ALGO="${ALGO:-sac}"
TIMESTEPS="${TIMESTEPS:-300000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
SAC_ENT_COEF="${SAC_ENT_COEF:-0.1}"
ENT_COEF="${ENT_COEF:-0.0}"
SEED="${SEED:-$TASK_ID}"

singularity exec "${BIND_ARGS[@]}" "$SIF" /opt/emdb/entrypoint.sh bash -c "
    set -euo pipefail

    ros2 launch emdb_simulator emdb_simulator.launch.py \
        teleop:=false headless:=true &
    SIM_PID=\$!
    trap 'kill \$SIM_PID 2>/dev/null || true' EXIT

    echo 'Waiting for /step_action to come up...'
    tries=300
    until ros2 service list 2>/dev/null | grep -q '^/step_action\$' || [ \"\$tries\" -le 0 ]; do
        sleep 1
        tries=\$((tries - 1))
    done
    if [ \"\$tries\" -le 0 ]; then
        echo 'ERROR: /step_action never came up after 300s' >&2
        exit 1
    fi

    ros2 run emdb_policy train_sb3 \
        --algo '$ALGO' \
        --timesteps '$TIMESTEPS' \
        --max-episode-steps '$MAX_EPISODE_STEPS' \
        --sac-ent-coef '$SAC_ENT_COEF' \
        --ent-coef '$ENT_COEF' \
        --seed '$SEED' \
        --save-path '$OUT_DIR/model'
"

echo "Done -- results in $OUT_DIR (model.zip, monitor.csv, TensorBoard logs)"
