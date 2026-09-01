#!/bin/bash
# Trains the PPO/SAC "black box" RL baseline (emdb_policy's train_sb3,
# talking to emdb_simulator purely over its ROS interface, same as the
# bare-metal workflow in docs/source/howto/training_rl.md) as a single
# SLURM job: launches the simulator in the background, waits for it to be
# ready, runs train_sb3 in the foreground, then tears the simulator down.
#
#   sbatch hpc/cesga/train_baseline_cpu.sbatch
#   sbatch --array=0-4 hpc/cesga/train_baseline_cpu.sbatch   # 5 seeds in parallel
#
# CPU only, deliberately -- env.step() here is MuJoCo physics (single-
# threaded per env), not GPU-accelerable in this codebase (robosuite/
# robocasa use the classic mujoco bindings, not MJX/Warp); a GPU node would
# be wasted on this job. See docs/source/howto/running_on_cesga.md for the
# image/asset/workspace setup this depends on.
#
# Override any of these on the command line, e.g.:
#   TIMESTEPS=1000000 sbatch hpc/cesga/train_baseline_cpu.sbatch
#SBATCH --job-name=emdb_train_baseline
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium
# ^ `medium`'s 3-day limit gives headroom over --time for a long sparse-
#   reward run; drop both for a quick smoke test.

set -euo pipefail
# ^ Set here too, not just inside _common.sh: if the source below fails,
# this makes the job abort immediately with bash's own clear "No such
# file or directory" instead of silently continuing with STORE/SIF/
# HOST_WORKSPACE/BIND_ARGS/OUT_DIR all unset -- which previously cascaded
# into confusing "Permission denied" errors from every "$OUT_DIR/foo.log"
# redirect collapsing to "/foo.log".
VARIANT=cpu
# Locate _common.sh via the same HOST_WORKSPACE default/override
# _common.sh itself uses for everything else (not $SLURM_SUBMIT_DIR) so
# `sbatch` can be invoked from any cwd -- not just from inside the repo
# checkout -- as long as HOST_WORKSPACE (env var, or this default) points
# at it.
HOST_WORKSPACE="${HOST_WORKSPACE:-$HOME/emdb_simulator_develop}"
source "$HOST_WORKSPACE/hpc/cesga/_common.sh"

# --algo sac (off-policy, replay buffer) tends to make better use of rare
# sparse successes than ppo here -- see docs/source/howto/training_rl.md.
# --sac-ent-coef fixed (not "auto"): auto-tuning decays toward ~0 with no
# success yet to justify exploring, killing exploration before finding one.
ALGO="${ALGO:-sac}"
TIMESTEPS="${TIMESTEPS:-300000}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
SAC_ENT_COEF="${SAC_ENT_COEF:-0.1}"
ENT_COEF="${ENT_COEF:-0.0}"
# One seed per array task by default, so `--array=0-4` gives 5 independent
# runs (useful for mean+variance across seeds, sparse-reward RL is high
# variance) instead of 5 copies of the same one; override SEED to pin it.
SEED="${SEED:-$TASK_ID}"

# Everything below runs inside the container via entrypoint.sh (ROS 2 +
# venv + colcon overlay + Xvfb already set up by the time this executes).
# $SIM_PID/$tries are inner-shell-only (escaped, resolved by the container's
# bash); the rest are this outer script's own variables, expanded here.
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
