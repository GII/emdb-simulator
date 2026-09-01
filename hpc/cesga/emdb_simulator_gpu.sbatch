#!/bin/bash
# GPU SLURM job for the emdb-simulator, submitted from a CESGA login node:
#   sbatch hpc/cesga/emdb_simulator_gpu.sbatch
#   sbatch --array=0-4 hpc/cesga/emdb_simulator_gpu.sbatch   # 5 parallel variations
#
# See docs/source/howto/running_on_cesga.md for the full setup this depends
# on (image pull, one-time kitchen-asset priming, HOST_WORKSPACE layout).
#SBATCH --job-name=emdb_simulator_gpu
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium
#SBATCH --gres=gpu:a100:1
# ^ Per `sinfo`, FT3's GPUs (a100 nodes) are scheduled via --gres within
#   the same shared partitions as CPU jobs, not a dedicated GPU partition.
#   `medium`'s 3-day limit comfortably covers this job's 48h --time (the
#   `short` default only allows 6h). Drop to `short` (shorter --time) for
#   quick smoke tests, or `long`/`ondemand` for multi-day runs.

set -euo pipefail
# ^ Set here too, not just inside _common.sh: if the source below fails,
# this makes the job abort immediately with bash's own clear "No such
# file or directory" instead of silently continuing with STORE/SIF/
# HOST_WORKSPACE/BIND_ARGS/OUT_DIR all unset -- which previously cascaded
# into confusing "Permission denied" errors from every "$OUT_DIR/foo.log"
# redirect collapsing to "/foo.log".
VARIANT=gpu
# Locate _common.sh via the same HOST_WORKSPACE default/override
# _common.sh itself uses for everything else (not $SLURM_SUBMIT_DIR) so
# `sbatch` can be invoked from any cwd -- not just from inside the repo
# checkout -- as long as HOST_WORKSPACE (env var, or this default) points
# at it.
HOST_WORKSPACE="${HOST_WORKSPACE:-$HOME/emdb_simulator_develop}"
source "$HOST_WORKSPACE/hpc/cesga/_common.sh"

# --nv exposes the host's NVIDIA driver/EGL libs into the container (see
# docker/Dockerfile.gpu's header comment for why those aren't baked into
# the image itself). Everything else mirrors emdb_simulator_cpu.sbatch --
# see that file's comments for the entrypoint.sh/xvfb-run rationale.
singularity exec --nv "${BIND_ARGS[@]}" "$SIF" /opt/emdb/entrypoint.sh \
    ros2 launch emdb_simulator emdb_simulator.launch.py \
        layout_id:="$((11 + TASK_ID % 50))" \
        style_id:="$((TASK_ID % 11))" \
        record_video:=true \
        record_video_dir:="$OUT_DIR/videos"
