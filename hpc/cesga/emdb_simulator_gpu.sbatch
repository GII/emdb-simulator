#!/bin/bash
#SBATCH --job-name=emdb_simulator_gpu
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium
#SBATCH --gres=gpu:a100:1

set -euo pipefail
VARIANT=gpu
HOST_WORKSPACE="${HOST_WORKSPACE:-$HOME/emdb_simulator_develop}"
source "$HOST_WORKSPACE/hpc/cesga/_common.sh"

singularity exec --nv "${BIND_ARGS[@]}" "$SIF" /opt/emdb/entrypoint.sh \
    ros2 launch emdb_simulator emdb_simulator.launch.py \
        layout_id:="$((11 + TASK_ID % 50))" \
        style_id:="$((TASK_ID % 11))" \
        headless:=true \
        record_video:=true \
        record_video_dir:="$OUT_DIR/videos"
