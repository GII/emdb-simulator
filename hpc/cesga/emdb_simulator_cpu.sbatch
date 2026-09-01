#!/bin/bash
# CPU SLURM job for the emdb-simulator, submitted from a CESGA login node:
#   sbatch hpc/cesga/emdb_simulator_cpu.sbatch
#   sbatch --array=0-4 hpc/cesga/emdb_simulator_cpu.sbatch   # 5 parallel variations
#
# See docs/source/howto/running_on_cesga.md for the full setup this depends
# on (image pull, one-time kitchen-asset priming, HOST_WORKSPACE layout).
#SBATCH --job-name=emdb_simulator_cpu
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium
# ^ `medium`'s 3-day limit comfortably covers this job's 48h --time (per
#   `sinfo`, FT3's `short` default only allows 6h). Drop to `short` (and a
#   shorter --time) for quick smoke tests, or `long`/`ondemand` for
#   multi-day runs beyond medium's 3-day cap.

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

# `entrypoint.sh` is invoked explicitly rather than relying on Singularity
# to honor the image's Docker ENTRYPOINT (`singularity exec` does not
# reliably do so, unlike `docker run`/`singularity run`, and getting this
# wrong would silently drop the xvfb-run wrap -- see docker/entrypoint.sh
# and docs/source/howto/recording_video.md, section 4, for why that's
# needed even in headless `rl` mode).
#
# layout_id/style_id/record_video_dir vary per SLURM array task via
# $TASK_ID (set in _common.sh); adjust freely for your actual experiment.
singularity exec "${BIND_ARGS[@]}" "$SIF" /opt/emdb/entrypoint.sh \
    ros2 launch emdb_simulator emdb_simulator.launch.py \
        layout_id:="$((11 + TASK_ID % 50))" \
        style_id:="$((TASK_ID % 11))" \
        record_video:=true \
        record_video_dir:="$OUT_DIR/videos"
