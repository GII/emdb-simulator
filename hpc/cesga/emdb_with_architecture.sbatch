#!/bin/bash
#SBATCH --job-name=emdb_with_architecture
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

set -euo pipefail
VARIANT=cpu
HOST_WORKSPACE="${HOST_WORKSPACE:-$HOME/emdb_simulator_develop}"
source "$HOST_WORKSPACE/hpc/cesga/_common.sh"

ARCH_SIF="${ARCH_SIF:-$STORE/emdb_${VARIANT}.sif}"
if [ ! -f "$ARCH_SIF" ]; then
    echo "ERROR: $ARCH_SIF not found -- pull the e-MDB architecture image" \
         "first (singularity pull \$STORE/emdb_${VARIANT}.sif" \
         "docker://santacrc/emdb_cesga_${VARIANT})." >&2
    exit 1
fi

export ROS_DOMAIN_ID=$(( 10 + TASK_ID ))
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

cleanup() {
    kill "${SIM_PID:-}" "${ARCH_PID:-}" 2>/dev/null
}
trap cleanup EXIT

singularity exec "${BIND_ARGS[@]}" \
    "$SIF" /opt/emdb/entrypoint.sh \
    ros2 launch emdb_simulator emdb_simulator.launch.py \
        perception_mode:=mdb \
        headless:=true \
        record_video:=true \
        record_video_dir:="$OUT_DIR/videos" &
SIM_PID=$!

ARCH_HOST_WORKSPACE="${ARCH_HOST_WORKSPACE:-$HOME/emdb_develop}"
if [ ! -f "$ARCH_HOST_WORKSPACE/install/setup.bash" ]; then
    echo "ERROR: $ARCH_HOST_WORKSPACE/install/setup.bash not found -- build" \
         "the architecture workspace first (colcon build), or set" \
         "ARCH_HOST_WORKSPACE to point at one that's already built." >&2
    exit 1
fi
ARCH_PACKAGE="${ARCH_PACKAGE:-experiments}"
ARCH_LAUNCH_FILE="${ARCH_LAUNCH_FILE:-sim2D_launch.py}"
singularity exec \
    --bind "$ARCH_HOST_WORKSPACE:/workspace/emdb_develop:ro" \
    "$ARCH_SIF" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /workspace/emdb_develop/install/setup.bash && \
     ros2 launch $ARCH_PACKAGE $ARCH_LAUNCH_FILE" &
ARCH_PID=$!

sleep 90

singularity exec "$ARCH_SIF" \
    bash -c "source /opt/ros/humble/setup.bash && \
             ros2 node list; echo ---; ros2 topic list | grep /emdb/simulator" \
    > "$OUT_DIR/plumbing_check.log" 2>&1

echo "Plumbing check written to $OUT_DIR/plumbing_check.log"
