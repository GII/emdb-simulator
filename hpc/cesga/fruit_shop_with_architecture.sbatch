#!/bin/bash
#SBATCH --job-name=fruit_shop_with_architecture
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

ARCH_SIF="${ARCH_SIF:-$STORE/emdb_${VARIANT}.sif}"
if [ ! -f "$ARCH_SIF" ]; then
    echo "ERROR: $ARCH_SIF not found -- pull the e-MDB architecture image" \
         "first (singularity pull \$STORE/emdb_${VARIANT}.sif" \
         "docker://santacrc/emdb_cesga_${VARIANT})." >&2
    exit 1
fi

ARCH_HOST_WORKSPACE="${ARCH_HOST_WORKSPACE:-$HOME/emdb_develop}"
if [ ! -f "$ARCH_HOST_WORKSPACE/install/setup.bash" ]; then
    echo "ERROR: $ARCH_HOST_WORKSPACE/install/setup.bash not found -- build" \
         "the architecture workspace first (colcon build), or set" \
         "ARCH_HOST_WORKSPACE to point at one that's already built (see" \
         "docs/source/howto/running_on_cesga.md, 'Running alongside the" \
         "e-MDB architecture')." >&2
    exit 1
fi
COMMANDER_CONFIG="${COMMANDER_CONFIG:-/workspace/emdb_develop/src/emdb_develop/emdb_core/core/config/commander.yaml}"

BIND_ARGS+=( --bind "$HOST_WORKSPACE/mdb_experiments:/opt/emdb/mdb_experiments" )
CONFIG_FILE="/opt/emdb/mdb_experiments/fruit_shop_experiment.yaml"

RECORD_VIDEO_EPISODES="${RECORD_VIDEO_EPISODES:-''}"
RECORD_VIDEO_CAMERA="${RECORD_VIDEO_CAMERA:-overhead_view}"
CUSTOM_CAMERAS_FILE="${CUSTOM_CAMERAS_FILE:-/opt/emdb/ros_packages/src/emdb_simulator/config/cameras/example_custom_cameras.yaml}"

export ROS_DOMAIN_ID=$(( 10 + TASK_ID ))
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

PIDS=()
cleanup() {
    [ "${#PIDS[@]}" -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null
    true
}
trap cleanup EXIT

singularity exec "${BIND_ARGS[@]}" "$SIF" /opt/emdb/entrypoint.sh \
    ros2 run emdb_simulator scene_loader --ros-args \
        -p control_mode:=rl -p task:=FruitShop -p perception_mode:=mdb \
        -p headless:=true \
        -p record_video:=true \
        -p record_video_episodes:="$RECORD_VIDEO_EPISODES" \
        -p record_video_keep_successes:=true \
        -p record_video_camera:="$RECORD_VIDEO_CAMERA" \
        -p custom_cameras_file:="$CUSTOM_CAMERAS_FILE" \
        -p record_video_dir:="$OUT_DIR/videos" \
    > "$OUT_DIR/scene_loader.log" 2>&1 &
PIDS+=("$!")

singularity exec \
    --bind "$ARCH_HOST_WORKSPACE:/workspace/emdb_develop:ro" \
    --bind "$HOST_WORKSPACE/mdb_experiments:/opt/emdb/mdb_experiments" \
    "$ARCH_SIF" bash -c "
        set -e
        source /opt/ros/humble/setup.bash
        source /workspace/emdb_develop/install/setup.bash
        ros2 run core ltm 0 &
        ros2 run core commander &
        wait
    " > "$OUT_DIR/architecture.log" 2>&1 &
PIDS+=("$!")

sleep 90

singularity exec \
    --bind "$ARCH_HOST_WORKSPACE:/workspace/emdb_develop:ro" \
    "$ARCH_SIF" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /workspace/emdb_develop/install/setup.bash && \
     ros2 service call commander/load_config core_interfaces/srv/LoadConfig \
        \"{file: '$COMMANDER_CONFIG'}\"" \
    > "$OUT_DIR/load_config.log" 2>&1

singularity exec "${BIND_ARGS[@]}" \
    --bind "$ARCH_HOST_WORKSPACE:/workspace/emdb_develop:ro" \
    "$SIF" /opt/emdb/entrypoint.sh bash -c "
        source /workspace/emdb_develop/install/setup.bash && \
        exec ros2 run emdb_policy fruit_shop_bridge --ros-args \
            -p config_file:='$CONFIG_FILE' -p random_seed:='$TASK_ID'
    " > "$OUT_DIR/fruit_shop_bridge.log" 2>&1 &
BRIDGE_PID=$!
PIDS+=("$BRIDGE_PID")

echo "FruitShop experiment running -- videos/logs -> $OUT_DIR"
echo "This runs for the job's full --time budget (the cognitive" \
     "architecture is long-running by design, same as" \
     "emdb_with_architecture.sbatch's architecture side) -- SLURM's time" \
     "limit or a manual scancel ends it, not a natural exit."

wait "$BRIDGE_PID"
