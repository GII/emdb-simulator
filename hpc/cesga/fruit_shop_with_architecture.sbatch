#!/bin/bash
# Runs a full FruitShop e-MDB experiment on CESGA: this repo's simulator
# (scene_loader + fruit_shop_bridge) alongside the separate, pre-existing
# e-MDB cognitive architecture container (santacrc/emdb_cesga_cpu/_gpu --
# commander + ltm; built and published independently, not by this repo)
# on the SAME compute node, so their ROS 2 graphs discover each other over
# the node's shared network namespace (same mechanism
# emdb_with_architecture.sbatch uses for its connectivity smoke test).
#
# Unlike emdb_with_architecture.sbatch (a 15-minute smoke test for a
# different, not-yet-bridged "lift" domain), this drives a real,
# long-running experiment: FruitShop already has its own bridge node
# (ros_packages/src/emdb_policy/emdb_policy/fruit_shop_bridge.py)
# translating cognitive_node_interfaces Policy/WorldReset calls into this
# repo's /step_action, /reset_episode services. That file's own module
# docstring is the ground truth for the manual 5-terminal sequence this
# script automates (scene_loader; commander; ltm; a one-shot
# commander/load_config call; fruit_shop_bridge) -- read it before
# changing anything here.
#
#   sbatch hpc/cesga/fruit_shop_with_architecture.sbatch
#   sbatch --array=0-4 hpc/cesga/fruit_shop_with_architecture.sbatch   # 5 parallel seeds
#
# NOTE: this script is untested on the actual cluster (no CESGA access
# from this dev environment) -- it was written by directly mirroring the
# already-proven emdb_with_architecture.sbatch/train_baseline_cpu.sbatch
# patterns and fruit_shop_bridge.py's own documented manual sequence, not
# verified end-to-end on FT3. The biggest unverified assumption: that the
# architecture image's Python venv already satisfies fruit_shop_bridge's
# own pip dependencies (numpy, pyyaml, yamlloader) once its `install/`
# is sourced from inside the SIMULATOR container (see step 4 below) --
# if fruit_shop_bridge fails on import with a ModuleNotFoundError, that's
# the first thing to check.
#
#SBATCH --job-name=fruit_shop_with_architecture
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=medium
# ^ `medium`'s 3-day limit, matching emdb_simulator_cpu.sbatch/
#   train_baseline_cpu.sbatch -- this is a real, long-running experiment
#   (the cognitive architecture runs indefinitely by design), not a short
#   connectivity check like emdb_with_architecture.sbatch's `short`.

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

# The architecture's own image -- pre-existing, unrelated .sif naming/pull
# workflow (docs/source/howto/running_on_cesga.md's "Pulling the image"
# section covers pulling both this and the simulator's image).
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
# Base commander config, loaded once via the one-shot service call in step
# 3 below (fruit_shop_bridge.py's docstring, terminal 4) -- distinct from
# mdb_experiments/fruit_shop_experiment.yaml, which fruit_shop_bridge
# loads itself via commander/load_experiment once it starts (terminal 5).
# Path is relative to $ARCH_HOST_WORKSPACE (a colcon workspace root, same
# assumption emdb_with_architecture.sbatch already makes for
# install/setup.bash); override if your checkout's layout differs.
COMMANDER_CONFIG="${COMMANDER_CONFIG:-/workspace/emdb_develop/src/emdb_develop/emdb_core/core/config/commander.yaml}"

# mdb_experiments/ isn't part of _common.sh's BIND_ARGS (only ros_packages/
# and the assets are, by design -- see that file's own comment) but
# fruit_shop_bridge needs the experiment yaml live-editable same as any
# ros_packages source change, so bind it here too rather than relying on
# the image's COPY-time (potentially stale) copy.
BIND_ARGS+=( --bind "$HOST_WORKSPACE/mdb_experiments:/opt/emdb/mdb_experiments" )
CONFIG_FILE="/opt/emdb/mdb_experiments/fruit_shop_experiment.yaml"

# Video output: keep_successes only (no fixed record_video_episodes range)
# by default -- the entire point of this run is to capture the FruitShop
# episodes that actually succeed, not a fixed debug range. Override
# RECORD_VIDEO_EPISODES to also force-record a fixed range regardless of
# outcome (see docs/source/howto/recording_video.md). Default is a
# literal empty-string YAML scalar (`''`, not just an empty shell value)
# -- `ros2 run ... --ros-args -p name:=` with nothing after `:=` parses
# as YAML null, not "", and this parameter is declared as a plain string
# on the node side (scene_loader.py's declare_parameter default "all"),
# so null would raise a parameter-type error instead of matching
# EpisodeRecordingSpec.parse()'s documented "blank spec -> no episodes"
# fallback.
RECORD_VIDEO_EPISODES="${RECORD_VIDEO_EPISODES:-''}"
RECORD_VIDEO_CAMERA="${RECORD_VIDEO_CAMERA:-overhead_view}"
CUSTOM_CAMERAS_FILE="${CUSTOM_CAMERAS_FILE:-/opt/emdb/ros_packages/src/emdb_simulator/config/cameras/example_custom_cameras.yaml}"

# Shared ROS 2 DDS domain so all four processes' graphs are mutually
# visible, same mechanism/rationale as emdb_with_architecture.sbatch.
export ROS_DOMAIN_ID=$(( 10 + TASK_ID ))
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

PIDS=()
cleanup() {
    # Guard against an empty array: expanding "${PIDS[@]}" under `set -u`
    # (inherited from _common.sh) raises "unbound variable" on bash < 4.4
    # when PIDS has no elements yet (e.g. this trap firing before the
    # first process is even backgrounded).
    [ "${#PIDS[@]}" -gt 0 ] && kill "${PIDS[@]}" 2>/dev/null
    true
}
trap cleanup EXIT

# --- 1. Simulator side: scene_loader (this repo's image) ----------------
# control_mode:=rl (fruit_shop_bridge drives it through /step_action, not
# a human teleop device); perception_mode:=mdb publishes the e-MDB-
# compatible perception topics fruit_shop_bridge/the architecture read.
singularity exec "${BIND_ARGS[@]}" "$SIF" /opt/emdb/entrypoint.sh \
    ros2 run emdb_simulator scene_loader --ros-args \
        -p control_mode:=rl -p task:=FruitShop -p perception_mode:=mdb \
        -p record_video:=true \
        -p record_video_episodes:="$RECORD_VIDEO_EPISODES" \
        -p record_video_keep_successes:=true \
        -p record_video_camera:="$RECORD_VIDEO_CAMERA" \
        -p custom_cameras_file:="$CUSTOM_CAMERAS_FILE" \
        -p record_video_dir:="$OUT_DIR/videos" \
    > "$OUT_DIR/scene_loader.log" 2>&1 &
PIDS+=("$!")

# --- 2. Architecture side: ltm + commander --------------------------------
# Both backgrounded in one singularity exec/bash -c (own `wait` inside so
# the outer PID tracks this whole sub-shell); matches
# emdb_with_architecture.sbatch's pattern of one singularity exec per
# image, not per process, to avoid extra full-extraction overhead (no
# FUSE on this cluster -- see docs/source/howto/running_on_cesga.md).
#
# Also needs mdb_experiments/ bound in: commander (started here) is what
# actually calls open() on CONFIG_FILE when it services fruit_shop_bridge's
# commander/load_experiment request in step 4 -- the path travels over a
# ROS 2 service call, but the file read happens in whichever process's
# filesystem namespace handles that request, i.e. THIS container, not the
# simulator one. Binding it only into BIND_ARGS (used by the simulator's
# singularity exec calls, step 1/4) left commander unable to see it,
# producing "Couldn't load experiment. File ... not found" even though the
# file exists on the host and in the simulator container.
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
# ^ each `source` is its own statement (not chained with && into the
#   following `&`) so both persist in this bash -c's own shell before
#   either `ros2 run` backgrounds -- `cmd1 && cmd2 &` would instead
#   background the WHOLE chain as one subshell, and the sourcing inside
#   it would never reach the sibling `ros2 run core commander &`
#   statement in the parent shell.
PIDS+=("$!")

# Let scene_loader (RoboCasa asset/fixture init) and commander/ltm come up
# before the one-shot config-load below -- 90s matches
# emdb_with_architecture.sbatch's own proven budget for this cluster's
# no-FUSE full-extraction startup cost; polling via repeated `singularity
# exec` calls instead would be far slower for the same reason (each
# invocation re-extracts the whole image, not just the first one).
sleep 90

# --- 3. One-shot: load the base commander config -------------------------
# Must happen after commander is up but before fruit_shop_bridge's own
# load_experiment_file_in_commander() (called automatically from its
# main(), step 4 below) tries to build the experiment's LTM graph on top
# of it -- matches fruit_shop_bridge.py's docstring, terminal 4. Issued
# from the architecture image since core_interfaces (this service's own
# message package) is guaranteed to be there; DDS discovery doesn't care
# which container actually places the call.
singularity exec \
    --bind "$ARCH_HOST_WORKSPACE:/workspace/emdb_develop:ro" \
    "$ARCH_SIF" bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /workspace/emdb_develop/install/setup.bash && \
     ros2 service call commander/load_config core_interfaces/srv/LoadConfig \
        \"{file: '$COMMANDER_CONFIG'}\"" \
    > "$OUT_DIR/load_config.log" 2>&1

# --- 4. Simulator side: fruit_shop_bridge ---------------------------------
# Needs BOTH this repo's install (emdb_interfaces/emdb_policy, already
# sourced by entrypoint.sh) AND the architecture's install
# (cognitive_node_interfaces/cognitive_processes_interfaces/core/
# core_interfaces/simulators_interfaces, bind-mounted read-only and
# sourced here) in the same shell -- exactly mirroring the two `source`
# lines in fruit_shop_bridge.py's own docstring's terminal 5.
# random_seed:="$TASK_ID" gives independent seeds under --array, matching
# train_baseline_cpu.sbatch's SEED pattern.
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

# Wait on fruit_shop_bridge specifically (the actual experiment driver) --
# if it dies (e.g. a bad procedurally-generated layout it can't recover
# from, see fruit_shop_bridge.py's executed_policy_callback error-recovery
# comment for what it DOES tolerate), end the job instead of silently
# burning the rest of the time budget with the other three processes
# still up but nothing actually driving policies.
wait "$BRIDGE_PID"
