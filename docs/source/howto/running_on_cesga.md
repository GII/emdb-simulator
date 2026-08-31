# Running on CESGA (FinisTerraeIII)

This guide covers running the simulator on [CESGA](https://www.cesga.es/)'s
FinisTerraeIII HPC cluster via Docker/Singularity images and SLURM, instead
of the bare-metal `setup.sh`/`env.sh` workflow described elsewhere in these
docs. It assumes you already have an active CESGA account and can SSH in
(`ft3.cesga.es`); see CESGA's own
[FT3 user guide](https://cesga-docs.gitlab.io/ft3-user-guide/first_steps.html)
for account/connection issues (e.g. the "Corrupted MAC on input" error some
Windows SSH clients hit).

```{important}
Never run experiments on a login node -- they're for quick checks only.
Everything that actually simulates goes through SLURM (`sbatch`), as
described below.
```

```{note}
Everything here runs **fully headless** -- CESGA compute nodes have no
display. `Xvfb` (X *virtual* framebuffer, no real display attached) is
what makes `scene_loader` importable at all (`pynput.keyboard` needs *some*
X connection even when nothing listens for keys, see
{doc}`recording_video`, section 4), not a step toward an actual GUI. Debug
visibility comes from `record_video:=true` rendering mp4s to disk instead
(already wired into `hpc/cesga/emdb_with_architecture.sbatch`;
{doc}`run_simulator` for the equivalent flags on the standalone
`emdb_simulator_{cpu,gpu}.sbatch` scripts) -- nothing here ever needs or
attempts to open a real window.
```

## 1. Storage tiers

| Directory | Use | Speed | Space limit | Files limit | Backup | Snapshot |
|---|---|---|---|---|---|---|
| `$HOME` | Source code | Low | 10 GB | 100,000 | Yes (daily, 2 months) | Yes (7 days) |
| `$STORE` | Final results | Low | 500 GB | 300,000 | No | Yes (3 days) |
| `$LUSTRE` | Simulation execution (fast NVMe) | High | 1 TB | 200,000 | No | No |

To recover a deleted file: `cp -p .snapshot/daily.<date>/path/file $HOME/destination`.

Job-scoped scratch (all wiped when the job ends):

| Variable | Storage | Notes |
|---|---|---|
| `$LUSTRE_SCRATCH` | Shared Lustre (NVMe) | Shared between the job's nodes |
| `$LOCAL_SCRATCH` / `$TMPDIR` | Node-local disk | Private per node |
| `$TMPSHM` | Node RAM (tmpfs) | Counts against the job's memory limit; max 50% of node RAM |

## 2. Pulling the container image

```{important}
This repo publishes the **simulator's** image only
(`santacrc/emdb_simulator_cpu`/`_gpu`). The **e-MDB cognitive architecture**
(from the separate `ws_bartender` workspace) has its own, already-existing
image, `santacrc/emdb_cesga_cpu`/`_gpu` — built and published independently,
not by this repo. Running a full experiment needs both; see
{ref}`cesga-with-architecture` below. Don't confuse the two `.sif` filenames
this section produces (`emdb_simulator_cpu.sif`) with the architecture's
(`emdb_cpu.sif`, no `simulator` in the name).
```

CESGA uses Singularity, so the Docker image published by
[`.github/workflows/docker.yml`](../../../.github/workflows/docker.yml)
(`santacrc/emdb_simulator_cpu` / `santacrc/emdb_simulator_gpu` on Docker Hub)
needs to be pulled and converted to `.sif` once. Point Singularity's
cache/tmp at `$LUSTRE` rather than `$HOME` (10GB is far too small for image
pulls):

```{important}
`singularity exec` on this cluster can't use FUSE (`fusermount3: mount
failed: Operation not permitted`) and falls back to fully extracting the
image into `$SINGULARITY_TMPDIR` on **every** invocation, not just this
one-time pull -- that extraction alone is enough small files
(torch/scipy/numpy's `.venv`, ~150k+) to risk hitting `$LUSTRE`'s
200,000-file quota (a *default* quota -- `lfs quota -u $USER $LUSTRE` won't
show a nonzero limit, but it's still enforced). If you hit
`EDQUOT`/"disk quota exceeded" errors mid-extraction, clean up
`$LUSTRE/singularity-tmp/build-temp-*` leftovers from failed attempts and
free up other `$LUSTRE` usage, or ask CESGA support for a higher file quota.
```

```bash
mkdir -p "$LUSTRE/singularity-cache" "$LUSTRE/singularity-tmp"
export SINGULARITY_CACHEDIR="$LUSTRE/singularity-cache"
export SINGULARITY_TMPDIR="$LUSTRE/singularity-tmp"
```

Add those two `export` lines to `.bashrc` so they persist across sessions,
or re-export them every login.

```bash
module load singularity
singularity pull --force $STORE/emdb_simulator_cpu.sif docker://santacrc/emdb_simulator_cpu   # CPU
# singularity pull --force $STORE/emdb_simulator_gpu.sif docker://santacrc/emdb_simulator_gpu # GPU
```

Verify it:

```bash
ls -lh $STORE/emdb_simulator_cpu.sif
singularity exec $STORE/emdb_simulator_cpu.sif python --version
```

Optionally symlink into `$HOME` for convenience:

```bash
ln -s $STORE/emdb_simulator_cpu.sif $HOME/
# ln -s $STORE/emdb_simulator_gpu.sif $HOME/
```

## 3. One-time kitchen-asset priming

RoboCasa's ~10GB of kitchen assets (textures, fixtures, objects) are
**not** baked into the image -- see the
[Dockerfiles'](../../../docker/Dockerfile.cpu) header comments for why. They're
downloaded once to fast `$LUSTRE` NVMe storage and bind-mounted read-only
into every job, so many parallel `--array` tasks share a single copy
instead of each holding or re-downloading their own.

```{important}
Priming is **two steps**. A `.sif` is read-only outside bind mounts (and
Singularity's own default auto-binds like `/tmp`/`$HOME`), and
`download_kitchen_assets.py` writes each type's zip into `models/assets/`
itself (the parent of its target folder) before extracting -- so the
*whole* `models/assets` directory needs to be bind-mounted, not just its
downloaded subdirectories. But that directory also holds files baked into
the image that aren't downloaded assets (`box_links/box_links_assets.json`,
needed at import time, plus `arenas/`, `scenes/`, etc.) -- bind-mounting
over it blind would shadow those with an empty host directory. Step 1 below
seeds the host directory with those baked-in files first (**no** bind mount
active yet, so it reads the image's real copies); step 2 downloads with the
now-safe whole-directory bind mount active.
```

```bash
module load singularity
mkdir -p "$LUSTRE/emdb_assets"

# Step 1: seed with this image's baked-in (non-downloaded) files.
singularity exec "$STORE/emdb_simulator_cpu.sif" bash -c \
  "cp -r /opt/emdb/misc/robocasa/robocasa/models/assets/. $LUSTRE/emdb_assets/"

# Step 2: download the real assets into the now-seeded directory.
singularity exec \
  --bind "$LUSTRE/emdb_assets:/opt/emdb/misc/robocasa/robocasa/models/assets" \
  "$STORE/emdb_simulator_cpu.sif" bash -c \
  "yes | python -m robocasa.scripts.download_kitchen_assets --type all"
```

```{warning}
Run this **exactly once**, from the login node (it needs internet access,
which compute nodes don't have). Running it again concurrently from
multiple jobs would race and corrupt partial extracts. `hpc/cesga/_common.sh`
checks `$LUSTRE/emdb_assets/textures` exists before letting a job proceed,
and fails fast with a clear error if priming hasn't happened yet.
```

```{note}
**If a run fails with `FileNotFoundError` under
`models/assets/objects/...` or `models/assets/fixtures/...`**, one asset
type didn't fully land (a download can report success per-`--type` while
still missing some files, or a manual `mv`/`cp` between directories gets
interrupted). Check what's actually there --
`ls "$LUSTRE/emdb_assets/objects/"` should show `objaverse`, `aigen_objs`,
`lightwheel`, `README.md` -- and re-download just the missing piece rather
than redoing everything (registry names: `tex`, `tex_generative`,
`fixtures_lw`, `objs_objaverse`, `objs_aigen`, `objs_lw`; `--type all` for
everything):

```bash
singularity exec \
  --bind "$LUSTRE/emdb_assets:/opt/emdb/misc/robocasa/robocasa/models/assets" \
  "$STORE/emdb_simulator_cpu.sif" bash -c \
  "yes | python -m robocasa.scripts.download_kitchen_assets --type objs_aigen objs_lw"
```
```

## 4. Getting code onto the cluster

Clone or `rsync` the repository to `$HOME/emdb_simulator_develop` (the default
`HOST_WORKSPACE` the sbatch scripts expect):

```bash
git clone <repo-url> $HOME/emdb_simulator_develop
```

```{important}
**Do NOT pass `--recurse-submodules`.** `misc/robosuite`, `misc/robocasa`,
`misc/robomimic`, `misc/mimicgen`, `misc/robosuite_models` are already
baked into the image at build time -- nothing on the CESGA side ever reads
them from a host checkout (`hpc/cesga/_common.sh`'s `BIND_ARGS` only binds
`ros_packages/` and the primed assets, never `misc/`). Recursively cloning
them anyway easily blows `$HOME`'s 10GB quota for zero benefit -- e.g.
`misc/robosuite`'s git history alone is 400+MB, on top of the other four.
```

```{important}
Only `ros_packages/` is meant to be live-edited here and bind-mounted into
jobs. **Never** bind-mount `misc/robosuite` or `misc/robocasa` over the
image -- those come from the baked-in editable install and mounting a
fresh/partial copy from `$HOME` would shadow it, silently breaking the
simulator (and losing the asset mount point from step 3, since the assets
live inside `misc/robocasa`'s own tree). Edit `ros_packages/src` freely;
[`docker/entrypoint.sh`](../../../docker/entrypoint.sh) rebuilds it
automatically if it looks newer than the last build, the first time a job
execs the container.
```

## 5. Submitting jobs

The SLURM job scripts live in `hpc/cesga/` in this repo
([`emdb_simulator_cpu.sbatch`](../../../hpc/cesga/emdb_simulator_cpu.sbatch),
[`emdb_simulator_gpu.sbatch`](../../../hpc/cesga/emdb_simulator_gpu.sbatch),
and a shared [`_common.sh`](../../../hpc/cesga/_common.sh) they both
source):

```bash
cd $HOME/emdb_simulator_develop
sbatch hpc/cesga/emdb_simulator_cpu.sbatch                  # single run
sbatch --array=0-4 hpc/cesga/emdb_simulator_cpu.sbatch      # 5 parallel variations
sbatch hpc/cesga/emdb_simulator_gpu.sbatch                  # GPU variant
```

Both scripts default `HOST_WORKSPACE` to `$HOME/emdb_simulator_develop` and the
`.sif` path to `$STORE/emdb_simulator_<variant>.sif`; override either on
the command line if you used different locations:

```bash
HOST_WORKSPACE=$HOME/some_other_checkout sbatch hpc/cesga/emdb_simulator_cpu.sbatch
```

```{note}
Both `.sbatch` files target FinisTerraeIII's `medium` partition (3-day
limit, comfortably covers the 48h `--time` both request; the `short`
default partition only allows 6h) — the GPU one requests
`--gres=gpu:a100:1`, since FT3's A100 GPUs are scheduled via `--gres`
within the same shared partitions as CPU jobs, not a dedicated GPU
partition (per `sinfo`). Adjust `--partition`/`--time`/`--gres` to taste
for shorter smoke tests or different node counts.
```

Each SLURM array task's `$SLURM_ARRAY_TASK_ID` varies `layout_id`/
`style_id` and picks a separate output directory under
`$STORE/emdb_runs/<job-name>_<job-id>/task_<id>/` -- edit the `ros2 launch`
invocation at the bottom of either `.sbatch` file to change what actually
runs (see {doc}`run_simulator` for the full parameter table).

```{note}
`--time=48:00:00` in the header is the max job runtime; adjust to taste. A
job's virtual-X/EGL teardown on `SIGINT` can report a nonzero exit code
even after shutting down cleanly at the Python level (see
{doc}`recording_video`, section 4's note) -- don't treat that alone as
proof a run failed.
```

## 6. Training the RL baseline (train_sb3)

[`hpc/cesga/train_baseline_cpu.sbatch`](../../../hpc/cesga/train_baseline_cpu.sbatch)
runs {doc}`training_rl`'s `train_sb3` as a single SLURM job: launches the
simulator in the background, waits for its `/step_action` service, runs
`train_sb3` in the foreground, tears the simulator down when it's done.
CPU only, deliberately -- `env.step()` here is single-threaded MuJoCo
physics, not GPU-accelerable in this codebase (robosuite/robocasa use the
classic `mujoco` bindings, not MJX/Warp), so a GPU node would be wasted on
this job.

```bash
sbatch hpc/cesga/train_baseline_cpu.sbatch
sbatch --array=0-4 hpc/cesga/train_baseline_cpu.sbatch   # 5 seeds in parallel
```

Every `train_sb3` CLI flag is overridable via an env var of the same name
(`ALGO`, `TIMESTEPS`, `MAX_EPISODE_STEPS`, `SAC_ENT_COEF`, `ENT_COEF`;
`SEED` defaults to the array task id, so `--array` gives independent seeds
instead of identical copies):

```bash
TIMESTEPS=1000000 ALGO=ppo sbatch hpc/cesga/train_baseline_cpu.sbatch
```

Results (`model.zip`, `monitor.csv`, TensorBoard event files) land in the
same `$STORE/emdb_runs/<job-name>_<job-id>/task_<id>/` layout as the other
`.sbatch` scripts (step 5) -- see {doc}`training_rl` for what each file
means and for the sparse-reward tuning notes (`--algo sac` vs `ppo`,
`--sac-ent-coef`) this script's defaults already bake in.

(cesga-with-architecture)=
## 7. Running alongside the e-MDB architecture

A full experiment needs the simulator running together with the actual
**e-MDB cognitive architecture** (from the separate `ws_bartender`
workspace — `emdb_core`, `emdb_cognitive_nodes_gii`,
`emdb_cognitive_processes_gii`, `emdb_discrete_event_simulator_gii`,
`emdb_experiments_gii`), published as its own, already-existing image,
`santacrc/emdb_cesga_cpu`/`_gpu` (unrelated to and unbuilt by this repo).

Pull it the same way as step 2 above, just with the architecture's own
image name and `.sif` filename (no `simulator` in either):

```bash
singularity pull --force $STORE/emdb_cpu.sif docker://santacrc/emdb_cesga_cpu
```

[`hpc/cesga/emdb_with_architecture.sbatch`](../../../hpc/cesga/emdb_with_architecture.sbatch)
launches both containers **on the same compute node** within one SLURM job:

```bash
sbatch hpc/cesga/emdb_with_architecture.sbatch
```

```{important}
Both containers run with no `--net`/network-isolation flags on their
`singularity exec` calls, so they share the node's network namespace —
this is what makes standard ROS 2 DDS multicast discovery work with zero
extra configuration, given a shared `ROS_DOMAIN_ID` (the script sets one).
This only works because both land on the **same** node; it deliberately
doesn't attempt to support the two containers running on different nodes,
which would need a real cross-node discovery mechanism (e.g. a CycloneDDS
static peer list) that hasn't been built or tested here.
```

```{important}
The architecture side needs its own **already-built** ROS workspace
(`install/setup.bash` present) bind-mounted in read-only, matching the
standalone architecture sbatch workflow's own pattern -- the image itself
doesn't bake in `experiments`/`sim2D_launch.py`, only ROS + Python deps.
Defaults to `$HOME/emdb_develop` (`ARCH_HOST_WORKSPACE` to override). The
launch command itself defaults to `ros2 launch experiments sim2D_launch.py`
(`ARCH_PACKAGE`/`ARCH_LAUNCH_FILE` to override).
```

If `$HOME/emdb_develop` (or wherever your checkout lives) doesn't have
`install/setup.bash` yet, build it once, from the login node, using the
architecture's own image (so the build uses the exact same ROS/Python
environment the job will run against):

```bash
cd $HOME/emdb_develop
singularity exec $STORE/emdb_cpu.sif bash -c \
  "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
```

```{note}
Skip `rosdep install` here -- the image has no `sudo`/root access to run
`rosdep init` first (`ERROR: no sources directory exists on the system
meaning rosdep has not yet been initialized`), and it isn't needed anyway:
this is a proven, already-working production image, so its Python/ROS
dependencies are already satisfied. Just `colcon build` directly.
```

This takes a minute or two (19 packages, mostly small `ament_python`
nodes plus a couple of message-generation ones) and only needs to happen
once -- rerun it after pulling architecture code changes, same as any
colcon workspace.

```{note}
`sim2D_launch.py`, not `bartender_launch.py` -- `bartender_launch.py` also
launches the architecture's own bundled discrete-event simulator under the
same `/emdb/simulator/sensor/...` prefix this repo's `mdb` perception mode
uses, which collides with (and drowns out) this simulator's own topics in
any visibility check. `sim2D_launch.py`'s bundled 2D reference simulator
publishes under `/mdb/baxter/sensor/...` instead, so it doesn't collide.
```

```{note}
**Scope**: this script is a connectivity smoke test, not a full experiment.
It proves both containers' ROS 2 graphs are mutually visible (writing
`ros2 node list`/`ros2 topic list` output to
`$STORE/emdb_runs/.../plumbing_check.log`) — it does not run an actual
e-MDB experiment against this simulator. That additionally needs a bridge
node translating the architecture's `cognitive_node_interfaces`
`Action`/`Policy`/`WorldReset` service calls into this repo's existing
`/step_action`/`/reset_episode` services (the role
`emdb_discrete_event_simulator_gii`'s `bartender_sim_discrete.py` plays for
the discrete-event simulator), plus new `Perception`/`Policy`/`WorldModel`
Python classes on the architecture side for the "lift" domain. Both are
explicit followups — see `mdb_experiments/lift_experiment.yaml`'s own TODO
comments for the current state of that gap. This is specific to "lift" —
FruitShop already has its own bridge node, see {ref}`cesga-fruit-shop`
below.
```

A full run takes roughly 1.5–2 minutes end to end before results are ready
(FUSE isn't available on this cluster, so both containers fall back to
fully extracting into a temp sandbox on every invocation, plus the
simulator side rebuilds `ros_packages` from scratch each run) — the script
waits 90s before checking, well inside its `--time=00:15:00` budget. Watch
it with:

```bash
squeue -u $USER
```

Once it drops out of the queue, logs are in your submission directory as
`emdb_with_architecture_<jobid>_<taskid>.out`/`.err` (SLURM splits
stdout/stderr — `ros2 launch` output and colcon build progress land in
`.out`; module-load/singularity messages in `.err`), and the check result
in `$STORE/emdb_runs/emdb_with_architecture_<jobid>/task_0/plumbing_check.log`.
A working run's log looks like this (both containers found each other; the
simulator's node and perception topic are visible from inside the
architecture container):

```text
/commander
/execution_node_1
/execution_node_2
/execution_node_3
/execution_node_4
/execution_node_5
/ltm_0
/robocasa_rollout_node
---
/emdb/simulator/sensor/progress
```

```{note}
`fusermount3: mount failed: Operation not permitted` / `WARNING: squashfuse
mount failed, falling back to extraction` on every `singularity exec`
(including the pull/priming commands above) is expected on this cluster,
not an error to chase -- see the FUSE note in step 2.
```

(cesga-fruit-shop)=
## 7a. Running the FruitShop experiment

Unlike step 7's connectivity smoke test,
[`hpc/cesga/fruit_shop_with_architecture.sbatch`](../../../hpc/cesga/fruit_shop_with_architecture.sbatch)
runs a real, long-running FruitShop experiment: FruitShop already has its
own bridge node
(`ros_packages/src/emdb_policy/emdb_policy/fruit_shop_bridge.py`)
translating the architecture's `cognitive_node_interfaces` calls into this
repo's `/step_action`/`/reset_episode` services, so no extra bridging work
is needed. The script automates that file's own module docstring's manual
5-terminal sequence (`scene_loader`; `commander`; `ltm`; a one-shot
`commander/load_config` call; `fruit_shop_bridge`) across the same
two-container layout as step 7.

```bash
sbatch hpc/cesga/fruit_shop_with_architecture.sbatch
sbatch --array=0-4 hpc/cesga/fruit_shop_with_architecture.sbatch   # 5 parallel seeds
```

It targets the `medium` partition with a 48h budget (this is a real
experiment, not a 15-minute check) and defaults to recording only
successful episodes (`record_video_keep_successes:=true`, no fixed
`record_video_episodes` range) to `$STORE/emdb_runs/.../videos/` — override
`RECORD_VIDEO_EPISODES`/`RECORD_VIDEO_CAMERA` to change that. It shares
`ARCH_SIF`/`ARCH_HOST_WORKSPACE` with step 7's script.

```{warning}
This script hasn't been run end to end on the actual cluster yet — it was
written by mirroring `emdb_with_architecture.sbatch`/
`train_baseline_cpu.sbatch`'s already-proven patterns and
`fruit_shop_bridge.py`'s documented manual sequence, not verified against
FT3 directly. The header comment flags the biggest unverified assumption
(whether the architecture image's Python environment already satisfies
`fruit_shop_bridge`'s own pip dependencies once its `install/` is sourced
inside the simulator container) — check `fruit_shop_bridge.log` in the
job's output directory first if it fails on import.
```

## 8. Downloading results

Results land under `$STORE/emdb_runs/...` (see step 5). Pull them back to
your local machine over SSH, e.g.:

```bash
rsync -avz ft3.cesga.es:'$STORE/emdb_runs/' ./emdb_runs/
```

## Next

- {doc}`run_simulator`: the full `scene_loader` parameter table used inside
  the `ros2 launch` calls above.
- {doc}`recording_video`: background on why headless runs need `xvfb-run`
  at all (handled automatically by the image's entrypoint here, but useful
  context if you're debugging outside a container).
- {doc}`training_rl`: `train_sb3`'s CLI flags and sparse-reward tuning
  notes, referenced by step 6's `train_baseline_cpu.sbatch`.
- {ref}`cesga-with-architecture`: running the simulator together with the
  separate e-MDB cognitive architecture container.
