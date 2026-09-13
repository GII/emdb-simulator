# EMDB Simulator
[![Docs](https://github.com/SantaCRC/emdb-simulator/actions/workflows/docs.yml/badge.svg)](https://github.com/SantaCRC/emdb-simulator/actions/workflows/docs.yml)
[![Docker](https://github.com/SantaCRC/emdb-simulator/actions/workflows/docker.yml/badge.svg)](https://github.com/SantaCRC/emdb-simulator/actions/workflows/docker.yml)

EMDB is a ROS 2 workspace that wraps [RoboCasa](https://github.com/SantaCRC/robocasa) /
[robosuite](https://github.com/SantaCRC/robosuite) kitchen-manipulation simulation behind a
ROS 2 topic/service interface, so that teleoperation, demo recording, and RL
training/inference can all talk to the simulator without depending on
robosuite/robocasa/MuJoCo directly.

Full documentation (architecture, how-to guides, interface/API reference)
lives under [`docs/`](docs/source/index.md) and is built with Sphinx — see
[Building this documentation](#building-the-documentation) below, or read it
already rendered on GitHub Pages once [`docs.yml`](.github/workflows/docs.yml)
has deployed it.

## About this project

This is the Trabajo Fin de Máster (Master's Thesis) of **Fabian Alvarez
Juarez**, for the **Máster en Informática Industrial y Robótica** at the
**Universidade da Coruña (UDC)**, carried out within the **Grupo Integrado de
Ingeniería (GII)**. See
[`docs/source/about.md`](docs/source/about.md) for the full project title
(in English, Castellano, and Galego) and acknowledgments, including the
computing resources provided by CESGA.

## Repository layout

```text
TFM/
├── misc/                     # git submodules + helper scripts
│   ├── robosuite/            # SantaCRC/robosuite fork
│   ├── robocasa/             # SantaCRC/robocasa fork
│   ├── robosuite_models/     # SantaCRC/robosuite_models fork
│   ├── robomimic/            # robomimic (demo/imitation-learning formats)
│   ├── mimicgen/             # mimicgen (data generation)
│   └── scripts/
├── ros_packages/
│   └── src/
│       ├── emdb_interfaces/  # custom msg/srv definitions (ament_cmake)
│       ├── emdb_simulator/   # simulator node, scene/robot/gripper loaders (ament_python)
│       └── emdb_policy/      # RL/policy node, gym wrapper, training (ament_python)
├── docker/                   # Dockerfiles (CPU/GPU) for CESGA/Singularity use
├── hpc/cesga/                # SLURM job scripts for running on CESGA
├── docs/                     # Sphinx documentation
├── setup.sh                  # one-time (idempotent) workspace setup
└── env.sh                    # source this in every new terminal
```

`emdb_interfaces` holds only message/service definitions and has no Python
code of its own; `emdb_simulator` and `emdb_policy` are plain `ament_python`
packages that import it.

## Prerequisites

- Ubuntu with **ROS 2** installed under `/opt/ros/<distro>` (`setup.sh`
  auto-detects whichever distro is present, e.g. Humble).
- Python 3 with the `venv` module available.
- `rosdep` (used by `setup.sh` to resolve ROS package dependencies).

## Installation

Clone the repository with submodules (RoboCasa/robosuite/robosuite_models
are git submodules — see [`.gitmodules`](.gitmodules)):

```bash
git clone --recurse-submodules <repo-url> TFM
# or, if already cloned:
git submodule update --init --recursive
```

Then run the setup script from the repo root:

```bash
./setup.sh            # full setup
./setup.sh --docs     # also install docs/ build dependencies
```

`setup.sh` is safe to re-run — every step checks whether it's already done
before acting. It:

1. locates the installed ROS 2 distro under `/opt/ros` and sources its
   `setup.bash`,
2. initializes any empty git submodules under `misc/`,
3. creates a Python virtualenv at `.venv/` (`--system-site-packages`, so
   `colcon`/`rosidl`'s system-installed toolchain is visible) and installs
   `robosuite`, `robocasa`, and (optionally) `robosuite_models` into it in
   editable mode, plus `colcon-common-extensions`,
4. resolves ROS package dependencies via `rosdep install`,
5. builds the `ros_packages` workspace with `colcon build --symlink-install`.

Override the venv location with `VENV_DIR=/path/to/venv ./setup.sh` if you
don't want it at `<repo>/.venv`.

## Activating the environment

For every new terminal that runs ROS 2 nodes from this workspace, source
`env.sh` instead of sourcing ROS/the venv manually — ordering matters (venv
vs. ROS `setup.bash`), and a couple of environment variables are required:

```bash
source env.sh
```

`env.sh`:

- sources ROS 2's `setup.bash`, then the `.venv` virtualenv (in that order,
  so the venv's Python doesn't shadow ROS 2's tools),
- sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (unless already set),
- sets `MUJOCO_GL=egl` so MuJoCo renders headlessly via EGL instead of GLX
  (unset it beforehand if your machine renders fine without it),
- sources `ros_packages/install/setup.bash` if the workspace has been built,
- enables colcon autocompletion if available,
- defines convenience aliases: `colcon_dev` (build with the venv's Python),
  `colcon_clean_build` (wipe `build/ install/ log/` and rebuild), and
  `emdb_env` (prints which `ros2`/`python` are active — useful for
  confirming you're not accidentally running the system Python).

Use `VENV_DIR=/path/to/venv source env.sh` if you set up the venv at a
non-default location.

## Building the ROS 2 workspace

`setup.sh` already builds it once; to rebuild after changing code:

```bash
source env.sh
cd ros_packages
colcon_dev             # alias for: python -m colcon build --symlink-install
source install/setup.bash
```

## Running the simulator

```bash
source env.sh
ros2 launch emdb_simulator emdb_simulator.launch.py
```

See [`docs/source/howto/run_simulator.md`](docs/source/howto/run_simulator.md)
for parameters, launch files, and topics, and the rest of
[`docs/source/howto/`](docs/source/howto/index.md) for teleoperation, demo
recording, RL training, and demo replay guides.

To run experiments on CESGA's FinisTerraeIII HPC cluster via
Docker/Singularity/SLURM instead, see
[`docs/source/howto/running_on_cesga.md`](docs/source/howto/running_on_cesga.md).

## Building the documentation

The docs are Sphinx + MyST (Markdown), with autodoc for the Python API
reference.

```bash
./setup.sh --docs     # or: pip install -r docs/requirements.txt (inside the venv)
source env.sh
cd docs
make html
```

Then open `docs/build/html/index.html` in a browser. See
[`docs/source/getting_started.md`](docs/source/getting_started.md) for more
detail, including autodoc's mocking of ROS-only imports.
