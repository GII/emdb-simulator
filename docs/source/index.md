# EMDB Simulator Documentation

EMDB is a ROS 2 workspace that wraps [RoboCasa](https://github.com/SantaCRC/robocasa) /
[robosuite](https://github.com/SantaCRC/robosuite) kitchen-manipulation simulation behind a
ROS 2 topic/service interface, so that teleoperation, demo recording, and RL
training/inference can all talk to the simulator without depending on
robosuite/robocasa/MuJoCo directly.

```{toctree}
:maxdepth: 2
:caption: Contents

about
getting_started
architecture
howto/index
interfaces/index
api/index
llms
```

This project is the Master's Thesis (TFM) of Fabian Alvarez Juarez, for the
Máster en Informática Industrial y Robótica at the Universidade da Coruña,
carried out within the Grupo Integrado de Ingeniería (GII) — see {doc}`about`
for the full acknowledgments, including CESGA's computing resources.

## At a glance

- **{ref}`emdb_simulator`** owns the MuJoCo/robosuite/RoboCasa
  environment, steps physics, and publishes observations/joint states/object
  poses while exposing action services.
- **{ref}`emdb_policy`** is a separate process that consumes the
  simulator's ROS interface only; hosts a `gymnasium.Env` wrapper, an SB3 PPO
  training script, and a demo-replay tool.
- **[emdb_interfaces](interfaces/index.md)** holds the custom `.msg`/`.srv` definitions shared
  between the two.

## Quick links

- New to the project? Start with {doc}`getting_started`.
- Want to drive the robot by hand? See {doc}`howto/teleoperation`.
- Want to train a policy? See {doc}`howto/training_rl`.
- Looking for a message/service field? See {doc}`interfaces/index`.
- Looking for a Python function/class? See {doc}`api/index`.
- Automating task/robot creation with an LLM agent? See {doc}`llms`.
