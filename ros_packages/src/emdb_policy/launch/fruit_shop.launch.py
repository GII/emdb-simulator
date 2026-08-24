"""Launches the e-MDB architecture side of the Fruit Shop experiment:
commander + ltm + fruit_shop_bridge, then triggers commander/load_config
and (via the bridge's own load_experiment_file_in_commander()) commander/
load_experiment.

Mirrors paper_experiment/src/emdb_develop/emdb_experiments_gii/experiments/
launch/lift_launch.py's structure -- same commander/ltm/config_service_call
shape, with mujoco_emdb_sim's sim_bridge swapped for this workspace's own
fruit_shop_bridge (executed_policy_service instead of executed_action_service
-- see fruit_shop_bridge.py's own module docstring for why a separate bridge
was needed instead of reusing sim_bridge).

Doesn't launch the physics sim itself -- start that separately first, same
two-invocation split lift_experiment.yaml's header documents ("A) sim ...
B) arch: ros2 launch experiments lift_launch.py"):

    ros2 run emdb_simulator scene_loader --ros-args -p control_mode:=rl -p task:=FruitShop -p perception_mode:=mdb -p layout_id:=-1
    ros2 launch emdb_policy fruit_shop.launch.py

layout_id:=-1 opts into scene_loader's FRUIT_SHOP_LAYOUT_IDS randomization
(scene_loader.py) -- a curated set of island layouts, re-sampled every
episode reset, instead of the node's generic layout_id default (12, a fixed
single layout chosen for KitchenLift's needs).
"""
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.substitutions import LaunchConfiguration, FindExecutable, PathJoinSubstitution


def launch_setup(context: LaunchContext, *args, **kwargs):
    logger = LaunchConfiguration("log_level")
    random_seed = LaunchConfiguration("random_seed")
    config_file = LaunchConfiguration("config_file")
    commander_config_file = LaunchConfiguration("commander_config_file")

    core_node = Node(
        package="core",
        executable="commander",
        output="screen",
        arguments=["--ros-args", "--log-level", logger],
        parameters=[{"random_seed": random_seed}],
    )

    ltm_node = Node(
        package="core",
        executable="ltm",
        output="screen",
        arguments=["0", "--ros-args", "--log-level", logger],
    )

    bridge_node = Node(
        package="emdb_policy",
        executable="fruit_shop_bridge",
        output="screen",
        arguments=["--ros-args", "--log-level", logger],
        parameters=[
            {
                "random_seed": random_seed,
                "config_file": config_file,
            }
        ],
    )

    config_service_call = ExecuteProcess(
        cmd=[
            [
                FindExecutable(name="ros2"),
                " ",
                "service call",
                " ",
                "commander/load_config",
                " ",
                "core_interfaces/srv/LoadConfig",
                " ",
                '"{file:',
                " ",
                commander_config_file,
                '}"',
            ]
        ],
        shell=True,
    )

    shutdown_on_exit = RegisterEventHandler(
        OnProcessExit(target_action=core_node, on_exit=[Shutdown()])
    )

    return [config_service_call, core_node, ltm_node, bridge_node, shutdown_on_exit]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "log_level", default_value=["info"], description="Logging level"),
        DeclareLaunchArgument(
            "random_seed", default_value="0",
            description="The seed to the random numbers generator"),
        DeclareLaunchArgument(
            "config_file",
            default_value="/home/fabian/Documents/TFM/mdb_experiments/fruit_shop_experiment.yaml",
            description="Absolute path to the Fruit Shop experiment yaml"),
        DeclareLaunchArgument(
            "commander_config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("core"), "config", "commander.yaml"]
            ),
            description="Base commander config -- matches the real (two-hand) "
                         "fruit_shop_experiment.yaml's own launch file, since "
                         "this experiment still uses MainLoop/threads:2, not "
                         "MainLoopLight (which is what lift_launch.py's "
                         "commander_threaded.yaml default is for)."),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
