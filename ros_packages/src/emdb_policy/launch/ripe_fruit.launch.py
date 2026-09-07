"""Launches the e-MDB architecture side of the RipeFruit experiment:
commander + ltm + ripe_fruit_bridge, then triggers commander/load_config
and (via the bridge's own load_experiment_file_in_commander()) commander/
load_experiment.

Mirrors this package's own fruit_shop.launch.py -- same commander/ltm/
config_service_call shape, with ripe_fruit_bridge swapped in for
fruit_shop_bridge (its own executed_policy_service, see
ripe_fruit_bridge.py's module docstring).

Doesn't launch the physics sim itself -- start that separately first, same
two-invocation split fruit_shop.launch.py's own docstring documents:

    ros2 run emdb_simulator scene_loader --ros-args -p control_mode:=rl -p task:=RipeFruit -p perception_mode:=mdb -p robot:=PandaOmron
    ros2 launch emdb_policy ripe_fruit.launch.py

Unlike FruitShop, RipeFruit has no layout_id pinning to opt into here --
confirmed live (2026-09) that FruitShop's own curated layout (21) sends
ChooseRipeFruit's fruit0/fruit1 placement into RoboCasa's _load_model()
retry loop indefinitely, so RipeFruit stays on scene_loader's default
random layout selection regardless of layout_id.
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
        executable="ripe_fruit_bridge",
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
            default_value="/home/fabian/Documents/TFM/ros_ws/src/emdb-simulator/mdb_experiments/ripe_fruit_experiment.yaml",
            description="Absolute path to the RipeFruit experiment yaml"),
        DeclareLaunchArgument(
            "commander_config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("core"), "config", "commander.yaml"]
            ),
            description="Base commander config -- RipeFruit's LTM graph is a "
                         "small, from-scratch MainLoop/threads:2 graph (see "
                         "ripe_fruit_experiment.yaml), same MainLoop shape "
                         "fruit_shop.launch.py's own default already assumes."),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
