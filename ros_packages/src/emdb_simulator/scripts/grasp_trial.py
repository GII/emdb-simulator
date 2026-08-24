#!/usr/bin/env python3
"""Empirical grasp trial for TwoFG7Gripper (~31mm jaw) against RoboCasa fruit
categories -- see fruit_shop_task.py's DEFAULT_OBJ_GROUPS comment for why
this is needed (static bbox math already shown unreliable, see
kitchen_lift_task.py's own history).

For each candidate category, launches scene_loader with task=KitchenLift and
scene_loader.py's obj_groups ROS param set to that category (overriding
KitchenLift's own DEFAULT_OBJ_GROUPS=["cube"] for the run, no source edits),
then drives PickAndLiftPolicy for a few episodes via policy_node
(emdb_policy/policy_node.py --policy pick_and_lift) and records whether
/emdb/simulator/sensor/obj/grasped ever fired.

Needs a sourced ROS2 + workspace environment (source /opt/ros/<distro>/setup.bash
&& source ros_packages/install/setup.bash) with emdb_simulator/emdb_policy
built. Run:

    python3 ros_packages/src/emdb_simulator/scripts/grasp_trial.py \\
        --out /tmp/grasp_trial_results.csv --log-dir /tmp/grasp_trial_logs
"""
import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path

DEFAULT_CATEGORIES = [
    "apple", "apricot", "banana", "cantaloupe", "cherry", "coconut", "dates",
    "grapes", "kiwi", "lime", "mango", "orange", "peach", "pear", "pineapple",
    "pomegranate", "raspberry", "strawberry", "tangerine", "watermelon",
]

SCENE_READY_TIMEOUT = 60
NUM_EPISODES = 3
POLICY_TIMEOUT = 150
GRASP_MONITOR_TIMEOUT = POLICY_TIMEOUT + 20
TEARDOWN_TIMEOUT = 10


def _terminate_process_group(proc, timeout):
    """`ros2 run` doesn't exec() into the entry-point script -- it forks a
    grandchild (the actual scene_loader/policy_node python process) that a
    plain proc.terminate() (SIGTERM to the immediate ros2-run child only)
    never reaches, leaving it orphaned and running. start_new_session=True
    at Popen time puts the whole tree in its own process group so a signal
    to -pgid reaches all of it."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
    except ProcessLookupError:
        pass


def wait_for_services(timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = subprocess.run(
            ["ros2", "service", "list"], capture_output=True, text=True
        ).stdout
        if "/reset_episode" in out and "/step_action" in out:
            return True
        time.sleep(1)
    return False


def run_category(category, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_log_path = out_dir / "scene_loader.log"
    scene_log = open(scene_log_path, "w")
    scene_proc = subprocess.Popen(
        [
            "ros2", "run", "emdb_simulator", "scene_loader", "--ros-args",
            "-p", "task:=KitchenLift",
            "-p", "control_mode:=rl",
            "-p", "perception_mode:=mdb",
            "-p", "headless:=true",
            "-p", f"obj_groups:={category}",
        ],
        stdout=scene_log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        if not wait_for_services(SCENE_READY_TIMEOUT):
            return {
                "category": category, "grasped": False, "success": False,
                "note": "scene_loader services never came up (see scene_loader.log)",
            }

        grasp_log_path = out_dir / "grasped.log"
        grasp_log = open(grasp_log_path, "w")
        monitor_proc = subprocess.Popen(
            [
                "timeout", str(GRASP_MONITOR_TIMEOUT), "ros2", "topic", "echo",
                "/emdb/simulator/sensor/obj/grasped", "std_msgs/msg/Bool",
            ],
            stdout=grasp_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        policy_log_path = out_dir / "policy_node.log"
        with open(policy_log_path, "w") as policy_log:
            subprocess.run(
                [
                    "timeout", str(POLICY_TIMEOUT), "ros2", "run", "emdb_policy",
                    "policy_node", "--policy", "pick_and_lift",
                    "--num-episodes", str(NUM_EPISODES),
                ],
                stdout=policy_log, stderr=subprocess.STDOUT,
            )

        _terminate_process_group(monitor_proc, timeout=5)
        grasp_log.close()

        grasped = "data: true" in grasp_log_path.read_text()
        success = "success=True" in policy_log_path.read_text()
        return {"category": category, "grasped": grasped, "success": success, "note": ""}
    finally:
        _terminate_process_group(scene_proc, timeout=TEARDOWN_TIMEOUT)
        scene_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--out", default="grasp_trial_results.csv")
    parser.add_argument("--log-dir", default="grasp_trial_logs")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    out_path = Path(args.out)
    log_root = Path(args.log_dir)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "grasped", "success", "note"])
        writer.writeheader()
        f.flush()
        for category in categories:
            print(f"=== {category} ===", flush=True)
            result = run_category(category, log_root / category)
            writer.writerow(result)
            f.flush()
            print(
                f"    grasped={result['grasped']} success={result['success']} {result['note']}",
                flush=True,
            )
            time.sleep(2)  # let the previous node's name/ports free up


if __name__ == "__main__":
    main()
