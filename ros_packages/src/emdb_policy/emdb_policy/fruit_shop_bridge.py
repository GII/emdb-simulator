#!/usr/bin/env python3
"""fruit_shop_bridge -- runs the real e-MDB Fruit Shop cognitive architecture
(MainLoop + LTM, unmodified, from paper_experiment/src/emdb_develop) against
TFM's own RoboCasa/robosuite physics simulator, with each of this single-arm
adaptation's policies implemented as a deterministic scripted motion (see
scripted_policies.py) instead of a learned/RL policy.

The e-MDB architecture speaks its own control protocol (same shape
mujoco_emdb_sim/sim_bridge.py already uses for the continuous-action case,
see paper_experiment/src/emdb_develop/emdb_experiments_gii/mujoco_emdb_sim):
  * executed_policy_service  (cognitive_node_interfaces/srv/Policy:
                               {policy: string} -> {success: bool})
  * world_reset_service      (cognitive_processes_interfaces/srv/WorldReset)
  * control_topic            (cognitive_processes_interfaces/msg/ControlMsg)

This node hosts those, and internally drives emdb_policy's own AgentBridge
(agent_bridge.py) against emdb_simulator's scene_loader -- the same
/step_action, /reset_episode protocol PickAndLiftPolicy/PlacePolicy already
use, just orchestrated policy-by-policy instead of continuously across an
episode.

Ported, single-arm-adapted state machine and stage-gated reward logic: see
paper_experiment/src/emdb_develop/emdb_discrete_event_simulator_gii/
simulators/simulators/fruit_shop_sim_discrete.py (FruitShopSim) -- that file
is the ground truth for what each policy does and when
classify_fruit/place_fruit rewards activate; read it directly before
changing anything here, don't guess from this file alone.

Single-arm simplifications vs. the two-hand reference (see
mdb_experiments/fruit_shop_experiment.yaml's own header for the full
change list):
  * fruit_in_left_hand/fruit_in_right_hand -> one fruit_in_hand; no
    change_hands policy.
  * test_fruit's hand/scale-angle-side matching is dropped -- a single arm
    can approach the scale from either side.
  * accept_fruit/discard_fruit's hand-specific catched-fruit fallback
    branches both check fruit_in_hand instead of a specific hand.
  * press_button/button_light removed entirely (no button perception, no
    press_button Policy node in the yaml) -- ApproachOnlyMotion's arm
    control couldn't reliably complete the approach within
    MAX_MOTION_STEPS even with the robot base already in reach.

Known deviation: ask_nicely's mid-episode "restock" (the reference sim
conjures new abstract fruit inventory out of nothing) has no physical
equivalent here -- scene_loader has no object-teleport service yet, so a
truly exhausted inventory is scripted as an idle wait instead of a real
restock (see ask_nicely_policy). This only matters if accept_fruit/
discard_fruit resolves the episode's one physical fruit and ask_nicely is
then called again in the same episode/trial.

Runtime note: like sim_bridge.py, this node imports emdb_interfaces (TFM
workspace) *and* cognitive_node_interfaces/cognitive_processes_interfaces/
core/core_interfaces/simulators_interfaces (emdb_develop workspace) -- the
shell that launches it must source BOTH workspaces' installs.

Full run sequence (mirrors emdb_experiments_gii/experiments/launch/
fruit_shop_launch.py's real orchestration -- commander dynamically builds
the whole LTM graph, including MainLoop itself, from the experiment yaml;
there is no separate "main_loop" process to launch by hand):

    # terminal 1 -- TFM physics sim
    source /home/fabian/Documents/TFM/env.sh
    ros2 run emdb_simulator scene_loader --ros-args -p control_mode:=rl -p task:=FruitShop -p perception_mode:=mdb

    # terminal 2 -- e-MDB commander
    source /home/fabian/Documents/paper_experiment/install/setup.sh
    ros2 run core commander

    # terminal 3 -- e-MDB LTM store
    source /home/fabian/Documents/paper_experiment/install/setup.sh
    ros2 run core ltm 0

    # terminal 4 -- one-shot: load the base commander config
    source /home/fabian/Documents/paper_experiment/install/setup.sh
    ros2 service call commander/load_config core_interfaces/srv/LoadConfig \
        "{file: '/home/fabian/Documents/paper_experiment/src/emdb_develop/emdb_core/core/config/commander.yaml'}"

    # terminal 5 -- this bridge; loads fruit_shop_experiment.yaml and tells
    # commander (via load_experiment_file_in_commander()) to build the LTM
    # graph from it
    source /home/fabian/Documents/paper_experiment/install/setup.sh
    source /home/fabian/Documents/TFM/ros_packages/install/setup.sh
    ros2 run emdb_policy fruit_shop_bridge --ros-args -p config_file:=/home/fabian/Documents/TFM/mdb_experiments/fruit_shop_experiment.yaml
"""
import os
import traceback

import numpy as np
import yaml
import yamlloader

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from core.service_client import ServiceClient
from core_interfaces.srv import LoadConfig
from core.utils import class_from_classname, resolve_seed

from std_msgs.msg import Bool, Float32
from simulators_interfaces.msg import FruitListMsg, FruitMsg, ScaleListMsg, ScaleMsg

from emdb_policy.agent_bridge import AgentBridge
from emdb_policy.scripted_policies import (
    PickFruitMotion,
    TransportReleaseMotion,
    IdleMotion,
)

DIM_MIN = 0.03  # meters, matches fruit_shop_sim_discrete.py's generate_fruits()
DIM_MAX = 0.1
FRUIT_SENTINEL = {"distance": 1.9, "angle": 1.4, "dim_max": 0.1}  # "nothing here"
MAX_MOTION_STEPS = 200


class FruitShopBridge(Node):
    """Bridges e-MDB's Policy/WorldReset/ControlMsg protocol to a physics
    scene, ported from FruitShopSim's state machine and reward logic."""

    def __init__(self):
        super().__init__("fruit_shop_bridge")

        self.random_seed = (
            self.declare_parameter("random_seed", value=0)
            .get_parameter_value().integer_value
        )
        self.config_file = (
            self.declare_parameter(
                "config_file", descriptor=ParameterDescriptor(dynamic_typing=True)
            ).get_parameter_value().string_value
        )
        self.standalone = (
            self.declare_parameter("standalone", value=False)
            .get_parameter_value().bool_value
        )

        self.rng = np.random.default_rng(resolve_seed(self.random_seed))
        self.cbgroup_server = MutuallyExclusiveCallbackGroup()

        # --- ported FruitShopSim state, single-arm-adapted -----------------
        self.iteration = 0
        # Unlike the reference's self.fruits (an abstract 0-3-item list),
        # this bridge only ever has one physical fruit object to act on
        # (see fruit_shop_task.py's own docstring), so the abstract
        # inventory collapses to a single availability flag + the one
        # fruit's sampled size.
        self.fruit_available = False
        self.fruit_dim_max = DIM_MAX
        self.closest_fruit = None  # {"distance", "angle", "dim_max"} or None
        self.catched_fruit = None
        self.tested_fruit = None
        self.fruit_in_hand = False
        self.scale_state = 0  # 0 untested, 1 good, 2 bad
        self.scale_active = False
        self.fruit_correctly_accepted = False
        self.fruit_correctly_rejected = False
        # Set (and read once by reward_classify_fruit_goal) when a *tested*
        # fruit gets dropped in the wrong basket -- see
        # reward_classify_fruit_goal's antipoint comment.
        self.fruit_incorrectly_classified = False
        self.classify_fruit_reward = 0.0
        self.test_fruit_reward = 0.0
        # Set by _resolve_tested_fruit/_release_held_fruit when the fruit
        # just got physically dropped into accepted/rejected -- consumed
        # at the *start* of the next executed_policy_callback, not
        # immediately, so the e-MDB architecture gets its normal
        # uninterrupted "read perceptions -> read rewards" pass over this
        # resolution's classify_fruit reward before the world (and that
        # reward) resets out from under it. See executed_policy_callback.
        self._pending_world_reset = False

        # --- physics-facing client ------------------------------------
        self._last_obs = None
        self.agent_bridge = AgentBridge()
        self.agent_bridge.start()
        self.agent_bridge.wait_for_services()

        # --- perception publishers (fixed, fruit-shop-specific topics) -
        self.fruits_pub = self.create_publisher(FruitListMsg, "/emdb/simulator/sensor/fruits", 10)
        self.scales_pub = self.create_publisher(ScaleListMsg, "/emdb/simulator/sensor/scales", 10)
        self.fruit_in_hand_pub = self.create_publisher(Bool, "/emdb/simulator/sensor/fruit_in_hand", 10)
        self.classify_fruit_pub = self.create_publisher(Float32, "/emdb/simulator/sensor/classify_fruit", 10)
        self.test_fruit_pub = self.create_publisher(Float32, "/emdb/simulator/sensor/test_fruit", 10)

    # ------------------------------------------------ yaml / e-MDB wiring
    def load_configuration(self):
        if not self.config_file or not os.path.isfile(self.config_file):
            self.get_logger().error(f"Config file '{self.config_file}' not found!")
            rclpy.shutdown()
            return
        config = yaml.load(
            open(self.config_file, "r", encoding="utf-8"),
            Loader=yamlloader.ordereddict.CLoader,
        )
        self.setup_control_channel(config["Control"])

        if not self.standalone:
            self.load_experiment_file_in_commander()
        else:
            self.get_logger().info("STANDALONE mode: not contacting the commander")

    def setup_control_channel(self, simulation):
        message = class_from_classname(simulation["control_msg"])
        self.create_subscription(
            message, simulation["control_topic"], self.control_callback, 0
        )

        service_policy = simulation.get("executed_policy_service")
        service_world_reset = simulation.get("world_reset_service")
        if service_policy:
            msg_policy = class_from_classname(simulation["executed_policy_msg"])
            self.create_service(
                msg_policy, service_policy, self.executed_policy_callback,
                callback_group=self.cbgroup_server,
            )
        if service_world_reset:
            msg_reset = class_from_classname(simulation["world_reset_msg"])
            self.create_service(
                msg_reset, service_world_reset, self.world_reset_callback,
                callback_group=self.cbgroup_server,
            )

        if service_policy:
            self.perceptions_timer = self.create_timer(
                0.01, self.publish_perceptions, callback_group=self.cbgroup_server
            )

    def load_experiment_file_in_commander(self):
        self.load_client = ServiceClient(LoadConfig, "commander/load_experiment")
        return self.load_client.send_request(file=self.config_file)

    # ------------------------------------------------ e-MDB service callbacks
    def control_callback(self, data):
        self.iteration = data.iteration
        self.update_reward_sensor()
        command = getattr(data, "command", "")
        if command == "reset_world":
            self._safe_reset_world()
        elif command == "end":
            self.get_logger().info("Ending fruit_shop_bridge as requested by LTM...")
            rclpy.shutdown()

    def world_reset_callback(self, request, response):
        response.success = self._safe_reset_world()
        return response

    def _safe_reset_world(self):
        """reset_world(), but never let an AgentBridge failure (e.g. a
        /reset_episode timeout because the sim is momentarily overloaded --
        seen live: a RoboCasa hard_reset occasionally outran even the 90s
        budget under CESGA CPU contention) escape as an exception.

        Same rationale as executed_policy_callback's own try/except around
        `method()`: an exception raised inside a ROS 2 callback has no
        default recovery in rclpy and kills the whole long-running node
        (and with it the rest of the job's --time budget). reset_world()
        itself is called from three callbacks (this one, control_callback's
        "reset_world" command, and executed_policy_callback's own deferred
        reset) that were all missing this guard -- only the *policy call*
        inside executed_policy_callback was ever wrapped, not the resets
        that surround it. Returns whether the reset actually succeeded.
        """
        try:
            self.reset_world()
            return True
        except Exception as e:
            self.get_logger().error(f"reset_world() failed unexpectedly: {e}")
            self.get_logger().error(traceback.format_exc())
            return False

    def executed_policy_callback(self, request, response):
        if self._pending_world_reset:
            # Deferred from the *previous* call's _resolve_tested_fruit/
            # _release_held_fruit (a fruit just got physically dropped
            # into accepted/rejected) until now instead of resetting
            # immediately at the end of that call -- MainLoop's own
            # per-iteration sequence is "execute policy -> read
            # perceptions -> ... -> read rewards" (all synchronous, before
            # it ever requests the *next* policy execution), so by the
            # time this callback fires again, the architecture has
            # already had its one uninterrupted chance to read the
            # classify_fruit reward from that resolution. Resetting at
            # the end of that same call instead (tried first) raced
            # reset_world()'s own perception/reward publish against
            # whatever MainLoop was reading right then, with no guarantee
            # the real value was seen before it got overwritten by the
            # freshly-reset one.
            self._pending_world_reset = False
            self._safe_reset_world()

        self.get_logger().info(f"Executing policy {request.policy} (iteration={self.iteration})")
        self.perceive_closest_fruit()
        method = getattr(self, request.policy + "_policy", None)
        if method is None:
            self.get_logger().error(f"Unknown policy {request.policy!r}")
            response.success = False
            return response
        try:
            success = method()
        except Exception as e:
            # A policy failing unexpectedly (e.g. an AgentBridge service
            # call raising because the sim rejected a request) used to
            # propagate straight out of this service callback and kill the
            # whole node -- rclpy has no default recovery for an exception
            # raised inside a service handler, so the entire
            # fruit_shop_bridge process died mid-episode, taking the rest
            # of the launched system down with it. Reset the episode
            # instead: this is already the recovery path for a bad/
            # unreachable procedurally-generated layout (world_reset_callback/
            # control_callback's "reset_world" command use the same
            # reset_world()), so it's a graceful way to shed a bad episode
            # rather than crashing the process.
            self.get_logger().error(f"Policy {request.policy!r} failed unexpectedly: {e}")
            self.get_logger().error(traceback.format_exc())
            self.reset_world()
            response.success = False
            return response
        self.perceive_closest_fruit()
        self.update_reward_sensor()
        self.publish_perceptions()
        response.success = bool(success)
        return response

    # ------------------------------------------------ world / perception
    def reset_world(self):
        self.get_logger().info("Resetting FruitShop world...")
        self._pending_world_reset = False
        self.fruit_correctly_accepted = False
        self.fruit_correctly_rejected = False
        self.fruit_incorrectly_classified = False
        self._last_obs = self.agent_bridge.reset()
        self.catched_fruit = None
        self.tested_fruit = None
        self.fruit_in_hand = False
        self.scale_state = 0
        self.scale_active = False
        self.fruit_available = bool(self.rng.uniform() > 0.5)
        self.fruit_dim_max = float(self.rng.uniform(DIM_MIN, DIM_MAX))
        self.perceive_closest_fruit()
        self.update_reward_sensor()
        self.publish_perceptions()

    def perceive_closest_fruit(self):
        if self.fruit_available and self._last_obs is not None and "fruit_pos" in self._last_obs:
            distance, angle = self._polar(self._last_obs["fruit_pos"])
            self.closest_fruit = {"distance": distance, "angle": angle, "dim_max": self.fruit_dim_max}
        else:
            self.closest_fruit = None

    def _polar(self, pos):
        base_pos = self._last_obs.get("robot_base_pos") if self._last_obs is not None else None
        if base_pos is None:
            base_pos = np.zeros(3)
        rel = np.asarray(pos, dtype=np.float64) - np.asarray(base_pos, dtype=np.float64)
        distance = float(np.linalg.norm(rel[:2]))
        angle = float(np.arctan2(rel[0], rel[1]))
        return distance, angle


    def update_reward_sensor(self):
        self.classify_fruit_reward = self.reward_classify_fruit_goal()
        self.test_fruit_reward = self.reward_test_fruit_goal()

    def reward_classify_fruit_goal(self):
        # No curriculum staging (deviates from the reference, which only
        # ever returns 0.0/1.0 here and gates it behind
        # change_reward_iterations['stage2']) -- always reflects the real
        # outcome. pick_fruit still only gets reward from the
        # architecture's generic effect_fruit_in_hand_data effectance
        # goal; test_fruit has its own dedicated test_fruit_drive/mission
        # now (see reward_test_fruit_goal), classify_fruit_goal only needs
        # to cover the actual sort decision.
        if self.fruit_correctly_accepted or self.fruit_correctly_rejected:
            return 1.0
        # Antipoint: a *tested* fruit (a real scale_state) dropped in the
        # wrong basket -- not in the reference, added so a wrong
        # classification reads as worse than simply not having resolved a
        # fruit yet.
        if self.fruit_incorrectly_classified:
            return -1.0
        return 0.0

    def reward_test_fruit_goal(self):
        # 1.0 for as long as a tested fruit is actively sitting on the
        # scale with a real, determined state (set by test_fruit_policy,
        # cleared once accept_fruit/discard_fruit resolves it or the world
        # resets) -- dedicated test_fruit_drive/test_fruit_mission
        # (fruit_shop_experiment.yaml), replacing the old place_fruit_
        # drive slot. Takes over from the generic effect_scales_active
        # effectance goal (which rewards *any* scales:active change, not
        # specifically "successfully tested a fruit").
        return 1.0 if self.scale_active else 0.0

    def publish_perceptions(self):
        fruit_entry = FruitMsg()
        source = self.closest_fruit or FRUIT_SENTINEL
        fruit_entry.distance = source["distance"]
        fruit_entry.angle = source["angle"]
        fruit_entry.dim_max = source["dim_max"]
        self.fruits_pub.publish(FruitListMsg(data=[fruit_entry]))

        scale_entry = ScaleMsg()
        if self._last_obs is not None and "scale_pos" in self._last_obs:
            distance, angle = self._polar(self._last_obs["scale_pos"])
        else:
            distance, angle = 0.6, 0.0
        scale_entry.distance = distance
        scale_entry.angle = angle
        scale_entry.dim_max = 0.06  # unused downstream -- FruitShopPerception's scales branch never reads it
        scale_entry.state = int(self.scale_state)
        scale_entry.active = bool(self.scale_active)
        self.scales_pub.publish(ScaleListMsg(data=[scale_entry]))

        self.fruit_in_hand_pub.publish(Bool(data=bool(self.fruit_in_hand)))
        self.classify_fruit_pub.publish(Float32(data=float(self.classify_fruit_reward)))
        self.test_fruit_pub.publish(Float32(data=float(self.test_fruit_reward)))

    # ------------------------------------------------ scripted-motion runner
    def _run_motion(self, motion, max_steps=MAX_MOTION_STEPS):
        """Drive `motion` to its DONE state via AgentBridge.step_vector(),
        updating self._last_obs as it goes. Returns True if the motion
        reached STATE_DONE within max_steps, False on timeout/termination --
        a real physical failure (e.g. a missed grasp or a target out of
        arm reach), unlike the reference sim where every policy call
        always succeeds."""
        for _ in range(max_steps):
            action = motion.policy_fn(self._last_obs, self.rng)
            obs, _reward, terminated, truncated, _info = self.agent_bridge.step_vector(action)
            self._last_obs = obs
            if motion.state == motion.STATE_DONE:
                return True
            if terminated or truncated:
                return False
        return False

    # ------------------------------------------------ policy implementations
    # Each ports its FruitShopSim counterpart's decision logic (when to act,
    # what state to mutate) but replaces instantaneous dict mutation with a
    # blocking scripted motion against the real robot.

    def pick_fruit_policy(self):
        if self.catched_fruit or not self.fruit_available:
            return True
        motion = PickFruitMotion()
        motion.on_episode_start()
        success = self._run_motion(motion)
        if success:
            self.fruit_in_hand = True
            if self.scale_active:
                self.scale_active = False
                self.tested_fruit = None
            self.fruit_correctly_rejected = False
            self.fruit_correctly_accepted = False
            self.catched_fruit = self.closest_fruit
        return success

    def place_fruit_policy(self):
        if not self.catched_fruit or not self.fruit_in_hand:
            return True
        motion = TransportReleaseMotion(target_pos_key="placed_pos")
        motion.on_episode_start()
        success = self._run_motion(motion)
        if success:
            self.fruit_in_hand = False
            self.catched_fruit = None
        return success

    def test_fruit_policy(self):
        if not self.catched_fruit or not self.fruit_in_hand:
            return True
        motion = TransportReleaseMotion(target_pos_key="scale_pos")
        motion.on_episode_start()
        success = self._run_motion(motion)
        if success:
            self.fruit_in_hand = False
            # Always determine and reveal the fruit's real state (1 good /
            # 2 rotten) once it's physically been placed on the scale --
            # this is a perception fact (the scale's own sensor reading of
            # the fruit sitting on it, publish_perceptions's scale_entry.
            # state, normalized downstream by FruitShopPerception's
            # n_states: 3), not a reward -- no curriculum staging here.
            self.scale_active = True
            if self.scale_state == 0:
                self.scale_state = 1 if self.rng.uniform() > 0.5 else 2
                if self.scale_state == 2:
                    # Reveal "rotten" only now -- the scale is what tells
                    # us it's bad, not a look at the fruit beforehand.
                    self.agent_bridge.mark_object_rotten("fruit")
            self.tested_fruit = self.catched_fruit
            self.catched_fruit = None
        return success

    def accept_fruit_policy(self):
        if self.scale_active:
            return self._resolve_tested_fruit("accepted_pos", correct_state=1, correct_attr="fruit_correctly_accepted")
        if self.catched_fruit and self.fruit_in_hand:
            return self._release_held_fruit("accepted_pos")
        return True

    def discard_fruit_policy(self):
        if self.scale_active:
            return self._resolve_tested_fruit("rejected_pos", correct_state=2, correct_attr="fruit_correctly_rejected")
        if self.catched_fruit and self.fruit_in_hand:
            return self._release_held_fruit("rejected_pos")
        return True

    def _resolve_tested_fruit(self, target_pos_key, correct_state, correct_attr):
        """Re-grasp the tested fruit off the scale (there is no teleport in
        physics -- test_fruit already released it there) and carry it to
        target_pos_key, matching accept/discard's `if scale.active` branch
        in the reference."""
        pick = PickFruitMotion()
        pick.on_episode_start()
        if not self._run_motion(pick):
            return False
        place = TransportReleaseMotion(target_pos_key=target_pos_key)
        place.on_episode_start()
        success = self._run_motion(place)
        if success:
            if self.scale_state == correct_state:
                setattr(self, correct_attr, True)
                # scene_loader.py can't see this outcome on its own --
                # FruitShop._check_success() always returns False by design
                # (see its docstring), so video_recorder's keep_successes
                # mode would otherwise never keep a single FruitShop
                # episode no matter how many fruits get classified
                # correctly.
                self.agent_bridge.mark_episode_success()
            else:
                self.fruit_incorrectly_classified = True
            self.scale_state = 0
            self.scale_active = False
            self.tested_fruit = None
            self.fruit_available = False  # resolved -- out of play until ask_nicely
            self._pending_world_reset = True
        return success

    def _release_held_fruit(self, target_pos_key):
        """accept/discard's fallback branch: directly move an already-held,
        not-yet-tested fruit (single-arm: both accept and discard use this,
        the reference restricts each to one specific hand)."""
        motion = TransportReleaseMotion(target_pos_key=target_pos_key)
        motion.on_episode_start()
        success = self._run_motion(motion)
        if success:
            self.fruit_in_hand = False
            self.catched_fruit = None
            self.fruit_available = False
            self._pending_world_reset = True
        return success

    def ask_nicely_policy(self):
        if self.fruit_available:
            return True
        # No object-teleport service exists yet to physically restock the
        # one fruit body once it's been resolved mid-episode -- see the
        # module docstring's "Known deviation" note. Scripted as an idle
        # wait for now.
        motion = IdleMotion()
        motion.on_episode_start(gripper_closed=self.fruit_in_hand)
        self._run_motion(motion)
        return True


def main(args=None):
    rclpy.init(args=args)
    bridge = FruitShopBridge()
    bridge.load_configuration()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(bridge, executor=executor)
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected: Shutting down fruit_shop_bridge...")
    finally:
        bridge.agent_bridge.close()
        bridge.destroy_node()


if __name__ == "__main__":
    main()
