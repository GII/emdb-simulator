#!/usr/bin/env python3
"""ripe_fruit_bridge -- runs the e-MDB cognitive architecture (MainLoop +
LTM) against TFM's own RoboCasa/robosuite physics simulator for the
RipeFruit task (emdb_simulator.core.ripe_fruit_task, a thin wrapper around
RoboCasa's own ChooseRipeFruit), with each policy implemented as a
deterministic scripted motion (see scripted_policies.py) instead of a
learned/RL policy.

Unlike fruit_shop_bridge.py, this is a direct, minimal port: RipeFruit
doesn't hide or randomize which fruit is ripe (fruit1 always is, per
RoboCasa's own ChooseRipeFruit._update_fruit_texture()), so there's no
multi-stage classify/test/accept/discard state machine here -- just two
policies, pick_fruit and place_in_blender. RipeFruit also has a real,
working env._check_success() (unlike FruitShop's, which is always False by
design), so scene_loader.py's own StepInfo.success already reflects the
real outcome -- this bridge has no need for /mark_episode_success.

The e-MDB architecture speaks its own control protocol (same shape
fruit_shop_bridge.py already uses):
  * executed_policy_service  (cognitive_node_interfaces/srv/Policy:
                               {policy: string} -> {success: bool})
  * world_reset_service      (cognitive_processes_interfaces/srv/WorldReset)
  * control_topic            (cognitive_processes_interfaces/msg/ControlMsg)

This node hosts those, and internally drives emdb_policy's own AgentBridge
(agent_bridge.py) against emdb_simulator's scene_loader -- the same
/step_action, /reset_episode protocol fruit_shop_bridge.py already uses.

Full run sequence (mirrors fruit_shop_bridge.py's own documented manual
sequence -- commander dynamically builds the whole LTM graph from the
experiment yaml; there is no separate "main_loop" process to launch by
hand):

    # terminal 1 -- TFM physics sim
    source /home/fabian/Documents/TFM/env.sh
    ros2 run emdb_simulator scene_loader --ros-args -p control_mode:=rl -p task:=RipeFruit -p perception_mode:=mdb

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

    # terminal 5 -- this bridge; loads ripe_fruit_experiment.yaml and tells
    # commander (via load_experiment_file_in_commander()) to build the LTM
    # graph from it
    source /home/fabian/Documents/paper_experiment/install/setup.sh
    source /home/fabian/Documents/TFM/ros_packages/install/setup.sh
    ros2 run emdb_policy ripe_fruit_bridge --ros-args -p config_file:=/home/fabian/Documents/TFM/mdb_experiments/ripe_fruit_experiment.yaml
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

from std_msgs.msg import Float32
from simulators_interfaces.msg import FruitListMsg, FruitMsg

from emdb_policy.agent_bridge import AgentBridge, SceneLoaderUnavailableError
from emdb_policy.scripted_policies import PickObjectMotion, TransportReleaseMotion

MAX_MOTION_STEPS = 200


class RipeFruitBridge(Node):
    """Bridges e-MDB's Policy/WorldReset/ControlMsg protocol to RipeFruit's
    physics scene: pick the ripe fruit (fruit1), place it in the blender."""

    def __init__(self):
        super().__init__("ripe_fruit_bridge")

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

        self.iteration = 0
        self.fruit_in_hand = False
        self._last_obs = None
        # Set from the "success" info dict AgentBridge.step()/step_vector()
        # already returns every call (scene_loader's own
        # bool(self.env._check_success()) or self._external_success) --
        # RipeFruit has a real, working _check_success() (unlike FruitShop's,
        # which is always False by design), so this is the real outcome,
        # not something this bridge needs to compute itself.
        self._last_info = None

        # --- physics-facing client ------------------------------------
        self.agent_bridge = AgentBridge()
        self.agent_bridge.start()
        self.agent_bridge.wait_for_services()

        # --- perception publishers -------------------------------------
        # Reuses cognitive_nodes.perception.FruitShopPerception's existing
        # "fruits" in self.name normalization branch on the LTM side (see
        # ripe_fruit_experiment.yaml) -- same FruitListMsg shape FruitShop's
        # own "fruits" perception already publishes, so no new perception
        # class is needed.
        self.fruits_pub = self.create_publisher(FruitListMsg, "/emdb/simulator/sensor/target_fruit", 10)
        self.ripe_fruit_in_blender_pub = self.create_publisher(
            Float32, "/emdb/simulator/sensor/ripe_fruit_in_blender", 10
        )

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
            self.get_logger().info("Ending ripe_fruit_bridge as requested by LTM...")
            rclpy.shutdown()

    def world_reset_callback(self, request, response):
        response.success = self._safe_reset_world()
        return response

    def _safe_reset_world(self):
        """reset_world(), but never let an AgentBridge failure (e.g. a
        /reset_episode timeout because the sim is momentarily overloaded)
        escape as an exception -- see fruit_shop_bridge.py's identical
        method for the full rationale. Returns whether the reset succeeded."""
        try:
            self.reset_world()
            return True
        except SceneLoaderUnavailableError:
            self.get_logger().fatal(
                "scene_loader appears to have died during reset_world() -- this "
                "cannot self-heal within this job; propagating to shut down "
                "ripe_fruit_bridge."
            )
            raise
        except Exception as e:
            self.get_logger().error(f"reset_world() failed unexpectedly: {e}")
            self.get_logger().error(traceback.format_exc())
            return False

    def executed_policy_callback(self, request, response):
        self.get_logger().info(f"Executing policy {request.policy} (iteration={self.iteration})")
        method = getattr(self, request.policy + "_policy", None)
        if method is None:
            self.get_logger().error(f"Unknown policy {request.policy!r}")
            response.success = False
            return response
        try:
            success = method()
        except SceneLoaderUnavailableError:
            self.get_logger().fatal(
                f"scene_loader appears to have died while executing policy "
                f"{request.policy!r} -- this cannot self-heal within this job; "
                "shutting down ripe_fruit_bridge so the sbatch script ends the job "
                "instead of burning the remaining time budget on a dead simulator."
            )
            raise
        except Exception as e:
            self.get_logger().error(f"Policy {request.policy!r} failed unexpectedly: {e}")
            self.get_logger().error(traceback.format_exc())
            # _safe_reset_world(), not a direct self.reset_world() call:
            # confirmed live (2026-09) that if THIS recovery reset also
            # times out (e.g. scene_loader is stuck, not dead -- a plain
            # TimeoutError, not SceneLoaderUnavailableError, so it wasn't
            # caught by the except clause above either), the raw call let
            # that second exception escape unguarded, killing the whole
            # bridge process outright (rclpy has no default recovery for an
            # exception raised inside a service callback). _safe_reset_world
            # still re-raises a genuine SceneLoaderUnavailableError (dead
            # forever, correctly fatal) but absorbs an ordinary transient
            # timeout here instead of crashing on a double failure.
            self._safe_reset_world()
            response.success = False
            return response
        self.update_reward_sensor()
        self.publish_perceptions()
        response.success = bool(success)
        return response

    # ------------------------------------------------ world / perception
    def reset_world(self):
        self.get_logger().info("Resetting RipeFruit world...")
        self._last_obs = self.agent_bridge.reset()
        self.fruit_in_hand = False
        self._last_info = None
        self.publish_perceptions()

    def _polar(self, pos):
        base_pos = self._last_obs.get("robot_base_pos") if self._last_obs is not None else None
        if base_pos is None:
            base_pos = np.zeros(3)
        rel = np.asarray(pos, dtype=np.float64) - np.asarray(base_pos, dtype=np.float64)
        distance = float(np.linalg.norm(rel[:2]))
        angle = float(np.arctan2(rel[0], rel[1]))
        return distance, angle

    def update_reward_sensor(self):
        self.ripe_fruit_in_blender_reward = (
            1.0 if (self._last_info and self._last_info.get("success")) else 0.0
        )

    def publish_perceptions(self):
        fruit_entry = FruitMsg()
        if self._last_obs is not None and "fruit1_pos" in self._last_obs:
            distance, angle = self._polar(self._last_obs["fruit1_pos"])
            fruit_entry.distance = distance
            fruit_entry.angle = angle
            fruit_entry.dim_max = 0.1
        else:
            fruit_entry.distance = 1.9
            fruit_entry.angle = 1.4
            fruit_entry.dim_max = 0.1
        self.fruits_pub.publish(FruitListMsg(data=[fruit_entry]))
        self.ripe_fruit_in_blender_pub.publish(
            Float32(data=float(getattr(self, "ripe_fruit_in_blender_reward", 0.0)))
        )

    # ------------------------------------------------ scripted-motion runner
    def _run_motion(self, motion, max_steps=MAX_MOTION_STEPS):
        """Drive `motion` to its DONE state via AgentBridge.step_vector(),
        updating self._last_obs/self._last_info as it goes. Returns True if
        the motion reached STATE_DONE within max_steps, False on
        timeout/termination."""
        for _ in range(max_steps):
            action = motion.policy_fn(self._last_obs, self.rng)
            obs, _reward, terminated, truncated, info = self.agent_bridge.step_vector(action)
            self._last_obs = obs
            self._last_info = info
            if motion.state == motion.STATE_DONE:
                return True
            if terminated or truncated:
                return False
        return False

    # ------------------------------------------------ policy implementations
    def pick_fruit_policy(self):
        if self.fruit_in_hand:
            return True
        # obstacle_clear_z=dest_top_z (the blender's own rim height): fruit1
        # spawns immediately next to the blender (ChooseRipeFruit's own
        # placement, ref=self.blender), so a plain diagonal approach can
        # drag along or get stuck against the blender's side while still
        # below its rim -- confirmed live (2026-09): up to 43 steps of
        # sustained contact on an otherwise "successful" pick. See
        # ScriptedPolicyBase's obstacle_clear_z for the up/over/down fix.
        motion = PickObjectMotion(
            obj_pos_key="fruit1_pos", obstacle_clear_z=self._last_obs.get("dest_top_z"),
        )
        motion.on_episode_start()
        success = self._run_motion(motion)
        if success:
            self.fruit_in_hand = True
        return success

    def place_in_blender_policy(self):
        if not self.fruit_in_hand:
            return True
        # target_pos_key="blender_int_pos", not the default "dest_pos": the
        # blender's own bounding-box center (dest_pos) is close but not
        # exact -- confirmed live (2026-09) that the real success check
        # (ChooseRipeFruit._check_success() -> OU.obj_inside_of, a tight
        # ~9x9cm region) sits a few cm off from it, tight enough that the
        # motion completing cleanly didn't mean the fruit actually landed
        # inside. blender_int_pos (ripe_fruit_task.py) is read directly
        # from the same Fixture.get_int_sites() geometry the success check
        # itself uses.
        #
        # NOT lower_target_z (tried live, 2026-09, made things worse): an
        # explicit descend-to-the-floor target got the arm stuck fighting
        # the ~9-13cm opening's walls (the gripper+held-fruit assembly
        # appears too wide to fit down into it at all) or knocked the fruit
        # loose mid-descent, landing on the counter or even the floor
        # entirely. Releasing from just above the rim (the generic
        # STATE_LOWER behavior, driven by blender_int_top_z) at least
        # sometimes succeeds; it also sometimes leaves the fruit stuck to
        # the gripper (never separating even after 6x the normal
        # RELEASE_SETTLE_STEPS) or dropped outside the opening. This is a
        # genuinely unreliable release, not a single fixable bug -- see the
        # conversation for the open question on how to proceed (accept the
        # failure rate, invest more in precision, or a different strategy
        # entirely).
        motion = TransportReleaseMotion(target_pos_key="blender_int_pos")
        motion.on_episode_start()
        success = self._run_motion(motion)
        if success:
            self.fruit_in_hand = False
        return success

    def open_blender_policy(self):
        """Grasp the blender's lid and set it aside on the counter -- a
        necessary precondition for place_in_blender_policy to actually
        succeed (the blender always spawns with a lid on, see
        RipeFruit's own docstring). Mirrors RoboCasa's own OpenBlenderLid
        atomic task's success condition (lid off the gripper, touching a
        counter) rather than gating place_in_blender itself: whether the
        lid is open only affects whether the fruit physically ends up
        inside the blender, which the terminal reward already reflects via
        the sim's real _check_success() -- this policy's own success is
        just "did the scripted motion complete", same as every other one
        here."""
        # Real, RoboCasa-tracked ground truth (Blender.get_state()
        # ["lid_on_blender"], refreshed every env.step() -- see
        # RipeFruit.lid_on_blender's own comment), not a self-reported
        # "my own motion said DONE once before" flag: the latter used to be
        # this policy's only gate, but a motion reaching STATE_DONE only
        # means it finished, not that the lid actually ended up off the
        # blender -- confirmed live (2026-09) this can diverge, the same
        # class of gap already found and fixed for place_in_blender's own
        # "motion succeeded but the fruit didn't land inside" issue.
        if self._last_obs is not None and not bool(self._last_obs.get("lid_on_blender", 1.0)):
            return True
        # The lid's graspable point is a small raised handle knob, not a
        # fruit-like body: measured live (2026-09) via the lid's own geoms
        # (BlenderLid's "..._handle_main" geom), it sits ~0.022m above the
        # lid's body origin (blender_lid_pos) and is only ~0.005m thick --
        # thinner than ScriptedPolicyBase's fruit-tuned defaults
        # (GRASP_HEIGHT_OFFSET=0.02, which undershoots the handle and lands
        # closer to the flat lid surface around its base; Z_DESCEND_
        # THRESHOLD=0.015, 3x the handle's own thickness, so DESCEND could
        # stop well off-center either way) -- confirmed live as "touches it
        # but doesn't hold". Tuned to the handle's actual measured height.
        #
        # z_descend_threshold=0.012, not tighter: an initial 0.008 attempt
        # got DESCEND stuck indefinitely on some layouts -- confirmed live
        # via raw contact-pair inspection (env.sim.data.contact) that the
        # gripper fingers WERE already touching the handle
        # ("..._handle_main" <-> "gripper0_right_fingerN_collision"),
        # exactly the contact we want before closing -- but OSC_POSE
        # position control leaves a small steady-state tracking error once
        # further motion is blocked by that contact, and 0.008 was tighter
        # than that residual ever got, so the DESCEND -> GRASP transition
        # (norm(err) < threshold) never fired. 0.012 is still tighter than
        # ScriptedPolicyBase's fruit-tuned default (0.015) but loose enough
        # to accept "already in contact with the handle" as arrived.
        #
        # obstacle_clear_z too: the lid sits ON the blender, so reaching it
        # from a low starting position (e.g. right after a fresh reset) can
        # also drag along/get stuck against the blender's side on the way
        # up -- confirmed live (2026-09): one grasp attempt stuck in
        # contact for 178/200 steps outright.
        pick = PickObjectMotion(
            obj_pos_key="blender_lid_pos",
            grasp_height_offset=0.022,
            z_descend_threshold=0.012,
            obstacle_clear_z=self._last_obs.get("dest_top_z"),
        )
        pick.on_episode_start()
        if not self._run_motion(pick):
            return False
        place = TransportReleaseMotion(target_pos_key="lid_drop_pos")
        place.on_episode_start()
        # No self.blender_open = True to set here: the next call re-checks
        # lid_on_blender fresh from self._last_obs instead (updated every
        # step by _run_motion), so this policy's own belief can't go stale
        # or diverge from what actually happened physically.
        return self._run_motion(place)


def main(args=None):
    rclpy.init(args=args)
    bridge = RipeFruitBridge()
    bridge.load_configuration()
    executor = MultiThreadedExecutor(num_threads=4)
    try:
        rclpy.spin(bridge, executor=executor)
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected: Shutting down ripe_fruit_bridge...")
    finally:
        bridge.agent_bridge.close()
        bridge.destroy_node()


if __name__ == "__main__":
    main()
