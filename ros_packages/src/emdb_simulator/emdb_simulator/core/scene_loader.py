#!/usr/bin/env python3
import json
import os
import queue
import re
import threading
import time
from concurrent.futures import Future
from copy import deepcopy
from collections import OrderedDict
from glob import glob
import traceback

import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger

import robosuite
import robocasa
import robocasa.utils.object_utils as OU
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.wrappers.enclosing_wall_render_wrapper import (
    EnclosingWallRenderWrapper,
    install_enclosing_wall_hotkeys,
)
from robosuite.wrappers import DataCollectionWrapper
from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format

from emdb_interfaces.srv import (
    SetDeltaAction,
    StepAction,
    StepActionRaw,
    ResetEpisode,
    SaveDemos,
    MarkObjectRotten,
)
from emdb_simulator.core import registered_robots  # noqa: F401  registers all custom robots
from emdb_simulator.core import registered_tasks  # noqa: F401  registers all custom tasks
from emdb_simulator.core.ros_keyboard_device import ROSKeyboardDevice
from emdb_simulator.core.camera_config import load_custom_cameras
from emdb_simulator.core.video_recorder import (
    EpisodeRecordingSpec,
    VideoRecorder,
)
from emdb_interfaces.msg import (
    ObjectState,
    ObjectStateArray,
    Observation,
    ObservationEntry,
    StepInfo,
)

# FruitShop's scene is entirely counter-based (fruit/scale/accepted/rejected
# baskets, see fruit_shop_task.py), so unlike scene_loader's generic
# layout_id=-1 (uniform over all 60 layouts, most without an island), it
# specifically wants the extra counter surface a kitchen island provides.
# Curated subset of layouts NOT in Kitchen.ISLAND_EXCLUDED_LAYOUTS
# (robocasa/environments/kitchen/kitchen.py) -- randomized per episode in
# _set_layout_style for scene diversity.
FRUIT_SHOP_LAYOUT_IDS = (2, 4, 12, 21, 23)



class SceneLoader(Node):
    def __init__(self):
        super().__init__("robocasa_rollout_node")

        self.declare_parameter("task", "PickPlaceCounterToCabinet")
        self.declare_parameter("robot", "UR5eOmron")
        # layout 12 has a kitchen island (layout 11, the previous default,
        # is in Kitchen.ISLAND_EXCLUDED_LAYOUTS and has none), which KitchenLift
        # needs since it spawns the robot and object on the island.
        self.declare_parameter("layout_id", -1)
        self.declare_parameter("style_id", -1)
        self.declare_parameter("show_walls", False)
        self.declare_parameter("renderer", "mjviewer")
        # Skips the per-step self.env.render() call (the on-screen mjviewer
        # window) regardless of control_mode -- independent of
        # record_video/has_offscreen_renderer, which is a separate,
        # unrelated render path. Off by default so nobody's current
        # workflow changes; turn on for a real rl-mode training run where
        # nobody's watching the window, off (default) to debug visually.
        self.declare_parameter("headless", False)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("control_mode", "teleop")
        self.declare_parameter("collect_demos", False)
        self.declare_parameter("demo_dir", "/tmp/emdb_demos")
        self.declare_parameter("perception_mode", "unified")
        self.declare_parameter("record_video", False)
        self.declare_parameter("record_video_dir", "/tmp/emdb_videos")
        self.declare_parameter("record_video_episodes", "all")
        self.declare_parameter("record_video_camera", "robot0_agentview_center")
        self.declare_parameter("record_video_fps", -1.0)
        self.declare_parameter("record_video_width", 1280)
        self.declare_parameter("record_video_height", 720)
        self.declare_parameter("record_video_stride", 1)
        self.declare_parameter("record_video_crf", 18)
        self.declare_parameter("record_video_keep_successes", False)
        self.declare_parameter("preview_camera", False)
        self.declare_parameter("preview_camera_names", "all")
        self.declare_parameter("custom_cameras_file", "")
        self.declare_parameter("env_seed", -1)
        # Empty (default) means "use whatever DEFAULT_OBJ_GROUPS the task class
        # already has" -- comma-separated category names (e.g. "apple" or
        # "apple,banana") override it per-run without editing task source, for
        # tasks whose obj_groups is a constructor kwarg (KitchenLift, FruitShop).
        self.declare_parameter("obj_groups", "")

        self.task = self.get_parameter("task").value
        self.robot = self.get_parameter("robot").value
        self.layout_id = int(self.get_parameter("layout_id").value)
        self.style_id = int(self.get_parameter("style_id").value)
        obj_groups_param = self.get_parameter("obj_groups").value
        self.obj_groups = (
            [g.strip() for g in obj_groups_param.split(",") if g.strip()]
            if obj_groups_param
            else None
        )
        self.show_walls = bool(self.get_parameter("show_walls").value)
        self.renderer = self.get_parameter("renderer").value
        self.headless = bool(self.get_parameter("headless").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.control_mode = self.get_parameter("control_mode").value
        self.collect_demos = bool(self.get_parameter("collect_demos").value)
        self.demo_dir = self.get_parameter("demo_dir").value
        self.demo_tmp_directory = None
        self.env_info = None
        # Per-step [obj_x, obj_y, obj_z, grasped] for the "obj" object,
        # captured live during teleop recording (see
        # _apply_env_action_and_publish) and keyed by DataCollectionWrapper's
        # ep_directory for that episode. Merged into demo.hdf5 by
        # _save_demos_cb so downstream consumers (emdb_policy's
        # prepare_lift_demo_episodes) don't have to reconstruct/replay it.
        self._demo_perception = {}

        self.record_video = bool(self.get_parameter("record_video").value)
        self.record_video_dir = self.get_parameter("record_video_dir").value
        self.record_video_episodes = self.get_parameter("record_video_episodes").value
        self.record_video_camera = self.get_parameter("record_video_camera").value
        self.record_video_fps = float(self.get_parameter("record_video_fps").value)
        self.record_video_width = int(self.get_parameter("record_video_width").value)
        self.record_video_height = int(self.get_parameter("record_video_height").value)
        self.record_video_stride = int(self.get_parameter("record_video_stride").value)
        self.record_video_crf = int(self.get_parameter("record_video_crf").value)
        self.record_video_keep_successes = bool(
            self.get_parameter("record_video_keep_successes").value
        )
        self.preview_camera = bool(self.get_parameter("preview_camera").value)
        self.preview_camera_names = self.get_parameter("preview_camera_names").value
        self.custom_cameras_file = self.get_parameter("custom_cameras_file").value
        self.custom_cameras = load_custom_cameras(
            self.custom_cameras_file, logger=self.get_logger()
        )
        # -1 (default) means unseeded -- RoboCasa's Kitchen(seed=None) draws
        # from system entropy, so object placement AND the robot's start
        # pose/facing (both driven by env.rng, see RoboCasa's
        # compute_robot_base_placement_pose) vary on every single reset, not
        # just across layout/style choices. Pin a seed for reproducible
        # testing (e.g. always spawning the robot facing the object the same
        # way) without affecting anyone who leaves this at -1.
        env_seed = int(self.get_parameter("env_seed").value)
        self.env_seed = env_seed if env_seed >= 0 else None

        self.video_recorder = VideoRecorder(
            enabled=self.record_video,
            output_dir=self.record_video_dir,
            episode_spec=EpisodeRecordingSpec.parse(
                self.record_video_episodes, logger=self.get_logger()
            ),
            camera_name=self.record_video_camera,
            width=self.record_video_width,
            height=self.record_video_height,
            fps=(
                self.record_video_fps
                if self.record_video_fps > 0
                else self.publish_rate / max(1, self.record_video_stride)
            ),
            stride=self.record_video_stride,
            crf=self.record_video_crf,
            keep_successes=self.record_video_keep_successes,
            logger=self.get_logger(),
        )

        # unified: single /object_states with every object (default).
        # grouped: one /object_states/<fixture_name> per fixture objects are placed on/in.
        # split: one /object_states/<object_name> per object.
        # mdb: e-MDB cognitive-architecture-compatible named perceptions --
        # one /emdb/simulator/sensor/<object_name> per object (like split),
        # a companion /emdb/simulator/sensor/<object_name>/grasped per
        # graspable object, and a single /emdb/simulator/sensor/progress
        # (sparse task-success signal, matching e-MDB's "reward is just
        # another perception" convention). See docs/source/architecture.md.
        self.perception_mode = self.get_parameter("perception_mode").value
        if self.perception_mode not in ("unified", "grouped", "split", "mdb"):
            self.get_logger().warning(
                f"Unknown perception_mode={self.perception_mode!r}, falling back to 'unified'."
            )
            self.perception_mode = "unified"
        self._dynamic_object_state_pubs = {}
        self._grasped_check_warned = False

        # env.reset()/env.step()/env.render() must only run on the thread
        # that created the MuJoCo/GLFW render context (this constructor's
        # thread -- see run_sim_loop() and main() below). ROS spins on a
        # separate thread, so every service/timer callback that touches
        # self.env enqueues its work here instead of calling it directly.
        self._sim_queue = queue.Queue()

        # All timers share this group, separate from services' default
        # group. Services block their callback (via _run_on_sim_thread)
        # until the sim thread answers, same as timers do -- under a
        # SingleThreadedExecutor with everything in one MutuallyExclusive
        # group, a continuously-firing timer (e.g. the rl perception loop)
        # starves service calls like /reset_episode indefinitely. main()
        # spins with a MultiThreadedExecutor so these two groups actually
        # run concurrently; _run_on_sim_thread's queue keeps the real MuJoCo
        # work serialized regardless.
        self._timer_cbgroup = MutuallyExclusiveCallbackGroup()

        # -1 = "no episode started yet". In teleop mode _create_env() below
        # bumps this to 0 immediately since the user is already driving the
        # robot; in rl mode it stays -1 until the client's first
        # /reset_episode call, so the client's own episode 0 lands on
        # episode_id 0 too instead of being shifted by a phantom episode.
        self.episode_id = -1
        self.step_id = 0
        # Cached from the last _apply_env_action_and_publish() / reset, so the
        # rl-mode heartbeat timer can re-publish it between steps without
        # re-stepping physics (see _mdb_perception_heartbeat).
        self._last_success = False
        # Set via /mark_episode_success by a driving bridge node whose real
        # success isn't computed by env._check_success() (e.g. FruitShop --
        # see fruit_shop_task.py's own docstring: its success/reward lives
        # in emdb_policy's fruit_shop_bridge.py, a separate process this
        # node has no visibility into otherwise). OR'd into the success
        # value _apply_env_action_and_publish computes, so video_recorder's
        # keep_successes mode (and StepInfo.success) reflect the task's
        # real outcome instead of always False for tasks like this one.
        self._external_success = False
        # Debounces "Success achieved!" in the teleop render loop to once per
        # episode (rising edge only), instead of once per render tick for as
        # long as _check_success() stays True.
        self._teleop_success_logged = False

        self.layouts = self._build_layouts()
        self.styles = self._build_styles()
        self.env = None
        self.robot_model = None

        self.robot_joint_names = []
        self.robot_qpos_idx = []
        self.robot_qvel_idx = []

        self.device = None
        self.all_prev_gripper_actions = None
        self.mirror_actions = True

        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)

        self.object_state_pub = (
            self.create_publisher(ObjectStateArray, "/object_states", 10)
            if self.perception_mode == "unified"
            else None
        )

        self.progress_pub = (
            self.create_publisher(Float32, "/emdb/simulator/sensor/progress", 10)
            if self.perception_mode == "mdb"
            else None
        )

        self.observation_pub = self.create_publisher(Observation, "/observations", 10)

        self.step_info_pub = self.create_publisher(StepInfo, "/reward", 10)

        self.set_delta_srv = self.create_service(
            SetDeltaAction,
            "/set_delta_action",
            self._set_delta_action_cb,
        )

        self.reset_env_srv = self.create_service(
            Trigger,
            "/reset_env",
            self._reset_env_cb,
        )

        self.step_action_srv = self.create_service(
            StepAction,
            "/step_action",
            self._step_action_cb,
        )

        self.step_action_raw_srv = self.create_service(
            StepActionRaw,
            "/step_action_raw",
            self._step_action_raw_cb,
        )

        self.reset_episode_srv = self.create_service(
            ResetEpisode,
            "/reset_episode",
            self._reset_episode_cb,
        )

        self.save_demos_srv = self.create_service(
            SaveDemos,
            "/save_demos",
            self._save_demos_cb,
        )

        self.mark_object_rotten_srv = self.create_service(
            MarkObjectRotten,
            "/mark_object_rotten",
            self._mark_object_rotten_cb,
        )

        self.mark_episode_success_srv = self.create_service(
            Trigger,
            "/mark_episode_success",
            self._mark_episode_success_cb,
        )

        self._create_env()
        # _create_env() already resolved layout/style before its own
        # robosuite.make()/env.reset() and set self.current_layout/style
        # to match -- calling _set_layout_style() again here would just
        # re-resolve a *different* value (for layout_id/style_id == -1)
        # and set it up for the *next* reset, while leaving
        # self.current_layout/style claiming a layout that hasn't
        # actually been loaded yet.
        self._init_robot_joint_mapping()
        self._init_device()

        self.get_logger().info(
            f"Scene loaded -> layout={self.current_layout}, style={self.current_style}, "
            f"perception_mode={self.perception_mode}"
        )
        try:
            self.get_logger().info(
                "Available MuJoCo cameras for record_video_camera: "
                f"{list(self.env.sim.model.camera_names)}"
            )
        except Exception:
            pass

        if self.preview_camera:
            self.timer = self.create_timer(
                1.0 / self.publish_rate, self._run_camera_preview_live,
                callback_group=self._timer_cbgroup)
        elif self.control_mode == "teleop":
            self.timer = self.create_timer(
                1.0 / self.publish_rate, self._render_loop,
                callback_group=self._timer_cbgroup)
        else:
            # Physics itself only advances on /step_action(_raw) calls, but
            # sensor topics still need to keep streaming between steps for
            # consumers that expect continuous perceptions (e.g. e-MDB's main
            # loop) rather than one-shot per-step values -- see
            # _mdb_perception_heartbeat.
            self.timer = self.create_timer(
                1.0 / self.publish_rate, self._mdb_perception_heartbeat
            )
            self.get_logger().info(
                "control_mode=rl -> periodic render loop disabled; physics "
                "only advances on /step_action calls, but sensor topics "
                f"still stream at {self.publish_rate} Hz via the heartbeat"
            )

    def _pump_viewer(self):
        """Refresh the on-screen window, if any. Must run on the thread that
        owns the render context (see _run_on_sim_thread).

        env.render() is a documented no-op for the mjviewer renderer (only
        env.step() pumps MjviewerRenderer.viewer.update() internally, see
        robosuite/renderers/viewer/mjviewer_renderer.py) -- dispatch the same
        way robosuite's own step() does (environments/base.py).
        """
        if self.headless or self.env.viewer is None:
            return
        if self.env.renderer == "mujoco":
            self.env.render()
        else:
            self.env.viewer.update()

    def _run_on_sim_thread(self, fn):
        """Run fn() on the thread that owns the MuJoCo/GLFW render context,
        blocking the caller until it completes.

        With the on-screen mjviewer renderer (has_renderer=True, the
        default headless=false), calling env.reset()/env.step()/env.render()
        from any thread other than the one that created the context hangs
        forever with no exception -- GLFW's context isn't thread-safe.
        Service/timer callbacks route through here instead of touching
        self.env directly. See docs/source/howto/run_simulator.md.
        """
        fut = Future()
        self._sim_queue.put((fn, fut))
        return fut.result()

    def run_sim_loop(self):
        """Drain queued env.reset()/step()/render() work on this thread.

        Must be called from the thread that constructed this node (where
        robosuite.make() created the render context) -- see main() below,
        which spins ROS on a separate background thread instead.
        """
        while rclpy.ok():
            try:
                fn, fut = self._sim_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)

    def _build_layouts(self):
        raw = dict((item.value, item.name.lower().capitalize()) for item in LayoutType)
        ordered = OrderedDict()
        for k in sorted(raw.keys()):
            if k < 0:
                continue
            ordered[int(k)] = raw[k]
        return ordered

    def _build_styles(self):
        raw = dict((item.value, item.name.lower().capitalize()) for item in StyleType)
        ordered = OrderedDict()
        for k in sorted(raw.keys()):
            if k < 0:
                continue
            ordered[int(k)] = raw[k]
        return ordered

    def _create_env(self):
        self.is_kitchen_task = issubclass(REGISTERED_ENVS[self.task], Kitchen)

        config = {
            "env_name": self.task,
            "robots": self.robot,
        }
        if self.is_kitchen_task:
            # Resolved *before* robosuite.make()/env.reset() below, not
            # left as -1 for robocasa's own internal random pick then
            # "corrected" via a _set_layout_style() call afterward -- that
            # ordering (used to be the pattern here) only ever updates
            # self.env.layout_and_style_ids for the *next* reset; it can't
            # retroactively change the layout a reset that already
            # happened just loaded. Confirmed live: with layout_id=-1, the
            # very first scene load used robocasa's own random layout
            # (e.g. "layout: 7" in scene_loader's own startup log) despite
            # _set_layout_style() claiming "layout=21" right after --
            # that 21 was real, just one reset too late.
            layout, style = self._resolve_layout_style()
            config["translucent_robot"] = False
            config["layout_ids"] = [layout]
            config["style_ids"] = [style]
            config["seed"] = self.env_seed
            if self.obj_groups is not None:
                config["obj_groups"] = self.obj_groups
            self.current_layout = layout
            self.current_style = style
            # custom_cameras is generic to any Kitchen-family task
            # (CustomCamerasMixin, camera_config.py) -- every task
            # registered in this package (KitchenLift, FruitShop) mixes
            # it in, so this doesn't need to special-case by task name.
            config["custom_cameras"] = self.custom_cameras

        self.get_logger().info(
            f"Initializing {'RoboCasa' if self.is_kitchen_task else 'robosuite'} scene..."
        )
        self.get_logger().info(json.dumps(config))

        self.env = robosuite.make(
            **config,
            has_renderer=not self.headless,
            has_offscreen_renderer=self.record_video,
            render_camera=self._first_preview_camera_name() if self.preview_camera else None,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=int(self.publish_rate),
            renderer=self.renderer,
        )

        if self.is_kitchen_task:
            self.env = EnclosingWallRenderWrapper(
                self.env, alpha=0.1, enabled=not self.show_walls
            )
            # Esc/[/] hotkeys toggle wall transparency in the on-screen
            # window -- meaningless (and one more long-lived Xlib
            # connection that can die under it) when nobody's watching one.
            if not self.headless:
                install_enclosing_wall_hotkeys(self.env)

        self.env_info = json.dumps(config)
        if self.collect_demos:
            t1, t2 = str(time.time()).split(".")
            self.demo_tmp_directory = f"/tmp/emdb_demo_raw_{t1}_{t2}"
            self.env = DataCollectionWrapper(
                self.env, self.demo_tmp_directory, use_env_xml_for_reset=True
            )
            self.get_logger().info(
                f"collect_demos=true -> recording teleop episodes to {self.demo_tmp_directory}"
            )

        self.env.reset()
        # Pump the window here (startup, on the thread that will own it) so
        # the on-screen viewer is already up by the time any client's first
        # /reset_episode call arrives. Without this, the mjviewer renderer's
        # mujoco.viewer.launch_passive() -- a real window/GL-context create,
        # not a cheap call -- happens lazily on whatever reset call is first,
        # making it look hung to that caller for several seconds.
        self._pump_viewer()
        if self.control_mode == "teleop":
            self.episode_id += 1
            self.video_recorder.maybe_start_episode(self.episode_id)

    def _resolve_layout_style(self):
        """Resolve self.layout_id/style_id (-1 meaning "pick one") into
        concrete ids. Shared by _create_env (before the very first
        robosuite.make()/env.reset()) and _set_layout_style (before every
        later reset) so a reset never runs one step behind whatever this
        just resolved -- see _create_env's comment for why that ordering
        matters."""
        layout = self.layout_id
        style = self.style_id

        if layout == -1:
            if self.task == "FruitShop":
                # Pinned to 21 (one of FRUIT_SHOP_LAYOUT_IDS) instead of
                # randomizing across the curated set -- keeps every episode
                # on the same island size/shape while iterating on reach
                # behavior, rather than reproducing across a moving target.
                # Pass -p layout_id:=<n> explicitly to override.
                layout = 21
            else:
                layout = int(np.random.choice(list(self.layouts.keys())))
        if style == -1:
            style = int(np.random.choice(list(self.styles.keys())))

        return layout, style

    def _set_layout_style(self):
        if not self.is_kitchen_task:
            self.current_layout = None
            self.current_style = None
            return

        layout, style = self._resolve_layout_style()
        self.env.layout_and_style_ids = [[layout, style]]
        self.current_layout = layout
        self.current_style = style

    def _init_robot_joint_mapping(self):
        if not hasattr(self.env, "robots") or len(self.env.robots) == 0:
            raise RuntimeError("No robots found in environment")

        self.robot_model = self.env.robots[0]

        if hasattr(self.robot_model, "robot_model") and hasattr(self.robot_model.robot_model, "joints"):
            self.robot_joint_names = list(self.robot_model.robot_model.joints)
        elif hasattr(self.robot_model, "joints"):
            self.robot_joint_names = list(self.robot_model.joints)
        elif hasattr(self.robot_model, "_ref_joint_indexes"):
            idxs = list(self.robot_model._ref_joint_indexes)
            self.robot_joint_names = [self.env.sim.model.joint_id2name(i) for i in idxs]
        else:
            raise RuntimeError("Could not determine robot joint names")

        sim_model = self.env.sim.model
        qpos_idx, qvel_idx, valid_names = [], [], []

        for name in self.robot_joint_names:
            try:
                qpos_addr = sim_model.get_joint_qpos_addr(name)
                qvel_addr = sim_model.get_joint_qvel_addr(name)
                if isinstance(qpos_addr, tuple) or isinstance(qvel_addr, tuple):
                    continue
                valid_names.append(name)
                qpos_idx.append(int(qpos_addr))
                qvel_idx.append(int(qvel_addr))
            except Exception:
                continue

        self.robot_joint_names = valid_names
        self.robot_qpos_idx = qpos_idx
        self.robot_qvel_idx = qvel_idx

        if len(self.robot_joint_names) == 0:
            raise RuntimeError("No valid 1-DoF robot joints found")

        self.get_logger().info(
            f"Joint mapping initialized with {len(self.robot_joint_names)} joints"
        )

    def _init_device(self):
        self.device = ROSKeyboardDevice(
            env=self.env,
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
        )
        self.device.start_control()

        self.all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in self.env.robots
        ]

    def _reset_env_cb(self, request, response):
        return self._run_on_sim_thread(lambda: self._reset_env_cb_impl(request, response))

    def _reset_env_cb_impl(self, request, response):
        try:
            # Resolve/apply the next layout+style *before* reset (like
            # _reset_episode_cb_impl already does), not after -- setting
            # self.env.layout_and_style_ids only affects the *next*
            # env.reset(), so calling it after this one just leaves the
            # scene that was actually (re)loaded one resolution behind.
            self._set_layout_style()
            # See _reset_episode_cb_impl's comment: close out the outgoing
            # episode's video before env.reset() rebuilds self.env.sim.
            self.video_recorder.close_episode()
            self.env.reset()
            self.device.start_control()
            self._init_device()
            self.episode_id += 1
            self.step_id = 0
            self.video_recorder.maybe_start_episode(self.episode_id)
            self._last_success = False
            self._external_success = False
            response.success = True
            response.message = "Environment reset successfully"
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to reset environment: {e}"
            self.get_logger().error(response.message)
        return response

    def _set_delta_action_cb(self, request, response):
        # self.device is also read/mutated by the sim thread (_render_loop,
        # _translate_delta_to_env_action), so route through it too rather
        # than racing those from the ROS callback thread.
        return self._run_on_sim_thread(lambda: self._set_delta_action_cb_impl(request, response))

    def _set_delta_action_cb_impl(self, request, response):
        if request.toggle_base_mode:
            self.device.toggle_base_mode()

        if request.grasp:
            self.device.toggle_grasp()

        if request.reset:
            self.device.trigger_reset()
            response.success = True
            response.message = "Reset solicitado"
            return response

        if hasattr(request, "next_arm") and request.next_arm:
            self.device.next_arm()

        if hasattr(request, "next_robot") and request.next_robot:
            self.device.next_robot()

        if self.device.base_mode:
            self.device.apply_delta(
                dx=request.base_dx,
                dy=request.base_dy,
                dyaw=request.base_dyaw,
            )
        else:
            self.device.apply_delta(
                dx=request.dx,
                dy=request.dy,
                dz=request.dz,
                droll=request.droll,
                dpitch=request.dpitch,
                dyaw=request.dyaw,
            )

        response.success = True
        response.message = (
            f"Delta recibido | robot={self.device.active_robot} "
            f"arm={self.device.active_arm} "
            f"base_mode={int(self.device.base_mode)} "
            f"grasp={int(self.device.grasp)}"
        )
        return response

    def _build_env_action(self, active_robot, input_ac_dict):
        action_dict = deepcopy(input_ac_dict)

        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError(f"Unsupported input_type: {controller_input_type}")

        env_action = [
            robot.create_action_vector(self.all_prev_gripper_actions[i])
            for i, robot in enumerate(self.env.robots)
        ]
        env_action[self.device.active_robot] = active_robot.create_action_vector(action_dict)

        for arm in active_robot.arms:
            key = f"{arm}_gripper"
            if key in action_dict:
                self.all_prev_gripper_actions[self.device.active_robot][key] = action_dict[key]

        return np.concatenate(env_action)

    def _translate_delta_to_env_action(self, request):
        if request.next_arm:
            self.device.next_arm()

        if request.next_robot:
            self.device.next_robot()

        if request.grasp:
            self.device.toggle_grasp()

        if self.device.base_mode:
            self.device.apply_delta(
                dx=request.base_dx,
                dy=request.base_dy,
                dyaw=request.base_dyaw,
            )
        else:
            self.device.apply_delta(
                dx=request.dx,
                dy=request.dy,
                dz=request.dz,
                droll=request.droll,
                dpitch=request.dpitch,
                dyaw=request.dyaw,
            )

        active_robot = self.env.robots[self.device.active_robot]
        input_ac_dict = self.device.input2action(mirror_actions=self.mirror_actions)
        if input_ac_dict is None:
            raise RuntimeError(
                "device.input2action() returned None (reset pending) during an RL step"
            )

        return self._build_env_action(active_robot, input_ac_dict)

    def _augment_obs_with_control_frame(self, obs_dict):
        """Add robot0_origin_ori (and dest_pos, for place-family tasks) to
        obs_dict so a policy client can turn a world-frame position error
        into the base-relative delta OSC_POSE actually expects.

        OSC_POSE's input_ref_frame is "base" (see robosuite's
        composite/basic.json): dx/dy/dz sent through /step_action get added
        to the controller's goal in its "origin" frame, not world. That
        origin is NOT the mobile base body -- robosuite's own
        CompositeController.get_controller_base_pose() docstring warns "this
        pose may likely differ from the robot base's pose": it's a
        per-arm-controller "<naming_prefix><part_name>_center" site pose,
        refreshed into part_controllers[arm].origin_ori every control step
        (Robot.control() -> composite_controller.update_state(), see
        robosuite/robots/robot.py). Published as the full 3x3 matrix (not a
        derived yaw): this site's local axes aren't guaranteed to be a pure
        rotation about world Z (e.g. an eef/arm-mount site convention can
        have its own Z pointing along the arm's facing direction rather than
        world-up), so a policy has to invert the whole matrix, not just an
        angle. dest_pos comes straight from RoboCasa's Fixture.pos when the
        task exposes one (e.g. KitchenPlace's self.dest).
        """
        robot = self.env.robots[0]
        origin_ori = robot.part_controllers[robot.arms[0]].origin_ori
        obs_dict["robot0_origin_ori"] = np.asarray(origin_ori, dtype=np.float64).flatten()
        # robot.base_pos/robot.base_ori (Robot.reset(), robosuite/robots/
        # robot.py:274-275) are a ONE-TIME snapshot taken at env.reset() --
        # never refreshed per control step, so they go stale the instant a
        # mobile base actually moves (confirmed empirically: they read the
        # spawn pose even after many base_dx steps). The live values come
        # from robosuite's own mobile-base sensors instead (MobileRobot.
        # _create_base_sensors, robosuite/robots/mobile_robot.py:301-317),
        # which query the base's "center" site directly every call -- same
        # site those sensors use, read the same way here rather than via
        # the (possibly stale/unrefreshed) observable cache.
        base_center_site = robot.robot_model.base.correct_naming("center")
        base_site_id = self.env.sim.model.site_name2id(base_center_site)
        obs_dict["robot_base_pos"] = np.asarray(
            self.env.sim.data.site_xpos[base_site_id], dtype=np.float64
        )
        # 3x3 world-rotation matrix (mirrors robot0_origin_ori's convention)
        # -- lets a policy client turn a world-frame position error into the
        # mobile base's own forward/side control frame, the same way
        # robot0_origin_ori does for the arm.
        obs_dict["robot_base_ori"] = np.asarray(
            self.env.sim.data.get_site_xmat(base_center_site), dtype=np.float64
        ).flatten()
        dest = getattr(self.env, "dest", None)
        if dest is not None:
            obs_dict["dest_pos"] = np.asarray(dest.pos, dtype=np.float64)

        # Generic pass-through for any task-exposed "<name>_pos" property
        # (e.g. FruitShop's collection_pos/placed_pos, both computed
        # fixture-relative since RoboCasa kitchens are procedurally laid out
        # per episode) -- same role as dest_pos above, generalized so new
        # tasks don't need their own scene_loader changes for fixed,
        # fixture-relative target zones.
        for zone_attr in getattr(self.env, "OBS_ZONE_ATTRS", ()):
            value = getattr(self.env, zone_attr, None)
            if value is not None:
                obs_dict[zone_attr] = np.asarray(value, dtype=np.float64)

        # Generic pass-through for any task-exposed spawned object's live
        # pose, published as "<name>_pos" (e.g. FruitShop's scale/accepted/
        # rejected). Unlike dest/OBS_ZONE_ATTRS above (static Fixture/
        # fixture-relative positions, cacheable), these are spawned objects
        # whose pose has to be read live from sim.data every step -- there's
        # no cached .pos to rely on.
        # Also publishes "<name>_top_z" (world Z of the object's own
        # top_site, e.g. a basket's rim) alongside "<name>_pos" -- a
        # policy releasing something *into* one of these (FruitShop's
        # accept_fruit/discard_fruit dropping a fruit into accepted/
        # rejected) needs to clear that top, not just the object's body
        # origin: some basket variants in the category are tall enough
        # that lowering to a small fixed offset above body_xpos alone
        # drives the gripper straight into the basket wall instead of
        # over its rim. top_offset (misc/robosuite/robosuite/models/
        # objects/objects.py's MujocoXMLObject) is the standard
        # robosuite/robocasa per-asset "{prefix}top_site" convention, so
        # this works for any object in OBS_LIVE_OBJECT_ATTRS, not just
        # baskets specifically.
        for obj_name in getattr(self.env, "OBS_LIVE_OBJECT_ATTRS", ()):
            if obj_name in getattr(self.env, "objects", {}):
                body_id = self.env.obj_body_id[obj_name]
                body_pos = np.asarray(self.env.sim.data.body_xpos[body_id], dtype=np.float64)
                obs_dict[f"{obj_name}_pos"] = body_pos
                top_offset_z = float(self.env.objects[obj_name].top_offset[2])
                obs_dict[f"{obj_name}_top_z"] = body_pos[2] + top_offset_z

        return obs_dict

    def _publish_observation(self, obs_dict):
        msg = Observation()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.episode_id = self.episode_id
        msg.step_id = self.step_id

        entries = []
        for key, value in obs_dict.items():
            arr = np.asarray(value)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            entry = ObservationEntry()
            entry.key = key
            entry.data = arr.astype(np.float64).flatten().tolist()
            entry.shape = [int(d) for d in arr.shape]
            entries.append(entry)

        msg.entries = entries
        self.observation_pub.publish(msg)

    def _publish_step_info(self, reward, terminated, truncated, success):
        msg = StepInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.episode_id = self.episode_id
        msg.step_id = self.step_id
        msg.reward = float(reward)
        msg.terminated = bool(terminated)
        msg.truncated = bool(truncated)
        msg.success = bool(success)
        self.step_info_pub.publish(msg)

    def _apply_env_action_and_publish(self, env_action):
        obs, reward, _done, _info = self.env.step(env_action)
        success = bool(self.env._check_success()) or self._external_success
        self._last_success = success
        self.step_id += 1

        self._publish_joint_states()
        self._publish_object_states()
        self._publish_observation(self._augment_obs_with_control_frame(obs))
        self._publish_step_info(reward, terminated=success, truncated=False, success=success)
        self._publish_progress(success)
        if self.collect_demos:
            self._capture_demo_perception()
        if not self.headless:
            self.env.render()
        self.video_recorder.capture_frame(self.env, success=success)

    def _capture_demo_perception(self):
        """Append this step's [obj_x, obj_y, obj_z, grasped] to
        self._demo_perception, keyed by DataCollectionWrapper's ep_directory
        for the current episode (set once the wrapper sees its first
        interaction -- see DataCollectionWrapper._start_new_episode()).

        Same computation _publish_object_states/_check_grasped already do
        for the "obj" object; this just also keeps a copy around so
        _save_demos_cb can merge it straight into demo.hdf5 later, instead
        of some downstream consumer having to replay the episode to
        reconstruct it (see kitchen_lift_task.py's DEFAULT_OBJ_GROUPS/
        _get_obj_cfgs -- KitchenLift's only object is always named "obj").
        """
        ep_directory = getattr(self.env, "ep_directory", None)
        if ep_directory is None:
            return  # DataCollectionWrapper hasn't seen this episode's first interaction yet
        # self.env.ep_directory is a full path (DataCollectionWrapper joins
        # it with self.directory); key by the basename to match what
        # os.listdir(directory) returns in _find_successful_episodes /
        # _ordered_successful_episode_dirs / _merge_demo_perception.
        ep_directory = os.path.basename(ep_directory)
        obj_model = getattr(self.env, "objects", {}).get("obj")
        if obj_model is None or not obj_model.joints:
            return
        qpos = self.env.sim.data.get_joint_qpos(obj_model.joints[0])
        grasped = self._check_grasped("obj")
        self._demo_perception.setdefault(ep_directory, []).append(
            [float(qpos[0]), float(qpos[1]), float(qpos[2]), float(grasped)]
        )

    def _mdb_perception_heartbeat(self):
        """Re-publish the current (cached) sensor state on a timer in rl mode.

        Physics only advances on /step_action(_raw), but consumers such as
        e-MDB's main loop expect perceptions to keep flowing continuously
        (like the teleop render loop does); without this, a perception read
        between steps times out and the client can deadlock waiting on a
        value that will only ever arrive on the next step it, itself, has to
        trigger. This does NOT call self.env.step() -- object poses are read
        live from sim.data (so an idle scene just re-publishes the same
        pose), and the reward/success streamed here is the last one computed
        by a real step (or False before the first reset).
        """
        if self.env is None or self.episode_id < 0:
            return
        try:
            self._publish_object_states()
            self._publish_progress(self._last_success)
        except Exception as e:
            self.get_logger().error(f"mdb perception heartbeat failed: {e}")

    def _step_action_cb(self, request, response):
        return self._run_on_sim_thread(lambda: self._step_action_cb_impl(request, response))

    def _step_action_cb_impl(self, request, response):
        if self.control_mode != "rl":
            response.success = False
            response.message = "control_mode is 'teleop'; /step_action is only available in 'rl' mode"
            response.episode_id = self.episode_id
            response.step_id = self.step_id
            return response

        if self.episode_id < 0:
            # -1 = no /reset_episode call yet (see __init__); episode_id is a
            # uint64 field, so sending -1 through crashes message encoding.
            response.success = False
            response.message = "No episode started yet; call /reset_episode first"
            response.step_id = self.step_id
            return response

        try:
            env_action = self._translate_delta_to_env_action(request)
            self._apply_env_action_and_publish(env_action)
            response.success = True
            response.message = "ok"
        except Exception as e:
            self.get_logger().error(f"/step_action failed: {e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Step failed: {e}"

        response.episode_id = self.episode_id
        response.step_id = self.step_id
        return response

    def _step_action_raw_cb(self, request, response):
        return self._run_on_sim_thread(lambda: self._step_action_raw_cb_impl(request, response))

    def _step_action_raw_cb_impl(self, request, response):
        if self.control_mode != "rl":
            response.success = False
            response.message = (
                "control_mode is 'teleop'; /step_action_raw is only available in 'rl' mode"
            )
            response.episode_id = self.episode_id
            response.step_id = self.step_id
            return response

        if self.episode_id < 0:
            response.success = False
            response.message = "No episode started yet; call /reset_episode first"
            response.step_id = self.step_id
            return response

        try:
            env_action = np.asarray(request.action, dtype=np.float64)
            self._apply_env_action_and_publish(env_action)
            response.success = True
            response.message = "ok"
        except Exception as e:
            self.get_logger().error(f"/step_action_raw failed: {e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Step failed: {e}"

        response.episode_id = self.episode_id
        response.step_id = self.step_id
        return response

    def _reset_episode_cb(self, request, response):
        return self._run_on_sim_thread(lambda: self._reset_episode_cb_impl(request, response))

    def _reset_episode_cb_impl(self, request, response):
        try:
            if request.layout_id != -1:
                self.layout_id = int(request.layout_id)
            if request.style_id != -1:
                self.style_id = int(request.style_id)

            self._set_layout_style()
            # Finalize the outgoing episode's video *before* env.reset() --
            # a deferred (keep_successes) episode replays its buffered
            # states straight into self.env.sim (see
            # VideoRecorder._finish_deferred_episode), which only works
            # while self.env.sim still matches the model those states were
            # captured from. FruitShop's env.reset() does a hard_reset
            # (fresh procedural layout, often a different total qpos size
            # -- e.g. a different set of jointed fixtures), so closing
            # after reset (as maybe_start_episode()'s own internal
            # close_episode() call would) replays into the wrong-shaped
            # model and fails with a qpos broadcast error.
            self.video_recorder.close_episode()
            obs = self.env.reset()
            self.device.start_control()
            self.all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in self.env.robots
            ]

            self.episode_id += 1
            self.step_id = 0
            self.video_recorder.maybe_start_episode(self.episode_id)

            self._publish_joint_states()
            self._publish_object_states()
            self._publish_observation(self._augment_obs_with_control_frame(obs))
            self._publish_step_info(reward=0.0, terminated=False, truncated=False, success=False)
            self._publish_progress(False)
            self._last_success = False
            self._external_success = False

            response.success = True
            response.message = (
                f"Episode reset -> layout={self.current_layout}, style={self.current_style}"
            )
        except Exception as e:
            self.get_logger().error(f"/reset_episode failed: {e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Reset failed: {e}"

        # episode_id is a uint64 field; self.episode_id is still -1 here only
        # if env.reset() itself failed on the very first-ever reset attempt.
        response.episode_id = max(self.episode_id, 0)
        return response

    def _find_successful_episodes(self, directory):
        successful = []
        for ep_directory in os.listdir(directory):
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            success = False
            for state_file in sorted(glob(state_paths)):
                dic = np.load(state_file, allow_pickle=True)
                success = success or bool(dic["successful"])
            if success:
                successful.append(ep_directory)
        return successful

    def _ordered_successful_episode_dirs(self, directory, successful_episodes):
        """Replicates gather_demonstrations_as_hdf5's own directory
        iteration order and non-empty-episode filter (robocasa/scripts/
        collect_demos.py, vendored, not modified here) so this episode's
        directory name can be matched to the demo_N group index that
        function assigns it -- gather_demonstrations_as_hdf5 numbers output
        groups purely by iteration order (num_eps += 1 per processed
        episode) and never stores the source directory name anywhere
        recoverable in the output hdf5.

        Only reliable if called in the same process, right around the real
        call, with nothing else writing to `directory` in between (true
        here: both calls happen back-to-back inside _save_demos_cb).
        """
        ordered = []
        for ep_directory in os.listdir(directory):
            if ep_directory not in successful_episodes:
                continue
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            n_states = sum(
                len(np.load(state_file, allow_pickle=True)["states"])
                for state_file in sorted(glob(state_paths))
            )
            if n_states == 0:
                continue
            ordered.append(ep_directory)
        return ordered

    def _merge_demo_perception(self, hdf5_path, ordered_dirs):
        """Adds mdb_obj_xyz/mdb_grasped datasets (captured live during
        teleop, see _capture_demo_perception) to each demo_N group, so
        downstream consumers (emdb_policy's prepare_lift_demo_episodes) get
        ground-truth perception instead of having to replay the episode to
        reconstruct it.
        """
        with h5py.File(hdf5_path, "a") as f:
            for i, ep_directory in enumerate(ordered_dirs, start=1):
                grp = f.get(f"data/demo_{i}")
                perception = self._demo_perception.get(ep_directory)
                if grp is None or not perception:
                    self.get_logger().warning(
                        f"No captured perception for demo_{i} (dir={ep_directory}); "
                        "it will be missing mdb_obj_xyz/mdb_grasped."
                    )
                    continue
                arr = np.array(perception, dtype=np.float64)
                # No trim here (unlike gather_demonstrations_as_hdf5's own
                # del states[-1]): robocasa's `states` array gets one entry
                # *before* the first action too (captured in
                # DataCollectionWrapper._on_first_interaction, then one more
                # per action), hence one extra at the end relative to
                # `actions`. Our capture (_capture_demo_perception) only
                # appends *after* each action -- one entry per action,
                # already the same length as `actions`, nothing to drop.
                n_actions = grp["actions"].shape[0]
                if arr.shape[0] != n_actions:
                    self.get_logger().warning(
                        f"demo_{i}: captured {arr.shape[0]} perception steps but "
                        f"{n_actions} actions; skipping merge for it."
                    )
                    continue
                for name in ("mdb_obj_xyz", "mdb_grasped"):
                    if name in grp:
                        del grp[name]
                grp.create_dataset("mdb_obj_xyz", data=arr[:, :3])
                grp.create_dataset("mdb_grasped", data=arr[:, 3])

    def _save_demos_cb(self, request, response):
        if not self.collect_demos:
            response.success = False
            response.message = "collect_demos param is false; no episodes are being recorded"
            response.hdf5_path = ""
            return response

        try:
            out_dir = request.out_dir if request.out_dir else self.demo_dir
            os.makedirs(out_dir, exist_ok=True)

            successful_episodes = self._find_successful_episodes(self.demo_tmp_directory)
            ordered_dirs = self._ordered_successful_episode_dirs(
                self.demo_tmp_directory, successful_episodes
            )
            hdf5_path = gather_demonstrations_as_hdf5(
                self.demo_tmp_directory,
                out_dir,
                self.env_info,
                successful_episodes=successful_episodes,
                verbose=True,
            )

            if hdf5_path is None:
                response.success = False
                response.message = "No successful episodes recorded yet"
                response.hdf5_path = ""
                return response

            convert_to_robomimic_format(hdf5_path)
            self._merge_demo_perception(hdf5_path, ordered_dirs)

            response.success = True
            response.message = (
                f"Saved {len(successful_episodes)} successful demo(s) to {hdf5_path}"
            )
            response.hdf5_path = hdf5_path
            self.get_logger().info(response.message)
        except Exception as e:
            self.get_logger().error(f"/save_demos failed: {e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Failed to save demos: {e}"
            response.hdf5_path = ""

        return response

    def _mark_object_rotten_cb(self, request, response):
        return self._run_on_sim_thread(lambda: self._mark_object_rotten_cb_impl(request, response))

    def _mark_object_rotten_cb_impl(self, request, response):
        # Brown recolor, mirroring RoboCasa's ChooseRipeFruit._update_fruit_texture()
        # (misc/robocasa/robocasa/environments/kitchen/composite/making_juice/
        # choose_ripe_fruit.py) -- direct in-place geom_rgba mutation, no
        # separate "rotten" mesh/asset exists.
        name = request.object_name
        if self.env is None or name not in getattr(self.env, "objects", {}):
            response.success = False
            response.message = f"no object named {name!r} in the current scene"
            return response

        try:
            obj = self.env.objects[name]
            for geom in obj.visual_geoms:
                geom_id = self.env.sim.model.geom_name2id(geom)
                self.env.sim.model.geom_rgba[geom_id, :3] = (0.4, 0.26, 0.14)
                prev_alpha = self.env.sim.model.geom_rgba[geom_id, 3]
                self.env.sim.model.geom_rgba[geom_id, 3] = min(prev_alpha * 2, 1)
            response.success = True
            response.message = "ok"
        except Exception as e:
            self.get_logger().error(f"/mark_object_rotten failed: {e}")
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = f"Failed to mark {name!r} rotten: {e}"

        return response

    def _mark_episode_success_cb(self, request, response):
        return self._run_on_sim_thread(lambda: self._mark_episode_success_cb_impl(request, response))

    def _mark_episode_success_cb_impl(self, request, response):
        # Lets a driving bridge node (e.g. fruit_shop_bridge.py) report a
        # real success this node has no other way to see -- see
        # self._external_success's own comment in __init__. Also flag the
        # video recorder directly: this call typically lands *after* the
        # episode's last real _apply_env_action_and_publish() step (e.g.
        # fruit_shop_bridge resolves the episode, then just resets --  no
        # further env.step() happens), so capture_frame() -- the only
        # other place video_recorder learns about success -- would
        # otherwise never see success=True for this episode.
        self._external_success = True
        if getattr(self, "video_recorder", None) is not None:
            self.video_recorder.mark_success()
        response.success = True
        response.message = "ok"
        return response

    def _publish_joint_states(self):
        sim = self.env.sim
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot_joint_names)
        msg.position = [float(sim.data.qpos[i]) for i in self.robot_qpos_idx]
        msg.velocity = [float(sim.data.qvel[i]) for i in self.robot_qvel_idx]
        msg.effort = []
        self.joint_state_pub.publish(msg)

    def _object_fixture_group(self, name, cfgs_by_name, seen=None):
        """Resolve which fixture `name` was placed on/in, for perception_mode='grouped'.

        Follows placement["object"] references (e.g. an ice cube placed "on"
        a bowl) transitively until a real placement["fixture"] is found, or
        falls back to "unplaced" if none can be resolved.
        """
        seen = seen or set()
        if name in seen:
            return "unplaced"
        seen.add(name)

        cfg = cfgs_by_name.get(name)
        if cfg is None:
            return "unplaced"

        placement = cfg.get("placement") or {}
        fixture = placement.get("fixture")
        if fixture is not None and hasattr(fixture, "name"):
            return fixture.name

        ref_object = placement.get("object")
        if ref_object:
            return self._object_fixture_group(ref_object, cfgs_by_name, seen)

        return "unplaced"

    @staticmethod
    def _sanitize_topic_segment(name):
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
        if not safe or not safe[0].isalpha():
            safe = "g_" + safe
        return safe

    def _get_or_create_pub(self, topic, msg_type):
        pub = self._dynamic_object_state_pubs.get(topic)
        if pub is None:
            pub = self.create_publisher(msg_type, topic, 10)
            self._dynamic_object_state_pubs[topic] = pub
            self.get_logger().info(f"Created perception topic {topic}")
        return pub

    def _get_or_create_object_states_pub(self, topic_suffix, topic_prefix="/object_states"):
        return self._get_or_create_pub(f"{topic_prefix}/{topic_suffix}", ObjectStateArray)

    def _check_grasped(self, name):
        try:
            return bool(OU.check_obj_grasped(self.env, name))
        except Exception as exc:
            if not self._grasped_check_warned:
                self.get_logger().warning(
                    f"OU.check_obj_grasped failed for object {name!r} ({exc}); "
                    "reporting grasped=False for it (and any future failures) going forward."
                )
                self._grasped_check_warned = True
            return False

    def _publish_progress(self, success):
        if self.progress_pub is None:
            return
        msg = Float32()
        msg.data = float(success)
        self.progress_pub.publish(msg)

    def _publish_object_states(self):
        sim = self.env.sim
        stamp = self.get_clock().now().to_msg()

        # Read every object the task itself spawned (self.env.objects is
        # populated generically by Kitchen for any task, keyed by the exact
        # `name` used in _get_obj_cfgs) rather than guessing from joint name
        # substrings -- composite tasks name objects semantically (e.g.
        # "ice_bowl", "glass_cup"), not just "obj"/"distr".
        object_states = {}
        for name, obj_model in getattr(self.env, "objects", {}).items():
            joints = obj_model.joints
            if not joints:
                continue
            joint_name = joints[0]
            qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
            if not isinstance(qpos_addr, tuple):
                continue
            qpos = sim.data.get_joint_qpos(joint_name)

            obj = ObjectState()
            obj.name = name
            obj.pose.position.x = float(qpos[0])
            obj.pose.position.y = float(qpos[1])
            obj.pose.position.z = float(qpos[2])
            obj.pose.orientation.w = float(qpos[3])
            obj.pose.orientation.x = float(qpos[4])
            obj.pose.orientation.y = float(qpos[5])
            obj.pose.orientation.z = float(qpos[6])

            object_states[name] = obj

        if self.perception_mode == "unified":
            msg = ObjectStateArray()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.objects = list(object_states.values())
            self.object_state_pub.publish(msg)
            return

        cfgs_by_name = None
        topic_prefix = "/object_states"

        if self.perception_mode == "split":
            groups = {name: [obj] for name, obj in object_states.items()}
        elif self.perception_mode == "mdb":
            groups = {name: [obj] for name, obj in object_states.items()}
            topic_prefix = "/emdb/simulator/sensor"
            cfgs_by_name = {
                cfg.get("name"): cfg for cfg in getattr(self.env, "object_cfgs", [])
            }
        else:  # grouped
            cfgs_by_name = {
                cfg.get("name"): cfg for cfg in getattr(self.env, "object_cfgs", [])
            }
            groups = {}
            for name, obj in object_states.items():
                key = self._object_fixture_group(name, cfgs_by_name)
                groups.setdefault(key, []).append(obj)

        for key, objs in groups.items():
            pub = self._get_or_create_object_states_pub(
                self._sanitize_topic_segment(key), topic_prefix
            )
            msg = ObjectStateArray()
            msg.header.stamp = stamp
            msg.header.frame_id = "world"
            msg.objects = objs
            pub.publish(msg)

        if self.perception_mode == "mdb":
            for name in object_states:
                cfg = cfgs_by_name.get(name) or {}
                if not cfg.get("graspable"):
                    continue
                grasped_pub = self._get_or_create_pub(
                    f"/emdb/simulator/sensor/{self._sanitize_topic_segment(name)}/grasped",
                    Bool,
                )
                grasped_msg = Bool()
                grasped_msg.data = self._check_grasped(name)
                grasped_pub.publish(grasped_msg)


    def _publish_perceptions_loop(self):
        self._run_on_sim_thread(self._publish_perceptions_loop_impl)

    def _publish_perceptions_loop_impl(self):
        # rl mode's periodic timer: publish current state only, never
        # env.step()/env.reset() -- physics still only advances on
        # /step_action, this just keeps sensor topics flowing between calls.
        self._publish_joint_states()
        self._publish_object_states()
        self._publish_progress(self.env._check_success())

    def _render_loop(self):
        self._run_on_sim_thread(self._render_loop_impl)

    def _render_loop_impl(self):
        try:
            active_robot = self.env.robots[self.device.active_robot]
            input_ac_dict = self.device.input2action(mirror_actions=self.mirror_actions)
            self.get_logger().debug(f"input_ac_dict={input_ac_dict}")
            sim_model = self.env.sim.model
            for i in range(sim_model.njnt):
                name = sim_model.joint_id2name(i)
                if name and ("obj" in name or "distr" in name):
                    self.get_logger().debug(f"Joint {i}: {name}")
            if self.env._check_success():
                if not self._teleop_success_logged:
                    self.get_logger().info("Success achieved!")
                    self._teleop_success_logged = True
            else:
                self._teleop_success_logged = False

            if input_ac_dict is None:
                # See _reset_episode_cb_impl's comment: close out the
                # outgoing episode's video before env.reset() rebuilds
                # self.env.sim.
                self.video_recorder.close_episode()
                self.env.reset()
                self.device.start_control()
                self.device.clear_reset()
                self.episode_id += 1
                self.step_id = 0
                self._teleop_success_logged = False
                self.video_recorder.maybe_start_episode(self.episode_id)
                return

            env_action = self._build_env_action(active_robot, input_ac_dict)
            self._apply_env_action_and_publish(env_action)

            if self.device._reset_state:
                # See _reset_episode_cb_impl's comment: close out the
                # outgoing episode's video before env.reset() rebuilds
                # self.env.sim.
                self.video_recorder.close_episode()
                self.env.reset()
                self.device.start_control()
                self.device.clear_reset()
                self.episode_id += 1
                self.step_id = 0
                self._teleop_success_logged = False
                self.video_recorder.maybe_start_episode(self.episode_id)

        except Exception as e:
            self.get_logger().error(f"Render / control failed: {e}")
            self.get_logger().error(traceback.format_exc())
            self.destroy_node()
            rclpy.shutdown()

    def _first_preview_camera_name(self):
        """Camera to fix the interactive mjviewer on for preview_camera.

        Called before self.env exists, so "all"/empty falls back to the
        default free camera instead of expanding to the model's camera list.
        """
        spec = (self.preview_camera_names or "").strip().lower()
        if spec in ("", "all", "*"):
            return None
        names = [name.strip() for name in self.preview_camera_names.split(",") if name.strip()]
        if len(names) > 1:
            self.get_logger().info(
                "preview_camera shows one fixed camera at a time; "
                f"using {names[0]!r} (first of {self.preview_camera_names!r})"
            )
        return names[0] if names else None

    def _run_camera_preview_live(self):
        self._run_on_sim_thread(self._run_camera_preview_live_impl)

    def _run_camera_preview_live_impl(self):
        try:
            self.env.step(np.zeros(self.env.action_dim))
        except Exception as e:
            self.get_logger().error(f"Camera preview render failed: {e}")
            self.get_logger().error(traceback.format_exc())
            self.destroy_node()
            rclpy.shutdown()

    def destroy_node(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        if getattr(self, "video_recorder", None) is not None:
            try:
                self.video_recorder.close()
            except Exception:
                pass
        super().destroy_node()


def _spin_or_shutdown(node):
    # An uncaught exception here (e.g. a callback bug outside its own
    # try/except) would otherwise kill just this daemon thread: rclpy stops
    # answering ANY service/topic forever with no crash and no further log --
    # the node becomes a silent zombie, exactly the failure mode this file's
    # threading fix exists to avoid. Shut down instead so run_sim_loop()'s
    # `while rclpy.ok()` notices and main() exits loudly.
    try:
        # MultiThreadedExecutor so the timers' callback group (perceptions
        # in rl mode, render loop in teleop) can't starve the services'
        # default group -- see the _timer_cbgroup comment in __init__.
        # _run_on_sim_thread's queue still serializes all actual MuJoCo
        # work onto run_sim_loop()'s single thread regardless.
        executor = MultiThreadedExecutor(num_threads=4)
        rclpy.spin(node, executor=executor)
    except Exception:
        node.get_logger().error("ROS spin thread crashed:\n" + traceback.format_exc())
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SceneLoader()
    # ROS spins on a background thread; this (main) thread -- which is
    # where robosuite.make() created the MuJoCo/GLFW render context --
    # stays free to run env.reset()/step()/render() via run_sim_loop().
    spin_thread = threading.Thread(target=_spin_or_shutdown, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run_sim_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
