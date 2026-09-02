#!/usr/bin/env python3
"""Gym-like ROS 2 client for the emdb simulator's RL interface.

Requires the simulator's scene_loader node to be running with
control_mode:=rl (see emdb_simulator). Observations and rewards arrive
asynchronously on /observations and /reward; actions are sent through the
blocking /step_action service, which only returns once the sim has stepped
physics and published the matching topic messages.
"""
import threading
import time
from collections import OrderedDict

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from emdb_interfaces.msg import Observation, StepInfo
from emdb_interfaces.srv import (
    MarkObjectRotten,
    ResetEpisode,
    StepAction,
    StepActionRaw,
)


def observation_to_dict(msg: Observation):
    """Rebuild a RoboCasa/robosuite-style obs_dict from a flattened Observation."""
    obs = OrderedDict()
    for entry in msg.entries:
        # scene_loader's _publish_observation (the only producer) always
        # sets entry.shape accurately from the source array's own .shape,
        # including an empty list for a true scalar (0-d) value -- trust
        # it directly instead of falling back to (len(entry.data),), which
        # silently turns any scalar entry back into a shape-(1,) array
        # instead of a real scalar (confirmed live: broke arithmetic on a
        # scalar "<name>_top_z" entry, e.g. `np.array([x, y, top_z_plus])`
        # raising "inhomogeneous shape" because top_z_plus was a (1,)
        # array, not a float).
        obs[entry.key] = np.array(entry.data, dtype=np.float64).reshape(tuple(entry.shape))
    return obs


class SceneLoaderUnavailableError(RuntimeError):
    """Raised when a service call fails because its server is gone (scene_loader
    died), not because it was merely slow. Distinct from TimeoutError, which
    callers may treat as transient/retryable (e.g. a slow hard_reset under CPU
    contention); this never self-heals within the same job."""


class AgentBridge(Node):
    def __init__(self, node_name="emdb_agent_bridge", step_timeout_sec=10.0, reset_timeout_sec=90.0):
        super().__init__(node_name)

        # RoboCasa envs do a hard_reset (procedural scene rebuild) on every
        # env.reset(), which takes as long as the initial scene load (~10s+).
        # A single physics step is cheap by comparison, so these get very
        # different budgets.
        self._step_timeout_sec = step_timeout_sec
        self._reset_timeout_sec = reset_timeout_sec
        self._cv = threading.Condition()
        self._latest_obs = None
        self._latest_step_info = None
        self._gripper_closed = False
        self._executor = None
        self._spin_thread = None

        self.create_subscription(Observation, "/observations", self._on_observation, 10)
        self.create_subscription(StepInfo, "/reward", self._on_step_info, 10)

        self._step_cli = self.create_client(StepAction, "/step_action")
        self._step_raw_cli = self.create_client(StepActionRaw, "/step_action_raw")
        self._reset_cli = self.create_client(ResetEpisode, "/reset_episode")
        self._mark_rotten_cli = self.create_client(MarkObjectRotten, "/mark_object_rotten")
        self._mark_episode_success_cli = self.create_client(Trigger, "/mark_episode_success")

    def start(self):
        """Spin this node on a background thread so blocking step()/reset() calls work."""
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def close(self):
        if self._executor is not None:
            self._executor.shutdown()
        self.destroy_node()

    def wait_for_services(self, timeout_sec=300.0):
        clients = (
            (self._step_cli, "/step_action"),
            (self._step_raw_cli, "/step_action_raw"),
            (self._reset_cli, "/reset_episode"),
            (self._mark_rotten_cli, "/mark_object_rotten"),
            (self._mark_episode_success_cli, "/mark_episode_success"),
        )
        for cli, name in clients:
            if not cli.wait_for_service(timeout_sec=timeout_sec):
                raise RuntimeError(f"Service {name} not available after {timeout_sec}s")

    def _on_observation(self, msg):
        with self._cv:
            self._latest_obs = msg
            self._cv.notify_all()

    def _on_step_info(self, msg):
        with self._cv:
            self._latest_step_info = msg
            self._cv.notify_all()

    def _call(self, client, request, name, timeout_sec):
        # Distinguish "server gone" from "server slow": a dead scene_loader
        # will never answer no matter how long we wait, so detect that
        # explicitly (via the cheap, non-blocking service_is_ready() check)
        # instead of always raising a plain TimeoutError -- callers can then
        # tell a permanent failure apart from a transient one worth retrying.
        if not client.service_is_ready():
            raise SceneLoaderUnavailableError(
                f"Service {name} is not available -- scene_loader appears to have died"
            )

        event = threading.Event()
        holder = {}

        def _on_done(future):
            holder["response"] = future.result()
            event.set()

        client.call_async(request).add_done_callback(_on_done)
        if not event.wait(timeout=timeout_sec):
            if not client.service_is_ready():
                raise SceneLoaderUnavailableError(
                    f"Service call to {name} timed out after {timeout_sec}s and the "
                    "service is no longer available -- scene_loader appears to have "
                    "died mid-call"
                )
            raise TimeoutError(f"Service call to {name} timed out after {timeout_sec}s")
        return holder["response"]

    def _wait_for_step(self, episode_id, step_id, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        with self._cv:
            while True:
                obs_ready = (
                    self._latest_obs is not None
                    and self._latest_obs.episode_id == episode_id
                    and self._latest_obs.step_id == step_id
                )
                info_ready = (
                    self._latest_step_info is not None
                    and self._latest_step_info.episode_id == episode_id
                    and self._latest_step_info.step_id == step_id
                )
                if obs_ready and info_ready:
                    return observation_to_dict(self._latest_obs), self._latest_step_info
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for /observations + /reward at "
                        f"episode_id={episode_id} step_id={step_id}"
                    )
                self._cv.wait(timeout=remaining)

    def reset(self, layout_id=-1, style_id=-1):
        request = ResetEpisode.Request()
        request.layout_id = layout_id
        request.style_id = style_id

        response = self._call(
            self._reset_cli, request, "/reset_episode", self._reset_timeout_sec
        )
        if not response.success:
            raise RuntimeError(f"/reset_episode failed: {response.message}")

        self._gripper_closed = False
        obs, _info = self._wait_for_step(response.episode_id, 0, self._reset_timeout_sec)
        return obs

    def mark_object_rotten(self, object_name):
        """Recolor object_name's mesh brown in place (see scene_loader.py's
        /mark_object_rotten -- mirrors RoboCasa's ChooseRipeFruit texture
        hack). Call this only once the sim has revealed the outcome (e.g.
        FruitShop's scale) so the recolor doesn't leak the answer early."""
        request = MarkObjectRotten.Request()
        request.object_name = object_name

        response = self._call(
            self._mark_rotten_cli, request, "/mark_object_rotten", self._step_timeout_sec
        )
        if not response.success:
            raise RuntimeError(f"/mark_object_rotten failed: {response.message}")

    def mark_episode_success(self):
        """Tell scene_loader.py this episode really succeeded (see its
        /mark_episode_success -- for tasks like FruitShop whose real
        success isn't computed by env._check_success(), OR'd into the
        success value it feeds video_recorder/StepInfo.success)."""
        request = Trigger.Request()

        response = self._call(
            self._mark_episode_success_cli, request, "/mark_episode_success", self._step_timeout_sec
        )
        if not response.success:
            raise RuntimeError(f"/mark_episode_success failed: {response.message}")

    def step(self, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0,
             base_dx=0.0, base_dy=0.0, base_dyaw=0.0, grasp=0, next_arm=0, next_robot=0):
        request = StepAction.Request()
        request.dx = dx
        request.dy = dy
        request.dz = dz
        request.droll = droll
        request.dpitch = dpitch
        request.dyaw = dyaw
        request.base_dx = base_dx
        request.base_dy = base_dy
        request.base_dyaw = base_dyaw
        request.grasp = grasp
        request.next_arm = next_arm
        request.next_robot = next_robot

        response = self._call(self._step_cli, request, "/step_action", self._step_timeout_sec)
        if not response.success:
            raise RuntimeError(f"/step_action failed: {response.message}")

        obs, info = self._wait_for_step(
            response.episode_id, response.step_id, self._step_timeout_sec
        )
        return obs, info.reward, info.terminated, info.truncated, {"success": info.success}

    def step_raw(self, action_vector):
        """Step with a raw action vector, bypassing the teleop delta translation.

        `action_vector` must already be in the sim's native env.step() action
        space (e.g. a recorded demo action, or output of robot.create_action_vector).
        Unlike step()/step_vector(), this does not go through the ROSKeyboardDevice,
        so gripper/base-mode fields here are absolute, not toggles.
        """
        request = StepActionRaw.Request()
        request.action = np.asarray(action_vector, dtype=np.float64).flatten().tolist()

        response = self._call(
            self._step_raw_cli, request, "/step_action_raw", self._step_timeout_sec
        )
        if not response.success:
            raise RuntimeError(f"/step_action_raw failed: {response.message}")

        obs, info = self._wait_for_step(
            response.episode_id, response.step_id, self._step_timeout_sec
        )
        return obs, info.reward, info.terminated, info.truncated, {"success": info.success}

    def step_vector(self, action_vector, grasp_threshold=0.0):
        """Step with a flat [dx, dy, dz, droll, dpitch, dyaw, gripper] action.

        Matches the default single-arm OSC_POSE + gripper layout a RoboCasa/
        robomimic policy would output. `gripper` is an absolute open(<=0)/
        closed(>0) command; the underlying sim only accepts a toggle, so this
        tracks the last commanded gripper state and toggles only on change.
        """
        action_vector = np.asarray(action_vector, dtype=np.float64).flatten()
        if action_vector.shape[0] < 7:
            raise ValueError(
                "Expected a 7-dim [dx,dy,dz,droll,dpitch,dyaw,gripper] action "
                f"vector, got shape {action_vector.shape}"
            )
        dx, dy, dz, droll, dpitch, dyaw, gripper = action_vector[:7]

        want_closed = bool(gripper > grasp_threshold)
        grasp_toggle = int(want_closed != self._gripper_closed)
        if grasp_toggle:
            self._gripper_closed = want_closed

        return self.step(
            dx=dx, dy=dy, dz=dz, droll=droll, dpitch=dpitch, dyaw=dyaw, grasp=grasp_toggle
        )
