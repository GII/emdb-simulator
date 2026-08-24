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

from emdb_interfaces.msg import Observation, StepInfo
from emdb_interfaces.srv import (
    MarkObjectRotten,
    ResetEpisode,
    SetDeltaAction,
    StepAction,
    StepActionRaw,
)


def observation_to_dict(msg: Observation):
    """Rebuild a RoboCasa/robosuite-style obs_dict from a flattened Observation."""
    obs = OrderedDict()
    for entry in msg.entries:
        shape = tuple(entry.shape) if len(entry.shape) > 0 else (len(entry.data),)
        obs[entry.key] = np.array(entry.data, dtype=np.float64).reshape(shape)
    return obs


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
        # /set_delta_action's toggle_base_mode is the only way to flip
        # scene_loader's shared device.base_mode flag (StepAction has no
        # such field) -- but _set_delta_action_cb_impl has no control_mode
        # guard, so it works fine alongside control_mode:=rl. Tracked here
        # so set_base_mode() only sends a toggle when the mode is actually
        # changing (it's a toggle, not a set).
        self._base_mode_on = False

        self.create_subscription(Observation, "/observations", self._on_observation, 10)
        self.create_subscription(StepInfo, "/reward", self._on_step_info, 10)

        self._step_cli = self.create_client(StepAction, "/step_action")
        self._step_raw_cli = self.create_client(StepActionRaw, "/step_action_raw")
        self._reset_cli = self.create_client(ResetEpisode, "/reset_episode")
        self._mark_rotten_cli = self.create_client(MarkObjectRotten, "/mark_object_rotten")
        self._set_delta_cli = self.create_client(SetDeltaAction, "/set_delta_action")

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

    def wait_for_services(self, timeout_sec=10.0):
        clients = (
            (self._step_cli, "/step_action"),
            (self._step_raw_cli, "/step_action_raw"),
            (self._reset_cli, "/reset_episode"),
            (self._mark_rotten_cli, "/mark_object_rotten"),
            (self._set_delta_cli, "/set_delta_action"),
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
        event = threading.Event()
        holder = {}

        def _on_done(future):
            holder["response"] = future.result()
            event.set()

        client.call_async(request).add_done_callback(_on_done)
        if not event.wait(timeout=timeout_sec):
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

    def set_base_mode(self, enabled):
        """Flip scene_loader's device.base_mode so subsequent step() calls'
        base_dx/base_dy/base_dyaw actually move the mobile base instead of
        being silently dropped (base and arm deltas are mutually exclusive
        -- see _translate_delta_to_env_action). No-op if already in the
        requested mode; StepAction has no toggle field, so this goes
        through /set_delta_action instead (works fine outside teleop mode,
        _set_delta_action_cb_impl has no control_mode guard)."""
        if enabled == self._base_mode_on:
            return
        request = SetDeltaAction.Request()
        request.toggle_base_mode = 1

        response = self._call(
            self._set_delta_cli, request, "/set_delta_action", self._step_timeout_sec
        )
        if not response.success:
            raise RuntimeError(f"/set_delta_action (toggle_base_mode) failed: {response.message}")
        self._base_mode_on = enabled

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
