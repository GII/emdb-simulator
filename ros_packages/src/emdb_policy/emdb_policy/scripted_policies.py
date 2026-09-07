#!/usr/bin/env python3
"""Deterministic, hand-coded (non-learned) policies for KitchenLift/KitchenPlace.

Each policy is a small closed-loop state machine driven by proportional
control toward object/end-effector positions in obs_dict, exposing a bound
policy_fn(obs_dict, rng) -> action_vector compatible with policy_node.py's
PolicyRunner, plus on_episode_start() to reset state between episodes.

Action vectors are the flat 7-dim [dx, dy, dz, droll, dpitch, dyaw, gripper]
layout AgentBridge.step_vector() expects: gripper is an absolute open(<=0)/
closed(>0) command, and position/rotation are OSC_POSE deltas.

OSC_POSE's input_ref_frame is "base" (composite/basic.json): dx/dy/dz get
added to the controller's goal in its "origin" frame (a per-arm-controller
mount site, not the mobile base body -- see
SceneLoader._augment_obs_with_control_frame), so a world-frame position
error has to be rotated into that frame before being sent. scene_loader
publishes the site's full 3x3 world-frame rotation matrix as
obs_dict["robot0_origin_ori"] (flattened, row-major) -- its transpose is
the world-to-origin-frame rotation.

scene_loader also runs with mirror_actions=True (SceneLoader.__init__),
i.e. every /step_action goes through robosuite's
Device.input2action(mirror_actions=True), which negates dx/dy before OSC
ever sees them (a teleop convenience: the operator's left/right,
forward/back matches what they see facing the robot). Composed with the
origin_ori rotate-then-un-rotate round trip, that negation survives
unchanged for any pure yaw rotation (R @ diag(-1,-1,1) @ R^T ==
diag(-1,-1,1) for rotations about Z, since a 180 deg turn commutes with
any other Z rotation) -- so _to_base_frame has to counter it by negating
x/y right back, regardless of the robot's current orientation.
"""
import numpy as np

XY_APPROACH_THRESHOLD = 0.02  # meters
Z_DESCEND_THRESHOLD = 0.015  # meters
HOVER_HEIGHT = 0.10  # meters above the object to approach from
GRASP_HEIGHT_OFFSET = 0.02  # meters above obj_pos to descend to before closing
GRASP_SETTLE_STEPS = 15  # gripper closes at speed=0.2/step (gripper_loader.py)
LIFT_HEIGHT = 0.12  # meters to raise above the grasp height
PLACE_HOVER_HEIGHT = 0.15  # meters above dest_pos to transport at
PLACE_LOWER_OFFSET = 0.05  # meters above dest_pos to lower to before releasing
# meters above a container's rim (TransportReleaseMotion's "<name>_top_z")
# to release into it, when the target has one -- see _container_top_z.
CONTAINER_RIM_CLEARANCE = 0.05
RELEASE_SETTLE_STEPS = 10
# How far TransportReleaseMotion retreats straight up after releasing,
# before reporting DONE -- see its STATE_RETREAT for why straight up (not
# sideways) and why this matters for RipeFruit's real _check_success()
# (OU.gripper_obj_far's 0.25m threshold).
RETREAT_HEIGHT = 0.10
ASK_NICELY_WAIT_STEPS = 3  # default idle duration for IdleMotion


class _FrameControlMixin:
    """World-frame-error -> base-frame-delta p-control, shared by every
    scripted state machine below (see module docstring for why the
    origin_ori un-rotate + mirror_actions negation is needed)."""

    def _to_base_frame(self, world_err, obs_dict):
        origin_ori = obs_dict["robot0_origin_ori"].reshape(3, 3)
        base_err = origin_ori.T @ world_err
        return base_err * np.array([-1.0, -1.0, 1.0])  # undo mirror_actions=True

    def _p_control(self, world_err, obs_dict, kp=4.0, max_delta=0.03):
        base_err = self._to_base_frame(np.asarray(world_err, dtype=np.float64), obs_dict)
        return np.clip(base_err * kp, -max_delta, max_delta)


class ScriptedPolicyBase(_FrameControlMixin):
    """Shared APPROACH -> DESCEND -> GRASP -> LIFT state machine."""

    # Overridden by subclasses that approach a differently-named object
    # (e.g. FruitShop's PickFruitMotion reads "fruit_pos" instead of "obj_pos").
    OBJ_POS_KEY = "obj_pos"

    # Class-attr defaults (module constants above) for the DESCEND/GRASP
    # height and convergence tolerance -- overridable per-instance for
    # objects whose graspable point isn't shaped like a fruit/cube (see
    # grasp_height_offset/z_descend_threshold below and RipeFruit's lid
    # grasp, which needs both: the lid's handle is a ~5mm-thick raised
    # knob, thinner than the fruit-tuned Z_DESCEND_THRESHOLD=0.015 default,
    # so DESCEND could stop up to 1.5cm off-target -- either closing on the
    # flat lid surface around the handle's base or well above it -- instead
    # of centered on the handle itself.
    GRASP_HEIGHT_OFFSET = GRASP_HEIGHT_OFFSET
    Z_DESCEND_THRESHOLD = Z_DESCEND_THRESHOLD

    STATE_APPROACH = "APPROACH"
    STATE_DESCEND = "DESCEND"
    STATE_GRASP = "GRASP"
    STATE_LIFT = "LIFT"
    STATE_DONE = "DONE"

    def __init__(self, obj_pos_key=None, grasp_height_offset=None, z_descend_threshold=None,
                 obstacle_clear_z=None):
        # Optional override of OBJ_POS_KEY, matching TransportReleaseMotion's
        # existing target_pos_key constructor param -- lets a caller target
        # a differently-named object without a one-line subclass per name
        # (see PickObjectMotion below). Safe to add here: on_episode_start(),
        # not __init__, is what resets per-episode state, so this doesn't
        # change existing zero-arg PickAndLiftPolicy()/PickFruitMotion()
        # construction.
        if obj_pos_key is not None:
            self.OBJ_POS_KEY = obj_pos_key
        if grasp_height_offset is not None:
            self.GRASP_HEIGHT_OFFSET = grasp_height_offset
        if z_descend_threshold is not None:
            self.Z_DESCEND_THRESHOLD = z_descend_threshold
        # None by default (existing callers unaffected): a known nearby
        # obstacle's height APPROACH must clear before moving horizontally,
        # same rationale as TransportReleaseMotion's top_z-aware hover.
        # Needed for RipeFruit specifically -- fruit1 and the blender's lid
        # both sit right next to a 0.46m-tall blender, and a single diagonal
        # p-control step toward the object's own (low) hover point can drag
        # along the blender's side for many steps, or get stuck against it
        # entirely, before ever clearing it -- confirmed live (2026-09): up
        # to 43/~150 steps in sustained contact on an otherwise "successful"
        # pick, and one grasp stuck in contact for 178/200 steps outright.
        self.obstacle_clear_z = obstacle_clear_z

    def on_episode_start(self):
        self.state = self.STATE_APPROACH
        self._counter = 0
        self._lift_start_z = None

    def policy_fn(self, obs_dict, rng):
        del rng  # deterministic policy
        return self._step(obs_dict)

    def _step(self, obs_dict):
        obj_pos = obs_dict[self.OBJ_POS_KEY]
        eef_pos = obs_dict["robot0_eef_pos"]
        action6 = np.zeros(6)
        gripper_cmd = -1.0

        if self.state == self.STATE_APPROACH:
            final_target = obj_pos + np.array([0, 0, HOVER_HEIGHT])
            # obstacle_clear_z set (RipeFruit's fruit1/lid, both next to the
            # blender): a proper "up, over, down" 3-phase waypoint instead
            # of one diagonal p-control step, which can drag along/get
            # stuck against the blender's side while still below it, or (if
            # applied naively as just "rise first") re-create the same
            # problem while descending once past that height. No-op
            # (identical to the original single-step behavior) when
            # obstacle_clear_z is None, so FruitShop/KitchenLift's existing
            # motions are unaffected.
            if self.obstacle_clear_z is not None:
                safe_z = max(self.obstacle_clear_z, final_target[2])
                xy_aligned = np.linalg.norm(obj_pos[:2] - eef_pos[:2]) < XY_APPROACH_THRESHOLD
                if not xy_aligned and eef_pos[2] < safe_z - 0.02:
                    waypoint = np.array([eef_pos[0], eef_pos[1], safe_z])  # 1. rise in place
                elif not xy_aligned:
                    waypoint = np.array([obj_pos[0], obj_pos[1], safe_z])  # 2. move over, still safe
                else:
                    waypoint = final_target  # 3. aligned -- now safe to descend
            else:
                waypoint = final_target
            action6[:3] = self._p_control(waypoint - eef_pos, obs_dict)
            final_err = final_target - eef_pos
            if np.linalg.norm(final_err[:2]) < XY_APPROACH_THRESHOLD and abs(final_err[2]) < 0.03:
                self.state = self.STATE_DESCEND

        elif self.state == self.STATE_DESCEND:
            target = obj_pos + np.array([0, 0, self.GRASP_HEIGHT_OFFSET])
            err = target - eef_pos
            action6[:3] = self._p_control(err, obs_dict)
            if np.linalg.norm(err) < self.Z_DESCEND_THRESHOLD:
                self.state = self.STATE_GRASP
                self._counter = 0

        elif self.state == self.STATE_GRASP:
            gripper_cmd = 1.0
            target = obj_pos + np.array([0, 0, self.GRASP_HEIGHT_OFFSET])
            action6[:3] = self._p_control(target - eef_pos, obs_dict, max_delta=0.005)
            self._counter += 1
            if self._counter >= GRASP_SETTLE_STEPS:
                self.state = self.STATE_LIFT
                self._lift_start_z = eef_pos[2]

        elif self.state == self.STATE_LIFT:
            gripper_cmd = 1.0
            target_z = self._lift_start_z + LIFT_HEIGHT
            err = np.array([0.0, 0.0, target_z - eef_pos[2]])
            action6[:3] = self._p_control(err, obs_dict)
            if eef_pos[2] >= target_z - 0.01:
                self.state = self.STATE_DONE

        elif self.state == self.STATE_DONE:
            gripper_cmd = 1.0

        return np.concatenate([action6, [gripper_cmd]])


class PickAndLiftPolicy(ScriptedPolicyBase):
    """Go straight to the cube, close the gripper, lift it (KitchenLift)."""


class PlacePolicy(ScriptedPolicyBase):
    """Pick up the cube, then carry it to dest_pos and release it (KitchenPlace)."""

    STATE_TRANSPORT = "TRANSPORT"
    STATE_LOWER = "LOWER"
    STATE_RELEASE = "RELEASE"

    def _step(self, obs_dict):
        if self.state not in (self.STATE_TRANSPORT, self.STATE_LOWER, self.STATE_RELEASE):
            action = super()._step(obs_dict)
            if self.state == self.STATE_DONE:
                # Base class reached DONE (lifted); hand off to transport.
                self.state = self.STATE_TRANSPORT
            return action

        eef_pos = obs_dict["robot0_eef_pos"]
        dest_pos = obs_dict["dest_pos"]
        action6 = np.zeros(6)
        gripper_cmd = 1.0

        if self.state == self.STATE_TRANSPORT:
            target = dest_pos + np.array([0, 0, PLACE_HOVER_HEIGHT])
            err = target - eef_pos
            action6[:3] = self._p_control(err, obs_dict)
            if np.linalg.norm(err[:2]) < XY_APPROACH_THRESHOLD and abs(err[2]) < 0.03:
                self.state = self.STATE_LOWER

        elif self.state == self.STATE_LOWER:
            target = dest_pos + np.array([0, 0, PLACE_LOWER_OFFSET])
            err = target - eef_pos
            action6[:3] = self._p_control(err, obs_dict)
            if np.linalg.norm(err) < Z_DESCEND_THRESHOLD:
                self.state = self.STATE_RELEASE
                self._counter = 0

        elif self.state == self.STATE_RELEASE:
            gripper_cmd = -1.0
            self._counter += 1
            if self._counter >= RELEASE_SETTLE_STEPS:
                self.state = self.STATE_DONE

        return np.concatenate([action6, [gripper_cmd]])


# --- FruitShop motion primitives ---------------------------------------
#
# fruit_shop_bridge.py drives these one at a time, to completion, per
# executed_policy_service call -- unlike PickAndLiftPolicy/PlacePolicy
# (driven continuously across an episode by policy_node.py's PolicyRunner),
# each of these is instantiated fresh, run via a small step loop, and
# discarded. See fruit_shop_sim_discrete.py (paper_experiment/src/
# emdb_develop) for the reference policy semantics these stand in for.


class PickFruitMotion(ScriptedPolicyBase):
    """Approach, grasp and lift the fruit (pick_fruit)."""

    OBJ_POS_KEY = "fruit_pos"


class PickObjectMotion(ScriptedPolicyBase):
    """Approach, grasp and lift any named object -- generic version of
    PickFruitMotion/PickAndLiftPolicy for tasks with more than one
    differently-named graspable target (e.g. RipeFruit's fruit1). Pass
    obj_pos_key to ScriptedPolicyBase's constructor, e.g.
    PickObjectMotion(obj_pos_key="fruit1_pos"). For an object whose
    graspable point isn't fruit/cube-shaped (e.g. a thin handle knob),
    also override grasp_height_offset/z_descend_threshold -- see
    RipeFruit's blender-lid grasp in ripe_fruit_bridge.py."""


class TransportReleaseMotion(_FrameControlMixin):
    """Carry an already-held object to TARGET_POS_KEY and release it.

    Standalone TRANSPORT -> LOWER -> RELEASE -> DONE state machine, factored
    out of PlacePolicy's tail so it can target different fixed zones
    (scale_pos, accepted_pos, rejected_pos, placed_pos) without re-grasping
    logic bundled in. Assumes the gripper is already closed around the
    object when on_episode_start() is called -- there is no physical
    teleport between zones, so accept_fruit/discard_fruit re-grasp from the
    scale via a fresh PickFruitMotion before chaining into this.
    """

    TARGET_POS_KEY = "dest_pos"

    STATE_TRANSPORT = "TRANSPORT"
    STATE_LOWER = "LOWER"
    STATE_RELEASE = "RELEASE"
    STATE_RETREAT = "RETREAT"
    STATE_DONE = "DONE"

    def __init__(self, target_pos_key=None, lower_target_z=None):
        if target_pos_key is not None:
            self.TARGET_POS_KEY = target_pos_key
        # Escape hatch from STATE_LOWER's generic
        # max(target_pos[2]+OFFSET, top_z+CLEARANCE) floor. That formula
        # assumes target_pos[2] might unsafely be AT/BELOW the container's
        # rim (a basket's accepted_pos/rejected_pos are fixed robot-relative
        # constants that could coincidentally land low) and clamps up to a
        # safe release-just-above-the-rim height -- correct for FruitShop's
        # shallow baskets, where the rim sits close to the object's own body
        # height anyway. It's actively wrong for a deep, narrow container
        # (RipeFruit's blender) whose target_pos[2] is deliberately the
        # INTERIOR'S OWN FLOOR, well below its top_z by design -- the max()
        # then always picks top_z, so LOWER never actually descends past
        # the rim at all. Confirmed live (2026-09): the fruit and gripper
        # ended up stuck together at rim height, never separating, because
        # RELEASE was opening the gripper right at the opening's edge
        # instead of down inside it. Set this to bypass the max() entirely
        # and descend straight to a known-safe explicit height instead.
        self.lower_target_z = lower_target_z

    def on_episode_start(self):
        self.state = self.STATE_TRANSPORT
        self._counter = 0
        self._retreat_start_z = None

    def policy_fn(self, obs_dict, rng):
        del rng
        return self._step(obs_dict)

    def _container_top_z(self, obs_dict):
        # scene_loader publishes "<name>_top_z" alongside "<name>_pos" for
        # every OBS_LIVE_OBJECT_ATTRS object (SceneLoader._augment_obs_with_
        # control_frame) -- None for target keys with no backing object
        # (e.g. "placed_pos", an abstract drop zone), which is exactly when
        # there's nothing to clear and the old PLACE_HOVER_HEIGHT/
        # PLACE_LOWER_OFFSET-only behavior below is already correct.
        if not self.TARGET_POS_KEY.endswith("_pos"):
            return None
        top_z = obs_dict.get(self.TARGET_POS_KEY[: -len("_pos")] + "_top_z")
        # obs_dict values normally arrive as arrays (even 0-d/1-element
        # ones for a scalar observation, depending on how the ROS message
        # round-trip reshapes it) -- force a plain float so the max()
        # comparisons below can't end up building a ragged array out of a
        # stray non-scalar element.
        return None if top_z is None else float(top_z)

    def _step(self, obs_dict):
        eef_pos = obs_dict["robot0_eef_pos"]
        target_pos = obs_dict[self.TARGET_POS_KEY]
        top_z = self._container_top_z(obs_dict)
        action6 = np.zeros(6)
        gripper_cmd = 1.0

        if self.state == self.STATE_TRANSPORT:
            # Some basket variants (accepted/rejected) are tall enough that
            # target_pos[2] + PLACE_HOVER_HEIGHT alone can still be below
            # the rim -- clear top_z too, not just the object's own body
            # origin, or the transport approach drives the held fruit
            # straight into the basket wall instead of over it.
            hover_z = target_pos[2] + PLACE_HOVER_HEIGHT
            if top_z is not None:
                hover_z = max(hover_z, top_z + PLACE_HOVER_HEIGHT)
            # Rise straight up in place to hover_z BEFORE moving sideways,
            # instead of one diagonal p-control step toward (target_x,
            # target_y, hover_z) -- for a short container the arm is
            # already above hover_z right after its pick-up lift, so this
            # is a no-op there, but for a tall one (e.g. RipeFruit's
            # blender, 0.46m) the arm can still be below hover_z when this
            # state starts, and a diagonal path drifts sideways while still
            # low enough to clip the container's side wall on the way --
            # confirmed live. Moving straight up first, then over at
            # hover_z, avoids that regardless of container height.
            #
            # final_target (not the rise-in-place waypoint below) is what
            # the STATE_LOWER transition check has to measure against --
            # using the waypoint instead would trivially satisfy the XY
            # threshold the instant the arm reaches hover_z at its
            # *starting* X/Y, before ever moving toward target_pos at all.
            final_target = np.array([target_pos[0], target_pos[1], hover_z])
            if eef_pos[2] < hover_z - 0.02:
                waypoint = np.array([eef_pos[0], eef_pos[1], hover_z])
            else:
                waypoint = final_target
            action6[:3] = self._p_control(waypoint - eef_pos, obs_dict)
            final_err = final_target - eef_pos
            if np.linalg.norm(final_err[:2]) < XY_APPROACH_THRESHOLD and abs(final_err[2]) < 0.03:
                self.state = self.STATE_LOWER

        elif self.state == self.STATE_LOWER:
            if self.lower_target_z is not None:
                lower_z = self.lower_target_z
            else:
                # Release just above the rim (not down at target_pos's own
                # body-origin height, which for a tall basket can be well
                # below it) so the fruit drops cleanly in instead of the
                # gripper trying to descend into solid basket wall.
                lower_z = target_pos[2] + PLACE_LOWER_OFFSET
                if top_z is not None:
                    lower_z = max(lower_z, top_z + CONTAINER_RIM_CLEARANCE)
            target = np.array([target_pos[0], target_pos[1], lower_z])
            err = target - eef_pos
            action6[:3] = self._p_control(err, obs_dict)
            if np.linalg.norm(err) < Z_DESCEND_THRESHOLD:
                self.state = self.STATE_RELEASE
                self._counter = 0

        elif self.state == self.STATE_RELEASE:
            gripper_cmd = -1.0
            self._counter += 1
            if self._counter >= RELEASE_SETTLE_STEPS:
                self.state = self.STATE_RETREAT
                self._retreat_start_z = eef_pos[2]

        elif self.state == self.STATE_RETREAT:
            # Straight up from wherever release happened, not sideways or
            # back toward target_pos -- the space directly above is already
            # known-clear (this is exactly the path TRANSPORT/LOWER just
            # descended through this same episode), so this can't introduce
            # a *new* collision the way picking some other retreat direction
            # could (e.g. into the container's own body). Needed so a real
            # success check that requires the gripper to have moved away
            # from the released object (e.g. RipeFruit's own
            # OU.gripper_obj_far, th=0.25m) gets a chance to observe it --
            # confirmed live (2026-09) that without this, the arm stayed
            # sitting right at the release point through STATE_DONE, so
            # info["success"] never went True even once the fruit had
            # settled inside the blender.
            gripper_cmd = -1.0
            target_z = self._retreat_start_z + RETREAT_HEIGHT
            err = np.array([0.0, 0.0, target_z - eef_pos[2]])
            action6[:3] = self._p_control(err, obs_dict)
            if eef_pos[2] >= target_z - 0.01:
                self.state = self.STATE_DONE

        elif self.state == self.STATE_DONE:
            gripper_cmd = -1.0

        return np.concatenate([action6, [gripper_cmd]])


class IdleMotion:
    """Hold position and the current gripper state for wait_steps ticks.

    Physical stand-in for ask_nicely (no second arm to hand off to, so the
    single-arm adaptation scripts this as an idle wait) and for test_fruit's
    "observe the fruit" phase.
    """

    STATE_WAIT = "WAIT"
    STATE_DONE = "DONE"

    def __init__(self, wait_steps=ASK_NICELY_WAIT_STEPS):
        self.wait_steps = wait_steps

    def on_episode_start(self, gripper_closed=False):
        self.state = self.STATE_WAIT
        self._counter = 0
        self._gripper_cmd = 1.0 if gripper_closed else -1.0

    def policy_fn(self, obs_dict, rng):
        del obs_dict, rng
        if self.state == self.STATE_WAIT:
            self._counter += 1
            if self._counter >= self.wait_steps:
                self.state = self.STATE_DONE
        return np.concatenate([np.zeros(6), [self._gripper_cmd]])
