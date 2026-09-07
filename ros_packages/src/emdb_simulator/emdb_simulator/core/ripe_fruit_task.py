"""Registers RipeFruit, a thin wrapper around RoboCasa's own ChooseRipeFruit
composite task exposing the object/fixture positions scripted motions and
the e-MDB bridge need. Direct port -- no ripeness randomization or hiding;
fruit1 is always the ripe target, per RoboCasa's own _update_fruit_texture().

Importing this module registers RipeFruit with robosuite's REGISTERED_ENVS
(via EnvMeta), so it must be imported before robosuite.make() is called
with this env name -- see registered_tasks.py.
"""
import numpy as np

from robocasa.environments.kitchen.composite.making_juice.choose_ripe_fruit import (
    ChooseRipeFruit,
)
from emdb_simulator.core.camera_config import CustomCamerasMixin


class RipeFruit(CustomCamerasMixin, ChooseRipeFruit):
    """Pick the ripe fruit (fruit1) and place it in the blender.

    The blender always spawns with a lid (Fixture.BASE_TO_AUXILIARY_FIXTURES
    maps "blender" -> "blender_lid" automatically at scene assembly, not
    something ChooseRipeFruit itself opts into), so fruit1 can't actually
    go in until the lid is removed -- see robocasa's own OpenBlenderLid
    atomic task (misc/robocasa/robocasa/environments/kitchen/atomic/
    kitchen_blender.py) for the reference success check this mirrors
    (lid off the blender, sitting on a counter). The bridge exposes this as
    its own open_blender policy rather than doing it automatically at
    reset, since removing the lid is a real precondition the architecture
    has to sequence, not scene setup.
    """

    # fruit0/fruit1 are both self.objects entries (see ChooseRipeFruit.
    # _get_obj_cfgs) -- scene_loader._augment_obs_with_control_frame already
    # publishes "<name>_pos"/"<name>_top_z" for any name listed here,
    # generically, no scene_loader change needed (scene_loader.py:826-851).
    OBS_LIVE_OBJECT_ATTRS = ("fruit0", "fruit1")

    # A few cm to the side of the blender -- anywhere on a counter satisfies
    # OpenBlenderLid's own success check (lid far from the gripper +
    # touching any Counter fixture), so this doesn't need to be precise.
    # Anchored to the blender itself, not self.counter.pos: confirmed live
    # (2026-09) that self.counter.pos is that counter fixture's own overall
    # centroid, which can be *meters* from the blender's actual spot on it
    # (a single counter run can span much of the kitchen) -- that sent the
    # arm reaching more than 2m away, well outside its limits. The blender
    # is the one guaranteed to be near the robot (ChooseRipeFruit's own
    # init_robot_base_ref = self.blender), so offset from that instead.
    # Tune by eye once a layout is picked, same as fruit_shop_task.py's own
    # COLLECTION_OFFSET/PLACED_OFFSET.
    LID_DROP_OFFSET = np.array([0.2, 0.0, 0.05])

    OBS_ZONE_ATTRS = (
        "blender_lid_pos", "lid_drop_pos", "dest_top_z", "lid_drop_top_z",
        "blender_int_pos", "blender_int_top_z", "lid_on_blender",
    )

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # Reuse scene_loader's existing hardcoded self.env.dest -> "dest_pos"
        # passthrough (the same one KitchenPlace's self.dest already relies
        # on, scene_loader.py:811-813) instead of adding a new OBS_ZONE_ATTRS
        # entry -- this way TransportReleaseMotion's default
        # target_pos_key="dest_pos" (scripted_policies.py) already targets
        # the blender with zero further plumbing.
        self.dest = self.blender

    @property
    def blender_lid_pos(self):
        # Fixture.pos (used for the blender/counter themselves) is a
        # nominal/static position -- the lid has a free joint (BlenderLid.
        # __init__, misc/robocasa/robocasa/models/fixtures/blender.py) and
        # moves once grasped, so its live pose has to come from sim.data
        # directly, the same way Blender.get_curr_lid_pos(env) itself reads
        # it. None (skipped by the generic OBS_ZONE_ATTRS passthrough,
        # scene_loader.py:821-824) if no lid was actually attached.
        if self.blender.blender_lid is None:
            return None
        return np.asarray(
            self.sim.data.get_body_xpos(f"{self.blender.blender_lid.name}_main"),
            dtype=np.float64,
        )

    @property
    def lid_drop_pos(self):
        return np.asarray(self.blender.pos, dtype=np.float64) + self.LID_DROP_OFFSET

    @property
    def lid_drop_top_z(self):
        # lid_drop_pos sits right next to the blender (LID_DROP_OFFSET,
        # ~0.2m away) -- the obstacle the open_blender motion has to clear
        # while carrying the lid there is still the blender's own rim, not
        # a rim of lid_drop_pos's own (it doesn't have one, it's an empty
        # spot on the counter). Reusing dest_top_z here is what makes
        # TransportReleaseMotion(target_pos_key="lid_drop_pos")'s hover
        # height (via its generic "<key>_top_z" lookup) stay above the
        # blender for the whole transit, not just at the final drop point.
        return self.dest_top_z

    @property
    def dest_top_z(self):
        # TransportReleaseMotion._container_top_z (scripted_policies.py)
        # already generically looks for "<target_pos_key without '_pos'>_top_z"
        # -- for target_pos_key="dest_pos" that's "dest_top_z" -- to clear a
        # container's rim before descending (built for FruitShop's baskets,
        # scene_loader.py:826-851's OBS_LIVE_OBJECT_ATTRS path). The blender
        # is wired through the separate hardcoded self.env.dest -> "dest_pos"
        # passthrough instead (scene_loader.py:811-813), which has no such
        # companion, so without this the fruit-placing motion only hovered
        # PLACE_HOVER_HEIGHT above blender.pos's Z (its bounding-box
        # *center*, not its rim) and drove the held fruit straight into the
        # blender's own body instead of over the opening -- confirmed live
        # (2026-09): the arm visibly collided with the blender.
        #
        # Fixture.pos is the bounding-box center, not the base: confirmed by
        # comparing it against blender_lid_pos, which sits right at the rim
        # when the lid is on (blender_lid_pos[2] landed within ~1cm of
        # blender.pos[2] + blender.size[2]/2, not blender.pos[2] +
        # blender.size[2], across several live resets) -- so this is a real
        # measured rim height, not a guessed constant.
        #
        # Only used as the "obstacle to clear" reference for transit (the
        # open_blender lid-carry and, historically, the fruit-placing hover
        # height) -- NOT precise enough for the actual release point, see
        # blender_int_pos below.
        return float(self.blender.pos[2] + self.blender.size[2] / 2.0)

    @property
    def lid_on_blender(self):
        # RoboCasa's own ground truth, not this task's opinion: Blender.
        # update_state(env) (misc/robocasa/robocasa/models/fixtures/
        # blender.py) recomputes self._lid_on_blender from the lid's live
        # distance/orientation relative to its closed-on-blender pose every
        # single env.step() (Kitchen._post_action() -> Kitchen.update_state()
        # -> fixtr.update_state(self) for every fixture, unconditionally --
        # confirmed in kitchen.py, not something this task has to opt into).
        # The bridge's own open_blender_policy used to gate solely on
        # whether ITS OWN scripted motion had reported success once before
        # -- but "the scripted motion reached STATE_DONE" only means the
        # motion finished, not that the lid actually ended up off the
        # blender (the same gap already found and fixed for
        # place_in_blender's own "motion succeeded but the fruit didn't
        # land inside" issue) -- surfacing the real state here lets the
        # bridge check reality directly instead of trusting its own past
        # optimism.
        return float(self.blender.get_state()["lid_on_blender"])

    @property
    def blender_int_pos(self):
        # The ACTUAL region ChooseRipeFruit._check_success() tests fruit1
        # against (OU.obj_inside_of -> Fixture.get_int_sites(), region
        # "int") -- confirmed live (2026-09) that this is a tight ~9x9cm box
        # whose XY center sits ~2.6cm off from blender.pos's own XY (Y in
        # particular), and whose top is ~4.4cm lower than dest_top_z's
        # bounding-box-based estimate. place_in_blender_policy's motion
        # reaching STATE_DONE (i.e. completing its transport/lower/release
        # sequence) doesn't mean the fruit actually registers as "inside"
        # per _check_success() unless it's released into *this* region
        # specifically, not just "somewhere near the blender's center" --
        # confirmed live: every individual motion step succeeded yet
        # _check_success() still read False until targeting this instead
        # of dest_pos.
        p0, px, py, _pz = self.blender.get_int_sites(relative=False)["int"]
        return np.array([(p0[0] + px[0]) / 2.0, (p0[1] + py[1]) / 2.0, p0[2]])

    @property
    def blender_int_top_z(self):
        # TransportReleaseMotion._container_top_z derives this key name
        # generically from "blender_int_pos" -- see blender_int_pos's own
        # comment for why this, not dest_top_z, is the real rim to clear
        # for an actual successful release.
        _p0, _px, _py, pz = self.blender.get_int_sites(relative=False)["int"]
        return float(pz[2])
