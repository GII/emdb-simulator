"""Registers FruitShop, a single-arm adaptation of the e-MDB Fruit Shop
experiment (see mdb_experiments/fruit_shop_experiment.yaml).

Importing this module has the side effect of registering FruitShop with
robosuite's REGISTERED_ENVS (via robosuite's EnvMeta), so it must be
imported before robosuite.make() is called with this env name.

Unlike KitchenLift/KitchenPlace, this task has no _check_success()-driven
mission of its own: success/reward for the real e-MDB experiment is computed
by emdb_policy's fruit_shop_bridge.py (a port of the reference
FruitShopSim's stage-gated reward logic, from the paper_experiment/src/
emdb_develop workspace), not by robosuite's env. This task only needs to
expose the physical scene (one fruit, one scale, an accepted/rejected
basket pair, and a handful of fixed robot-relative target zones) that the
bridge's scripted motions act on.

Only one fruit is ever physically instantiated at a time -- the reference
FruitShopSim also only ever perceives/acts on the single closest fruit in
its internal inventory (see perceive_closest_fruit() in
fruit_shop_sim_discrete.py), so there's no need for multiple simultaneous
graspable fruit bodies here.
"""
import os

import numpy as np

import robocasa.macros as robocasa_macros
from robocasa.environments.kitchen.kitchen import *
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
from emdb_simulator.core.camera_config import CustomCamerasMixin


class FruitShop(CustomCamerasMixin, Kitchen):
    """Pick a fruit, test it on the scale, accept/discard it, and place it.

    A single UR5e+2FG7 arm plays every role the two-hand reference
    experiment splits across hands: pick_fruit, place_fruit, test_fruit,
    accept_fruit, discard_fruit, ask_nicely (change_hands has no
    single-arm equivalent and is dropped; press_button/button_light were
    also dropped -- ApproachOnlyMotion's arm control couldn't reliably
    complete the approach -- matching the adapted single-arm experiment
    yaml).
    """

    # Confirmed via scripts/grasp_trial.py (emdb_simulator) against all 19
    # RoboCasa "fruit"-typed categories plus lime: launches KitchenLift with
    # each candidate (scene_loader's obj_groups param), drives
    # PickAndLiftPolicy for 3 episodes, and requires BOTH a contact-based
    # grasp (/emdb/simulator/sensor/obj/grasped) AND a completed 10cm lift
    # (KitchenLift._check_success) -- the same bar kitchen_lift_task.py's own
    # DEFAULT_OBJ_GROUPS comment used, since static bbox math was already
    # shown unreliable there. Only these 6 passed both; apple/cantaloupe/
    # cherry/coconut/kiwi got grasped but never completed a lift (dropped or
    # slipped -- unreliable), and the rest (apricot, dates, grapes, lime,
    # pineapple, pomegranate, raspberry, strawberry, watermelon) never got
    # grasped at all by the ~31mm TwoFG7Gripper jaw. Notably the previous
    # provisional list (lime/kiwi/cherry/strawberry/raspberry) passed none
    # of these.
    DEFAULT_OBJ_GROUPS = ["apple", "mango", "orange", "peach", "pear", "tangerine"]

    # _setup_kitchen_references below requires an island (FixtureType.ISLAND)
    # -- fail fast at construction time (mirrors KitchenLift's identical
    # guard) rather than letting register_fixture_ref raise a more opaque
    # error if ever instantiated with a non-island layout.
    EXCLUDE_LAYOUTS = Kitchen.ISLAND_EXCLUDED_LAYOUTS

    # Fixed offsets (meters, world-frame axes) from the counter fixture's
    # own position -- never raw world constants, since RoboCasa kitchens are
    # procedurally laid out per episode/layout. Mirrors the reference sim's
    # fixed canonical zones (collection_area, weighing_area,
    # fruit_left/right_side_pos in fruit_shop_sim_discrete.py), collapsed
    # from that sim's polar (distance, angle) convention into offsets
    # convenient for a real placement/fixture-relative frame. Tune by eye
    # once a layout is picked.
    COLLECTION_OFFSET = np.array([-0.25, -0.15, 0.0])
    PLACED_OFFSET = np.array([0.0, -0.15, 0.0])

    # Read generically by scene_loader._augment_obs_with_control_frame and
    # published into obs_dict as "<name>" for each property listed here.
    OBS_ZONE_ATTRS = ("collection_pos", "placed_pos")

    # Spawned objects whose live body position (not a fixed offset property)
    # should be published into obs_dict as "<name>_pos" -- read generically
    # by scene_loader._augment_obs_with_control_frame. accepted/rejected are
    # the accept_fruit_policy/discard_fruit_policy drop targets (a basket
    # pair standing in for accepted-fruit vs. discarded/"trash" fruit,
    # RoboCasa has no dedicated trash-bin mesh) and must track wherever the
    # placement sampler actually put them, not a fixed offset that might not
    # line up with the mesh.
    OBS_LIVE_OBJECT_ATTRS = ("scale", "accepted", "rejected")

    # kitchen_objects.py's generic "basket" category mixes ordinary woven
    # baskets with a couple of black wire-mesh variants that read visually
    # as a trash bin -- confirmed by offscreen-rendering every variant in
    # the category. "rejected" pins to these (see _basket_mjcf_path) so it
    # always looks bin-like; "accepted" excludes them so it never does.
    BASKET_TRASH_BIN_IDS = ("Basket027", "Basket044")

    # Basket038 (a small single-handle woven purse/handbag shape, not a
    # produce basket) is a poor fit for "accepted" specifically -- excluded
    # in addition to BASKET_TRASH_BIN_IDS, not because it looks bin-like.
    ACCEPTED_BASKET_EXCLUDE_IDS = BASKET_TRASH_BIN_IDS + ("Basket038",)

    def __init__(self, obj_groups=DEFAULT_OBJ_GROUPS, exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        # Kitchen._load_model's fixture/object placement retry loop (misc/
        # robocasa/robocasa/environments/kitchen/kitchen.py:643-834) already
        # prints exactly which fixture/object PlacementError triggered each
        # retry -- but only if macros.VERBOSE is True (default False, so
        # retries -- e.g. an accepted/rejected basket instance too big for a
        # given island's counter region -- fail silently up to 50 times
        # before the generic "Ran _load_model() 50 times" RuntimeError).
        # Flipping this here (not editing the vendored submodule) surfaces
        # those per-attempt reasons in scene_loader's stdout. It's a
        # process-global macro, so it stays on for the rest of the process
        # once a FruitShop instance exists -- harmless (retry-path-only
        # prints), same tradeoff RoboCasa's own macros module documents.
        robocasa_macros.VERBOSE = True
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # FixtureType.COUNTER matches ANY non-corner counter (island or
        # wall) and, with multiple matches, get_fixture() resolves via
        # self.rng.choice() -- i.e. randomly, per episode. That would
        # silently defeat scene_loader.py's FRUIT_SHOP_LAYOUT_IDS curation
        # (picking island-containing layouts) by letting objects land on a
        # wall counter instead. FixtureType.ISLAND is the distinct enum
        # KitchenLift already relies on for the same reason (see
        # kitchen_lift_task.py) -- don't "simplify" this back to COUNTER.
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.ISLAND))
        self.init_robot_base_ref = self.counter

    def _robot_base_pos(self):
        base_center_site = self.robots[0].robot_model.base.correct_naming("center")
        return np.asarray(
            self.sim.data.site_xpos[self.sim.model.site_name2id(base_center_site)],
            dtype=np.float64,
        )

    @property
    def collection_pos(self):
        # Relative to the robot's own *live* base position, not
        # self.counter.pos -- the counter fixture's own reference point can
        # be a meter or more from wherever the fruit/scale/accepted/rejected
        # cluster and robot spawn actually landed on a large island (its
        # local origin isn't necessarily anywhere near whichever full-depth
        # region/corner _get_obj_cfgs' placement sampler picked). There is
        # no base or object repositioning in this sim (removed -- too
        # unreliable, see fruit_shop_bridge.py's git history), so
        # everything the robot needs to reach has to already be
        # guaranteed close by construction; tying this to the base
        # directly means it's always in reach without needing any
        # runtime correction.
        return self._robot_base_pos() + self.COLLECTION_OFFSET

    @property
    def placed_pos(self):
        return self._robot_base_pos() + self.PLACED_OFFSET

    def _basket_mjcf_path(self, include_ids=None, exclude_ids=None):
        """Pick one "basket" category model.xml path via self.rng, filtered
        by folder id (e.g. "Basket027") -- object cfgs only support
        exclude_obj_groups (whole categories, see EnvUtils.create_obj), not
        excluding specific models within one, so this reproduces
        sample_kitchen_object's exact-xml-path pinning
        (kitchen_object_utils.py) manually over a filtered candidate list.
        """
        paths = []
        for obj_cat in OBJ_CATEGORIES["basket"].values():
            for path in obj_cat.mjcf_paths:
                model_id = os.path.basename(os.path.dirname(path))
                if include_ids is not None and model_id not in include_ids:
                    continue
                if exclude_ids is not None and model_id in exclude_ids:
                    continue
                paths.append(path)
        return str(self.rng.choice(paths))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang("fruit")
        ep_meta["lang"] = (
            f"Pick up the {obj_lang}, test it on the scale, and accept or "
            "discard it."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="fruit",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                init_robot_here=True,
                # No max_size cap: robocasa's sample_kitchen_object() retry
                # loop (misc/robocasa/robocasa/models/objects/
                # kitchen_object_utils.py:260-296) is unbounded -- if no
                # instance across DEFAULT_OBJ_GROUPS ever fits a cap, it
                # spins forever with no error. None of these categories'
                # measured reg_bbox sizes reliably clear 5cm on all 3 axes,
                # so a max_size here previously hung scene_loader at
                # startup. An oversized fruit just makes PickFruitMotion
                # fail to grasp (a visible _run_motion timeout), which is
                # the correct failure mode until the empirical grasp trial
                # above narrows DEFAULT_OBJ_GROUPS down for real.
                placement=dict(
                    fixture=self.counter,
                    size=(0.35, 0.25),
                    # pos=(0, 0) + a fixed-meters "offset" (not a normalized
                    # pos toward a corner) -- offset lands in intra_offset
                    # un-scaled by outer_size (misc/robocasa/robocasa/utils/
                    # env_utils.py:_get_placement_initializer, ~line 1237),
                    # so this corner of the cluster below sits a *constant*
                    # 0.2m/0.2m from the region center regardless of how big
                    # the sampled region actually is -- normalized pos can't
                    # give that guarantee since the same pos value maps to
                    # wildly different real distances per layout (measured,
                    # island full-depth-region "size" ranges from ~0.92m to
                    # ~3.75m across FRUIT_SHOP_LAYOUT_IDS -- see the cluster
                    # comment below).
                    pos=(0.0, 0.0),
                    offset=(-0.2, -0.2),
                    # Split islands (sink/cooktop strip cutting the top
                    # surface into multiple geoms) would otherwise let any
                    # of these 4 objects land in the strip's narrow segment.
                    # full_depth_region drops that segment and keeps only
                    # the full-depth one(s) -- see Counter.get_reset_regions
                    # (misc/robocasa/robocasa/models/fixtures/counter.py:
                    # 595-652). No-op on islands without a split.
                    sample_region_kwargs=dict(full_depth_region=True),
                ),
            )
        )
        # digital_scale: a real, non-graspable RoboCasa prop (see
        # WeighIngredients in misc/robocasa/robocasa/environments/kitchen/
        # composite/measuring_ingredients/weigh_ingredients.py for the
        # precedent this mirrors) used as the physical "scale" surface.
        #
        # ref_obj="fruit" pins scale/accepted/rejected (below) to reuse
        # "fruit"'s own already-sampled reset_region instead of each
        # independently calling Fixture.sample_reset_region() (misc/
        # robocasa/robocasa/models/fixtures/fixture.py:330-383, ends in
        # self.rng.choice(valid_regions)) -- on an island with more than one
        # valid top region (e.g. a sink/cooktop strip splitting it), that
        # let each of the 4 objects land on a *different* region/corner
        # independently, which is what actually produced the "opposite
        # corners of a wide island" problem this offset-based clustering
        # exists to prevent (there's no runtime reach correction in this
        # sim -- see collection_pos's comment above).
        #
        # All 4 objects sit at the 4 corners of a small, *fixed-size*
        # 0.4m x 0.4m square (offset, not normalized pos -- see fruit's
        # comment above) centered on the shared region, instead of spread
        # across it -- keeps them within real arm reach on every layout
        # regardless of how big that layout's island happens to be. An
        # earlier version instead compressed *normalized* pos toward
        # fruit's corner, which (combined with ref_obj) crammed all 4
        # objects into far too little of the *smallest* curated layout's
        # region (measured: layout 2's full-depth region is only ~0.92m x
        # 1.4m) and made RoboCasa's placement sampler retry _load_model()
        # forever ("Cannot place all objects, failed for rejected"),
        # stalling the sim thread long enough to time out unrelated
        # /step_action calls. This fixed-meters 0.4m square was checked
        # against the actual sampled full-depth region size on all 5
        # FRUIT_SHOP_LAYOUT_IDS (via a standalone robosuite.make() +
        # Counter.sample_reset_region() probe, not just guessed) and fits
        # with margin on the tightest one (layout 21, ~1.4m x 0.8m) --
        # worst-case pairwise object clearance is fruit-accepted at
        # ~0.4m center distance vs. their combined ~0.355m half-size sum.
        # Any layout/style combo this still doesn't fit falls back to
        # RoboCasa's own outer _load_model() retry (a fresh style/seed),
        # same as any other placement failure.
        cfgs.append(
            dict(
                name="scale",
                obj_groups="digital_scale",
                placement=dict(
                    fixture=self.counter,
                    ref_obj="fruit",
                    size=(0.3, 0.3),
                    pos=(0.0, 0.0),
                    offset=(0.2, -0.2),
                ),
            )
        )
        # accepted/rejected: a pair of real, non-graspable RoboCasa "basket"
        # props that accept_fruit_policy/discard_fruit_policy drop the
        # tested fruit into. Mirrors the "scale" prop above --
        # OBS_LIVE_OBJECT_ATTRS tracks their live body position rather than
        # a fixed offset, so the drop target always matches wherever the
        # placement sampler put the mesh. obj_groups is pinned to a specific
        # model.xml (not the bare "basket" category) so "rejected" always
        # gets one of BASKET_TRASH_BIN_IDS's bin-like variants and
        # "accepted" never does -- see _basket_mjcf_path.
        #
        # placement "size" here is only an upper cap, not a request: RoboCasa
        # actually samples within min(outer_size, size) per axis
        # (misc/robocasa/robocasa/utils/env_utils.py:_get_placement_initializer,
        # ~line 1171), then further shrinks by the *sampled mesh's own real
        # footprint* before checking min<=max (UniformRandomSampler._sample_x/
        # _sample_y, misc/robocasa/robocasa/utils/placement_samplers.py:
        # 225-265). So "size" must be >= the largest real candidate mesh's
        # footprint, or that combination is *guaranteed* to raise "Invalid
        # x/y range" regardless of how roomy the island is -- measured (via
        # robocasa.macros.VERBOSE + the misc/robocasa kitchen.py debug patch)
        # basket footprints run up to ~0.337m for accepted's pool and ~0.285m
        # for rejected's (Basket027). A too-small "size" (an earlier version
        # of this code tried 0.18/0.18 for "accepted" to fix crowding, which
        # made this specific failure mode worse, not better) is the wrong
        # lever for that; only "size" being too *big* relative to a narrow
        # island's outer_size costs anything, and even then it's just
        # silently clamped down, not an error.
        cfgs.append(
            dict(
                name="accepted",
                obj_groups=self._basket_mjcf_path(exclude_ids=self.ACCEPTED_BASKET_EXCLUDE_IDS),
                placement=dict(
                    fixture=self.counter,
                    ref_obj="fruit",
                    size=(0.36, 0.36),
                    pos=(0.0, 0.0),
                    offset=(0.2, 0.2),
                ),
            )
        )
        cfgs.append(
            dict(
                name="rejected",
                obj_groups=self._basket_mjcf_path(include_ids=self.BASKET_TRASH_BIN_IDS),
                placement=dict(
                    fixture=self.counter,
                    ref_obj="fruit",
                    size=(0.32, 0.32),
                    pos=(0.0, 0.0),
                    offset=(-0.2, 0.2),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        # Success/reward for this experiment is computed by emdb_policy's
        # fruit_shop_bridge.py against /emdb/simulator/sensor/classify_fruit
        # and /emdb/simulator/sensor/place_fruit, not by this env -- there
        # is no single "done" condition to report here (the reference
        # experiment's classify_fruit_mission is terminal, but is satisfied
        # by the bridge's DriveExponential-fed reward, not by scene_loader's
        # generic env._check_success()/progress topic).
        return False
