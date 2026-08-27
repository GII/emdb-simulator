"""Registers a "lift" task that still uses RoboCasa kitchen layouts/styles.

Importing this module has the side effect of registering KitchenLift with
robosuite's REGISTERED_ENVS (via robosuite's EnvMeta), so it must be
imported before robosuite.make() is called with this env name.
"""
from robocasa.environments.kitchen.kitchen import *
from emdb_simulator.core.cube_object import OBJ_CATEGORIES, OBJ_GROUPS
from emdb_simulator.core.camera_config import CustomCamerasMixin


class KitchenLift(CustomCamerasMixin, Kitchen):
    """Pick up a counter-top object and raise it above a height threshold.

    Mirrors robosuite's plain Lift task (success = object raised above a
    fixed margin over its starting height) but spawns the object on the
    RoboCasa kitchen island instead of Lift's bare table arena, so
    layout_ids/style_ids still apply. Only layouts with an island fixture
    can be used.
    """

    LIFT_HEIGHT = 0.10  # meters above starting height counted as "lifted"
    EXCLUDE_LAYOUTS = Kitchen.ISLAND_EXCLUDED_LAYOUTS

    # TwoFG7Gripper (see gripper_loader.py) is a real OnRobot 2FG7 small-parts
    # gripper: ~31mm max jaw opening. Most RoboCasa "graspable" categories are
    # full-size kitchen items (an apple's narrowest side is ~54-75mm) that the
    # jaw physically cannot close around, so the object just sits in the open
    # gap instead of being gripped. These four categories were verified by
    # measuring every instance's reg_bbox (restricted to Kitchen's default
    # obj_registries=("objaverse", "lightwheel") -- categories that only exist
    # under "aigen", e.g. chili_pepper, are never actually reachable and will
    # raise during sampling) and then confirmed with real end-to-end grasp
    # trials: every sampled instance's narrowest dimension stays under the
    # jaw's mechanical limit with margin, and all lift successfully.
    DEFAULT_OBJ_GROUPS = ["cube"]

    def __init__(
        self,
        obj_groups=DEFAULT_OBJ_GROUPS,
        exclude_obj_groups=None,
        *args,
        **kwargs,
    ):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        self._obj_start_z = None
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.island = self.register_fixture_ref("island", dict(id=FixtureType.ISLAND))
        self.init_robot_base_ref = self.island

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"Pick up and lift the {obj_lang}."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.island,
                    size=(0.40, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def reset(self):
        # Just invalidates the cached baseline -- see _check_success() for
        # why it's captured lazily instead of here.
        obs = super().reset()
        self._obj_start_z = None
        return obs

    def _check_success(self):
        # Baseline captured lazily, on the first real query after a reset,
        # rather than eagerly in reset(). Eager capture (in _reset_internal()
        # or reset() itself) is unreliable here: with collect_demos:=true,
        # robosuite's DataCollectionWrapper._start_new_episode() (misc/
        # robosuite/robosuite/wrappers/data_collection_wrapper.py:60-93,
        # vendored, not edited) runs on the first post-reset interaction and
        # calls env.reset_from_xml_string(), which rebuilds self.sim and
        # calls self.reset() again -- with a fresh, unrelated random
        # placement -- and only *afterwards* restores the actual intended
        # state via sim.set_state_from_flattened(). Any baseline captured
        # during that inner reset() is silently stale once the restore
        # happens, and nothing re-syncs it. By the time anything outside
        # robosuite/robocasa (e.g. scene_loader's render loop) can actually
        # query _check_success(), that whole dance has already finished and
        # self.sim reflects the final, correct state -- confirmed by logging
        # obj_z at every render tick while debugging spurious "Success
        # achieved!" reports: it was already stable and correct from the
        # very first tick after "Scene loaded".
        obj_z = self.sim.data.body_xpos[self.obj_body_id["obj"]][2]
        if self._obj_start_z is None:
            self._obj_start_z = obj_z
            return False
        return (obj_z - self._obj_start_z) > self.LIFT_HEIGHT
