"""Registry of custom task modules.

Each import below registers that task class with robosuite as a side
effect (see the individual task module's own docstring). scene_loader
imports this module once instead of importing every task module directly.

New entries are appended by scripts/create_task.py -- keep one import per
line and don't remove the trailing marker comment.
"""
from emdb_simulator.core import kitchen_lift_task  # noqa: F401  registers KitchenLift
# --- new tasks are appended below this line by scripts/create_task.py ---
from emdb_simulator.core import kitchen_place_task  # noqa: F401  registers KitchenPlace
from emdb_simulator.core import fruit_shop_task  # noqa: F401  registers FruitShop
from emdb_simulator.core import ripe_fruit_task  # noqa: F401  registers RipeFruit
