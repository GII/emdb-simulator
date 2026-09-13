"""Sphinx configuration for the EMDB Simulator documentation."""
import os
import sys
import types

# -- Path setup --------------------------------------------------------
# Make the ament_python packages importable for autodoc without needing a
# sourced ROS 2 environment. Each entry is the package's setup.py directory
# (the parent of the actual `import`-able module).
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "ros_packages", "src", "emdb_policy"))
sys.path.insert(0, os.path.join(REPO_ROOT, "ros_packages", "src", "emdb_simulator"))

# emdb_simulator.core.gripper_loader calls get_package_share_directory() at
# module import time and feeds the result straight into os.path.join(), so a
# generic autodoc_mock_imports entry isn't enough (a bare Mock isn't a valid
# path-like object). Install a minimal real stub instead, resolving to the
# package's own source share/ dir, so importing it at doc-build time works
# without a sourced ROS 2 environment.
_ament_index_python = types.ModuleType("ament_index_python")
_ament_index_python_packages = types.ModuleType("ament_index_python.packages")
_ament_index_python_packages.get_package_share_directory = lambda name: os.path.join(
    REPO_ROOT, "ros_packages", "src", name
)
_ament_index_python.packages = _ament_index_python_packages
sys.modules.setdefault("ament_index_python", _ament_index_python)
sys.modules.setdefault("ament_index_python.packages", _ament_index_python_packages)

# -- Project information -------------------------------------------------
project = "EMDB Simulator"
author = "Fabian Alvarez Juarez"
copyright = "2026, Fabian Alvarez Juarez"
release = "0.0.0"

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Autodoc / autosummary -------------------------------------------------
# rclpy and the ROS message/service packages are not importable outside a
# sourced ROS 2 + matching-Python-version environment (this project's docs
# venv is Python 3.12; ROS Humble targets 3.10). Mock them out so autodoc can
# still inspect our own modules and extract signatures/docstrings.
autodoc_mock_imports = [
    "rclpy",
    "sensor_msgs",
    "geometry_msgs",
    "std_msgs",
    "std_srvs",
    "launch",
    "launch_ros",
    "emdb_interfaces",
    "lerobot",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autosummary_generate = True
# Don't fall back to a base class's docstring for undocumented overrides
# (e.g. gymnasium.Env.step) -- upstream docstrings can use RST our own pages
# don't need, and we want this reference to only reflect our own code.
autodoc_inherit_docstrings = False
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
}

# -- HTML output ------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "EMDB Simulator"
