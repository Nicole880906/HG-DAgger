"""Decide which test files can run on this interpreter, and say so.

No single environment here has everything.  The dVRK desktop's ROS install has
rclpy and OpenCV but no torch; the training container has torch, zarr and
pytorch3d but no rclpy.  So parts of this suite are expected to be unrunnable
wherever you run it, and the run has to stay green and honest about what it
skipped.

Why ``collect_ignore`` rather than ``pytest.importorskip`` at the top of each
file: on the pytest shipped with ROS 2 Humble (6.2.5, with the ament plugins
loaded), a module-level skip raised during collection aborts collection of the
whole directory -- the run reports "1 skipped, no tests collected" and every
other file silently never runs.  That failure is invisible in CI, because a run
that collects nothing still exits 0.  ``collect_ignore`` is evaluated before any
test module is imported, so it cannot do that.

The header below prints what was left out and why, so a run that quietly covers
half of what you expected says so.
"""

from __future__ import annotations

import importlib

# Each file, the modules it needs, and where you would go to run it.
REQUIREMENTS = {
    "test_pedal.py": (("rclpy", "sensor_msgs"), "ROS 2 (source /opt/ros/humble/setup.bash)"),
    "test_run_recorder.py": (("cv2",), "OpenCV"),
    "test_geometry.py": (("scipy",), "anywhere with scipy"),
    "test_pipeline_integration.py": (("zarr", "torch", "pytorch3d", "cv2"),
                                     "the training container (bash start_container.sh)"),
    # test_handover_machine.py needs nothing but the standard library.
}

collect_ignore: list[str] = []
_skipped: list[tuple[str, list[str], str]] = []


_probe_cache: dict[str, bool] = {}


def _importable(name: str) -> bool:
    """Actually import the module -- ``find_spec`` is not enough.

    Inside the training container, ROS's Python 3.10 site-packages are on
    PYTHONPATH while the interpreter is conda's 3.9. ``find_spec("rclpy")``
    happily finds the pure-Python package there and reports success; the import
    then fails on a C extension built for the wrong Python. A find_spec-based
    check therefore collects the ROS tests in the one environment that cannot
    run them, and the whole run ends in a collection error.
    """
    if name not in _probe_cache:
        try:
            importlib.import_module(name)
            _probe_cache[name] = True
        except Exception:  # noqa: BLE001 - any import failure means "not usable here"
            _probe_cache[name] = False
    return _probe_cache[name]


def _missing(modules: tuple[str, ...]) -> list[str]:
    return [name for name in modules if not _importable(name)]


for _filename, (_modules, _where) in REQUIREMENTS.items():
    _gone = _missing(_modules)
    if _gone:
        collect_ignore.append(_filename)
        _skipped.append((_filename, _gone, _where))


def pytest_report_header(config):
    if not _skipped:
        return "expert_intervention: all test files runnable here"
    lines = ["expert_intervention: some test files are not runnable on this interpreter"]
    for filename, missing, where in _skipped:
        lines.append(f"  {filename}: needs {', '.join(missing)} -- run it in {where}")
    return lines
