"""The SUMO path runs without the vendored OpenCDA core.

The public wheel ships no ``opencda/`` (its licence does not allow the core
to travel in an uploaded package), so a SUMO run must never import it. This
runs a short bundled scenario in a process where importing ``opencda`` raises,
and fails with the first module that reached for it.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

pytest.importorskip("traci", reason="SUMO client not installed")

SCRIPT = textwrap.dedent("""
    import sys, importlib.abc

    class NoCore(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "opencda" or name.startswith("opencda."):
                caller = sys._getframe(1)
                while caller and "importlib" in caller.f_code.co_filename:
                    caller = caller.f_back
                where = f"{caller.f_code.co_filename}:{caller.f_lineno}" if caller else "?"
                raise ImportError(f"the SUMO path imported the OpenCDA core ({name}) at {where}")

    sys.meta_path.insert(0, NoCore())
    from opencda_marl.cli import main
    raise SystemExit(main(["run", "-t", "sumo-marl/intersection/core_free_probe", "--config-root", "probe_configs"]))
""")


def write_probe_profile(root, bundled_profile):
    """A copy of the bundled random-speed profile with the web monitor off, so the
    run leaves no monitor process listening behind it."""
    config = yaml.safe_load(bundled_profile.read_text(encoding="utf-8"))
    config.setdefault("outputs", {})["monitor"] = {"enabled": False}
    target = root / "probe_configs" / "sumo-marl" / "intersection" / "core_free_probe.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_a_sumo_episode_runs_with_the_core_unimportable(tmp_path):
    import opencda_marl

    write_probe_profile(tmp_path, Path(opencda_marl.__file__).parent / "_bundled" / "configs" / "sumo-marl"
                        / "intersection" / "random_speed.yaml")
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=tmp_path,
                          capture_output=True, text=True, timeout=600)
    output = done.stdout + done.stderr
    assert "the SUMO path imported the OpenCDA core" not in output, output[-1500:]
    assert done.returncode == 0, output[-1500:]
    assert "Episode summary" in output, output[-1500:]


def test_the_carla_backend_names_the_missing_core(monkeypatch):
    from opencda_marl._kernel.lifecycle.coordinator import require_opencda_core

    monkeypatch.setitem(sys.modules, "opencda", None)  # makes `import opencda` raise ImportError
    with pytest.raises(ImportError, match="repository checkout"):
        require_opencda_core()
