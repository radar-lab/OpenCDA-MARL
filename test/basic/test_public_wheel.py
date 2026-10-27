"""The public wheel carries nothing UCLA-licensed and still runs SUMO.

UCLA's licence forbids transferring the OpenCDA core, or anything derived from
it, to others, so the wheel a package index may carry is built from a tree
without every path ``THIRD_PARTY_NOTICES.md`` lists under UCLA's terms. These
tests derive that list from the notice (never from memory), build the public
wheel, and run a SUMO episode from its unpacked contents with the core
unimportable.
"""
import importlib.util
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / "scripts" / "build_wheel.py"

pytestmark = pytest.mark.skipif(not BUILD.is_file(), reason="not a source checkout")


@pytest.fixture(scope="module")
def build_wheel():
    spec = importlib.util.spec_from_file_location("build_wheel_script", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def public_wheel(build_wheel, tmp_path_factory):
    pytest.importorskip("setuptools")
    out = tmp_path_factory.mktemp("public-wheel")
    wheel = build_wheel.build(out, public=True)
    assert not build_wheel.check(wheel, public=True)
    return wheel


def test_the_notice_names_the_paths_the_public_wheel_must_omit(build_wheel):
    paths = build_wheel.ucla_paths()
    for expected in ("opencda/", "opencda_marl/_bundled/configs/opencda/",
                     "opencda_marl/builtin/agents/vanilla_agent.py",
                     "opencda_marl/builtin/agents/marl_behavior_agent.py",
                     "opencda_marl/builtin/planning/debug_helper.py"):
        assert expected in paths, f"{expected} is no longer listed under UCLA's terms in the notice"


def test_the_public_manifest_drops_the_core_and_the_ucla_licence(build_wheel):
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    public = build_wheel.public_pyproject(text)
    assert 'license = "Apache-2.0 AND MIT"' in public
    assert "UCLA" not in public and "opencda/LICENSE" not in public
    assert 'include = ["opencda_marl*"]' in public
    assert "\nopencda = [" not in public, "the core's package data must not be declared"
    assert "_bundled/configs/**/*.yaml" in public, "the engine's own data must stay"


def test_no_member_of_the_public_wheel_is_ucla_licensed(build_wheel, public_wheel):
    paths = build_wheel.ucla_paths()
    members = [m for m in zipfile.ZipFile(public_wheel).namelist()
               if not m.endswith("/") and ".dist-info/" not in m]
    assert members, "the public wheel is empty"
    assert not [m for m in members if m.startswith("opencda/") or build_wheel.is_under(m, paths)]
    assert any(m.startswith("opencda_marl/_bundled/configs/sumo-marl/") for m in members)
    assert any(m.endswith(".pth") for m in members), "the bundled baselines must ship"


def test_the_public_wheel_declares_only_apache_and_mit(public_wheel):
    archive = zipfile.ZipFile(public_wheel)
    metadata = archive.read(next(n for n in archive.namelist() if n.endswith("METADATA"))).decode()
    assert "License-Expression: Apache-2.0 AND MIT" in metadata
    headers = re.split(r"\r?\n\r?\n", metadata, maxsplit=1)[0]  # the long description (the README) follows
    assert "UCLA" not in headers


def test_a_sumo_episode_runs_from_the_public_wheel_without_the_core(public_wheel, tmp_path):
    pytest.importorskip("traci", reason="SUMO client not installed")
    unpacked = tmp_path / "site"
    zipfile.ZipFile(public_wheel).extractall(unpacked)
    import yaml

    probe = yaml.safe_load((unpacked / "opencda_marl" / "_bundled" / "configs" / "sumo-marl"
                            / "intersection" / "random_speed.yaml").read_text(encoding="utf-8"))
    probe.setdefault("outputs", {})["monitor"] = {"enabled": False}  # leave no monitor process listening
    target = tmp_path / "probe_configs" / "sumo-marl" / "intersection" / "core_free_probe.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(probe), encoding="utf-8")
    script = textwrap.dedent(f"""
        import sys, importlib.abc
        sys.path.insert(0, {str(unpacked)!r})

        class NoCore(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "opencda" or name.startswith("opencda."):
                    raise ImportError(f"imported the OpenCDA core ({{name}})")

        sys.meta_path.insert(0, NoCore())
        import opencda_marl
        assert opencda_marl.__file__.startswith({str(unpacked)!r}), opencda_marl.__file__
        from opencda_marl.cli import main
        raise SystemExit(main(["run", "-t", "sumo-marl/intersection/core_free_probe", "--config-root", "probe_configs"]))
    """)
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                          capture_output=True, text=True, timeout=900)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output[-1500:]
    assert "Episode summary" in output, output[-1500:]
