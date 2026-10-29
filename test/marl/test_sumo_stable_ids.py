"""SUMO edge and junction ids must not depend on the process's hash seed.

The planner breaks ties by ``road_id``; with ``hash(edge_id)`` the same frozen
cell gave different counters under different PYTHONHASHSEED values.
"""
import os
import subprocess
import sys

import pytest

CODE = (
    "from opencda_marl._kernel.traffic import sumo_adapter as m;"
    "print(m._stable_id('-E12'), m._stable_id('E3'))"
)


def _ids(hash_seed):
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_edge_ids_are_the_same_under_different_hash_seeds():
    pytest.importorskip("opencda_marl._kernel.traffic.sumo_adapter")
    assert _ids(1) == _ids(2) == _ids(3)
