"""Build the engine wheel from a clean tree and verify every member has a source.

Two variants come from one source tree:

* the **full** wheel (default) carries the vendored OpenCDA core beside
  ``opencda_marl``. It is for installing from Git (``pip install git+...``) and
  for local use, and it is never uploaded to a package index: UCLA's licence does
  not allow the core, or anything derived from it, to be transferred to others.
* the **public** wheel (``--public``) is what a package index may carry. It is
  built from a temporary tree without any path that ``THIRD_PARTY_NOTICES.md``
  lists under UCLA's terms (the core, the copied OpenCDA configs, the three files
  with adapted OpenCDA code), its metadata says ``Apache-2.0 AND MIT``, and it has
  no source distribution. The SUMO benchmark, the library, the plugins and the
  bundled baselines run on it; CARLA and the OpenCDA scenarios need a ``git clone``.

setuptools copies package data into ``build/lib`` incrementally and never
prunes a file that was deleted from the source tree, so a wheel built from
a long-lived checkout can ship files that no longer exist, and any project
that installed such a wheel could run profiles the framework no longer
defines. ``build/`` is git-ignored, so a fresh clone never shows the
problem and a working checkout always can.

So: remove ``build/`` and the egg-info first, build, then open the wheel
and refuse it if any member has no file behind it in the source tree (and, for
the public wheel, if any member sits under a UCLA path).

    pixi run wheel                    # -> dist/opencda_marl-<version>-py3-none-any.whl (full)
    pixi run wheel some/dir           # -> that directory instead
    pixi run wheel --public [dir]     # -> dist/public/, the wheel without the UCLA-licensed paths
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTICES = REPO / "THIRD_PARTY_NOTICES.md"
#: Files copied beside ``opencda_marl`` into the public build tree.
PUBLIC_TOP_LEVEL = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "README.md")


def ucla_paths(notices: Path = NOTICES) -> list[str]:
    """The repository paths ``THIRD_PARTY_NOTICES.md`` lists under UCLA's terms.

    Read from the notice itself (section 1, "Paths under these terms") so the
    public wheel can never drift from what the notice says: a backticked token
    with a slash, under ``opencda/``, ``opencda_marl/`` or ``opencda.py``.
    """
    text = notices.read_text(encoding="utf-8")
    start = text.index("Paths under these terms:")
    end = text.index("The terms of these files are", start)
    paths = []
    for token in re.findall(r"`([^`\n]+)`", text[start:end]):
        if token == "opencda.py" or (("/" in token) and token.split("/", 1)[0] in ("opencda", "opencda_marl")):
            paths.append(token)
    return sorted(set(paths))


def is_under(member: str, paths: list[str]) -> bool:
    """Whether the wheel member (a repository-relative path) is one of ``paths``
    or lies inside one that names a directory."""
    for path in paths:
        if member == path or (path.endswith("/") and member.startswith(path)):
            return True
    return False


def public_pyproject(text: str) -> str:
    """The build manifest of the public wheel: no core, no UCLA licence."""
    def swap(pattern: str, replacement: str, source: str) -> str:
        result, count = re.subn(pattern, replacement, source, count=1, flags=re.S | re.M)
        if count != 1:
            raise SystemExit(f"pyproject.toml no longer matches {pattern!r}; update scripts/build_wheel.py")
        return result

    text = swap(r'^# The wheel carries three licences:.*?(?=^license = )',
                "# The public wheel carries OpenCDA-MARL's own files (Apache-2.0) and the MIT\n"
                "# files derived from CARLA; see THIRD_PARTY_NOTICES.md.\n", text)
    text = swap(r'^license = ".*?"$', 'license = "Apache-2.0 AND MIT"', text)
    text = swap(r'^license-files = \[.*?\]$', 'license-files = ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]', text)
    text = swap(r'^# The vendored OpenCDA core ships with the engine:.*?(?=^include = )', "", text)
    text = swap(r'include = \["opencda_marl\*", "opencda\*"\]', 'include = ["opencda_marl*"]', text)
    text = swap(r"^# The vendored core's data:.*?^\]\n", "", text)
    return text


def _prepare_public_tree(tree: Path, excluded: list[str]) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "opencda_marl", tree / "opencda_marl", ignore=ignore)
    for path in excluded:
        if path.startswith("opencda_marl/"):
            target = tree / path
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    for name in PUBLIC_TOP_LEVEL:
        shutil.copy2(REPO / name, tree / name)
    (tree / "pyproject.toml").write_text(
        public_pyproject((REPO / "pyproject.toml").read_text(encoding="utf-8")), encoding="utf-8")


def build(out: Path, public: bool) -> Path:
    for stale in [REPO / "build", *REPO.glob("*.egg-info")]:
        shutil.rmtree(stale, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("opencda_marl-*.whl"):
        old.unlink()

    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(out)]
    if public:
        excluded = ucla_paths()
        with tempfile.TemporaryDirectory(prefix="opencda-marl-public-") as scratch:
            tree = Path(scratch)
            _prepare_public_tree(tree, excluded)
            subprocess.run([*command, str(tree)], check=True, cwd=tree)
    else:
        subprocess.run([*command, str(REPO)], check=True, cwd=REPO)
    return max(out.glob("opencda_marl-*.whl"), key=lambda p: p.stat().st_mtime)


def check(wheel: Path, public: bool) -> list[str]:
    """Reasons to refuse ``wheel``; empty when it is sound."""
    problems = []
    members = [m for m in zipfile.ZipFile(wheel).namelist() if not m.endswith("/")]
    orphans = [m for m in members if ".dist-info/" not in m and not (REPO / m).is_file()]
    if orphans:
        problems.append(f"{len(orphans)} member(s) have no source file, so this build tree was stale "
                        f"even after cleaning: {orphans[:20]}")
    if public:
        excluded = ucla_paths()
        leaked = [m for m in members if ".dist-info/" not in m and (m.startswith("opencda/") or is_under(m, excluded))]
        if leaked:
            problems.append(f"UCLA-licensed paths in the public wheel: {leaked[:20]}")
    return problems


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--public"]
    public = "--public" in argv[1:]
    # Both variants carry the same file name, so the public one defaults to its own
    # directory: a full wheel must never be mistaken for the one a package index may carry.
    out = Path(args[0]).resolve() if args else REPO / "dist" / ("public" if public else "")

    wheel = build(out, public)
    problems = check(wheel, public)
    if problems:
        print(f"{wheel.name}: refused", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        wheel.unlink()
        return 1

    count = len([m for m in zipfile.ZipFile(wheel).namelist() if not m.endswith("/")])
    size_mb = wheel.stat().st_size / 1048576
    kind = "public, without the UCLA-licensed paths" if public else "full, with the vendored core"
    print(f"{wheel}  ({size_mb:.1f} MB, {count} files, {kind}; every member has a source file)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
