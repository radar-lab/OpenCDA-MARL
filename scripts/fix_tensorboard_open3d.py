#!/usr/bin/env python
"""
Fix Open3D TensorBoard plugin compatibility issue.

Open3D 0.19.0 has a TensorBoard plugin that tries to import O3DVisualizer,
which doesn't exist in some builds. This script removes the plugin entry point
to allow TensorBoard to start without errors.

Usage:
    python scripts/fix_tensorboard_open3d.py

Or with pixi:
    pixi run fix-tensorboard
"""

import sys
from pathlib import Path


def find_open3d_dist_info() -> Path | None:
    """Find the Open3D dist-info directory in site-packages."""
    for site_packages in sys.path:
        site_path = Path(site_packages)
        if site_path.is_dir():
            for dist_info in site_path.glob("open3d-*.dist-info"):
                if dist_info.is_dir():
                    return dist_info
    return None


def fix_entry_points(dist_info: Path) -> bool:
    """Remove TensorBoard plugin from Open3D entry points."""
    entry_points_file = dist_info / "entry_points.txt"

    if not entry_points_file.exists():
        print(f"No entry_points.txt found in {dist_info}")
        return False

    content = entry_points_file.read_text()

    if "[tensorboard_plugins]" not in content:
        print("TensorBoard plugin entry not found - already fixed or not present")
        return True

    # Keep only console_scripts section
    lines = content.strip().split("\n")
    new_lines = []
    skip_section = False

    for line in lines:
        if line.strip() == "[tensorboard_plugins]":
            skip_section = True
            continue
        elif line.strip().startswith("[") and skip_section:
            skip_section = False

        if not skip_section:
            new_lines.append(line)

    new_content = "\n".join(new_lines).strip() + "\n"
    entry_points_file.write_text(new_content)

    print(f"Fixed: {entry_points_file}")
    print("Removed [tensorboard_plugins] section from Open3D")
    return True


def main() -> int:
    print("Fixing Open3D TensorBoard plugin compatibility issue...")

    dist_info = find_open3d_dist_info()

    if dist_info is None:
        print("Open3D dist-info not found. Is Open3D installed?")
        return 1

    print(f"Found Open3D at: {dist_info}")

    if fix_entry_points(dist_info):
        print("\nSuccess! TensorBoard should now start without Open3D plugin errors.")
        return 0
    else:
        print("\nFailed to fix entry points.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
