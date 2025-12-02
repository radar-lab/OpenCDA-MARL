#!/usr/bin/env python
"""
Fix Open3D TensorBoard plugin compatibility issue.

Open3D 0.19.0 has a TensorBoard plugin that tries to import O3DVisualizer,
which doesn't exist in some builds. This script:
1. Removes the plugin entry point from dist-info
2. Patches the plugin file to handle missing O3DVisualizer gracefully

Usage:
    python scripts/fix_tensorboard_open3d.py

Or with pixi:
    pixi run fix-tensorboard
"""

import sys
import site
from pathlib import Path


def find_site_packages() -> list[Path]:
    """Find all site-packages directories."""
    site_packages_dirs = []
    
    # Get from site module
    try:
        site_packages_dirs.extend(site.getsitepackages())
    except:
        pass
    
    if site.ENABLE_USER_SITE:
        try:
            site_packages_dirs.append(site.getusersitepackages())
        except:
            pass
    
    # Check sys.path for site-packages directories
    for path in sys.path:
        if "site-packages" in path:
            site_packages_dirs.append(path)
    
    # Also check sys.prefix
    site_packages_dirs.extend([
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.prefix) / "Lib" / "site-packages",  # Windows
    ])
    
    # Remove duplicates and convert to Path objects
    seen = set()
    result = []
    for sp in site_packages_dirs:
        sp_path = Path(sp)
        sp_resolved = str(sp_path.resolve())
        if sp_resolved not in seen:
            seen.add(sp_resolved)
            result.append(sp_path)
    
    return result


def find_open3d_dist_info() -> Path | None:
    """Find the Open3D dist-info directory in site-packages."""
    for site_packages in find_site_packages():
        if site_packages.is_dir():
            # Try both open3d and open3d_cpu patterns
            for pattern in ["open3d*.dist-info"]:
                for dist_info in site_packages.glob(pattern):
                    if dist_info.is_dir() and "open3d" in dist_info.name.lower():
                        return dist_info
    
    return None


def find_open3d_package() -> Path | None:
    """Find the Open3D package directory in site-packages."""
    for site_packages in find_site_packages():
        if site_packages.is_dir():
            open3d_dir = site_packages / "open3d"
            if open3d_dir.is_dir():
                return open3d_dir
    
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

    print(f"✓ Fixed: {entry_points_file}")
    print("✓ Removed [tensorboard_plugins] section from Open3D")
    return True


def fix_plugin_file(open3d_dir: Path) -> bool:
    """Patch the TensorBoard plugin file to handle missing O3DVisualizer."""
    plugin_file = open3d_dir / "visualization" / "tensorboard_plugin" / "plugin.py"

    if not plugin_file.exists():
        print(f"Plugin file not found at {plugin_file} - skipping")
        return True

    content = plugin_file.read_text()

    # Check if already patched
    if "# PATCHED: Disable Open3D TensorBoard plugin" in content:
        print(f"Plugin file already patched: {plugin_file}")
        return True

    # Replace entire content to disable the plugin gracefully
    new_content = """# PATCHED: Disable Open3D TensorBoard plugin
# The Open3D TensorBoard plugin has compatibility issues with the current
# Open3D build. This disables it while keeping TensorBoard functional.
raise ImportError("Open3D TensorBoard plugin disabled due to compatibility issues. This is normal and TensorBoard will continue to work.")
"""

    plugin_file.write_text(new_content)
    print(f"✓ Patched: {plugin_file}")
    return True


def main() -> int:
    print("Fixing Open3D TensorBoard plugin compatibility issue...")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version_info.major}.{sys.version_info.minor}")
    print()

    # First try to find and patch the plugin file directly
    open3d_dir = find_open3d_package()
    if open3d_dir is None:
        print("❌ Open3D package not found!")
        print("Searched in:")
        for sp in find_site_packages():
            print(f"  {sp}")
        return 1

    print(f"✓ Found Open3D package at: {open3d_dir}")
    print()

    # Fix plugin file first (most important)
    if not fix_plugin_file(open3d_dir):
        print("⚠ Warning: Failed to patch plugin file")
    
    print()

    # Fix entry points if dist-info exists
    dist_info = find_open3d_dist_info()
    if dist_info is None:
        print("⚠ Warning: Open3D dist-info not found - skipping entry points removal")
        print("  (This is okay if you're using a pixi/conda environment)")
    else:
        print(f"✓ Found Open3D dist-info at: {dist_info}")
        if not fix_entry_points(dist_info):
            print("⚠ Warning: Failed to fix entry points")

    print()
    print("✓ Success! TensorBoard should now start without Open3D plugin errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
