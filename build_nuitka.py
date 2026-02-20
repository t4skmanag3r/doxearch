"""
Optimized build script for packaging Doxearch GUI with Nuitka.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path


def clean_build_dirs():
    """Clean previous build directories."""
    patterns = [
        "*.dist",
        "*.build",
        "*.onefile-build",
        "doxearch-gui",
        "doxearch-gui.exe",
    ]

    for pattern in patterns:
        for path in Path.cwd().glob(pattern):
            if path.is_dir():
                print(f"Cleaning {path.name}...")
                shutil.rmtree(path)
            elif path.is_file() and path.name.startswith("doxearch-gui"):
                print(f"Removing old executable {path.name}...")
                path.unlink()


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import zstandard

        print("✓ zstandard installed (compression enabled)")
    except ImportError:
        print(
            "⚠ Warning: zstandard not installed. Install with: uv add --dev zstandard"
        )
        print("  Executable will be larger without compression.")

    # Check for ccache
    if shutil.which("ccache"):
        print("✓ ccache found (faster rebuilds enabled)")
    else:
        print("⚠ Warning: ccache not found. Install for faster rebuilds:")
        print("  Ubuntu/Debian: sudo apt-get install ccache")
        print("  macOS: brew install ccache")


def build_executable():
    """Build the executable using Nuitka."""

    print("=" * 60)
    print("Doxearch GUI - Nuitka Build")
    print("=" * 60)

    # Check dependencies
    check_dependencies()
    print()

    # Clean previous builds
    clean_build_dirs()
    print()

    system = platform.system()

    # Base Nuitka command
    nuitka_cmd = [
        sys.executable,
        "-m",
        "nuitka",
        # Output settings
        "--standalone",
        "--onefile",
        # GUI settings
        (
            "--disable-console"
            if system == "Linux"
            else (
                "--windows-disable-console"
                if system == "Windows"
                else "--macos-disable-console"
            )
        ),
        # Application settings
        "--enable-plugin=pyqt6",
        # spaCy language models (include all installed)
        "--spacy-language-model=all",
        # Output name
        "--output-filename=doxearch-gui" + (".exe" if system == "Windows" else ""),
        # Include package data
        "--include-package=doxearch",
        "--include-package=doxearch_gui",
        "--include-package=doxearch_cli",
        # Performance optimizations
        "--lto=yes",
        "--jobs=4",  # Use 4 parallel jobs
        # Clean build directories after successful build
        "--remove-output",
        # Assume yes for downloads
        "--assume-yes-for-downloads",
        # Entry point
        "doxearch_gui/main.py",
    ]

    print("Building executable with Nuitka...")
    print(f"Platform: {system}")
    print(f"Command: {' '.join(nuitka_cmd)}")
    print()

    try:
        subprocess.run(nuitka_cmd, check=True)

        # Make executable on Linux/macOS
        exe_name = "doxearch-gui.exe" if system == "Windows" else "doxearch-gui"
        exe_path = Path.cwd() / exe_name

        if system != "Windows" and exe_path.exists():
            exe_path.chmod(0o755)

        print()
        print("=" * 60)
        print("✓ Build completed successfully!")
        print("=" * 60)
        print(f"Executable: {exe_path}")

        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"Size: {size_mb:.2f} MB")

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"✗ Build failed with error: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    build_executable()
