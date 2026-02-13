import platform
from pathlib import Path


def get_app_data_dir() -> Path:
    """Get the appropriate application data directory based on the operating system."""
    system = platform.system()

    if system == "Windows":
        # Windows: %APPDATA%\doxearch
        app_data = Path.home() / "AppData" / "Roaming" / "doxearch"
    elif system == "Linux":
        # Linux: ~/.local/share/doxearch
        app_data = Path.home() / ".local" / "share" / "doxearch"
    else:
        # Fallback to home directory
        app_data = Path.home() / ".doxearch"

    # Create directory if it doesn't exist
    app_data.mkdir(parents=True, exist_ok=True)

    return app_data
