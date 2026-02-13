import hashlib
from pathlib import Path


def get_db_path_for_directory(directory_path: str, app_data_dir: Path) -> Path:
    """
    Generate a unique database path for a given directory.

    Args:
        directory_path: Absolute path to the indexed directory
        app_data_dir: Application data directory

    Returns:
        Path to the database file for this directory
    """
    # Create a hash of the directory path for uniqueness
    dir_hash = hashlib.sha256(directory_path.encode()).hexdigest()[:16]

    # Create a sanitized name from the directory
    dir_name = Path(directory_path).name
    sanitized_name = "".join(c if c.isalnum() else "_" for c in dir_name)

    # Combine for a readable yet unique name
    db_name = f"doxearch_{sanitized_name}_{dir_hash}.db"

    return app_data_dir / "indexes" / db_name
