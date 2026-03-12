
import os
import platform
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox


def open_file(filepath: str, parent_widget=None):
    """Open a file with the system's default application.
    
    Args:
        filepath: Path to the file to open
        parent_widget: Parent widget for error dialogs
    """
    if not Path(filepath).exists():
        if parent_widget:
            QMessageBox.warning(
                parent_widget,
                "File Not Found",
                f"The file does not exist:\n{filepath}",
            )
        return

    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(filepath)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", filepath], check=True)
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", filepath], check=True)
    except Exception as e:
        if parent_widget:
            QMessageBox.critical(
                parent_widget,
                "Error Opening File",
                f"Failed to open file:\n{filepath}\n\nError: {str(e)}",
            )


def open_folder(filepath: str, parent_widget=None):
    """Open the folder containing the specified file.
    
    Args:
        filepath: Path to the file whose folder should be opened
        parent_widget: Parent widget for error dialogs
    """
    folder_path = str(Path(filepath).parent)

    if not Path(folder_path).exists():
        if parent_widget:
            QMessageBox.warning(
                parent_widget,
                "Folder Not Found",
                f"The folder does not exist:\n{folder_path}",
            )
        return

    try:
        system = platform.system()
        if system == "Windows":
            # Use explorer with /select to highlight the file
            subprocess.run(["explorer", "/select,", filepath], check=True)
        elif system == "Darwin":  # macOS
            # Use open -R to reveal the file in Finder
            subprocess.run(["open", "-R", filepath], check=True)
        else:  # Linux and other Unix-like systems
            # Just open the folder (file managers vary too much to select the file)
            subprocess.run(["xdg-open", folder_path], check=True)
    except Exception as e:
        if parent_widget:
            QMessageBox.critical(
                parent_widget,
                "Error Opening Folder",
                f"Failed to open folder:\n{folder_path}\n\nError: {str(e)}",
            )
