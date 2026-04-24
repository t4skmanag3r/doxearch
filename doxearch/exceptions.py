class ContextManagerError(Exception):
    """Base exception for all ContextManager errors"""


class InvalidDirectoryPathError(ContextManagerError):
    """Raised when an invalid directory path is provided"""

    def __init__(self, directory_path):
        self.directory_path = directory_path
        super().__init__(f"Invalid directory path: {directory_path}")


class DirectoryAlreadyIndexedError(ContextManagerError):
    """Raised when attempting to add a directory that's already indexed"""

    def __init__(self, directory_path):
        self.directory_path = directory_path
        super().__init__(f"Directory already indexed: {directory_path}")


class DirectoryNotFoundError(ContextManagerError):
    """Raised when a directory is not found in the index"""

    def __init__(self, directory_path):
        self.directory_path = directory_path
        super().__init__(f"Directory not found in index: {directory_path}")


class DirectoryDoesntExistError(ContextManagerError):
    """Raised when a directory does not exist on the system"""

    def __init__(self, directory_path):
        self.directory_path = directory_path
        super().__init__(f"Directory does not exist: {directory_path}")


class EmptyDirectoryError(ContextManagerError):
    """Raised when an empty directory is provided"""

    def __init__(self, directory_path):
        self.directory_path = directory_path
        super().__init__(f"Directory is empty: {directory_path}")


class NoSupportedFilesFoundError(ContextManagerError):
    """Raised when no supported files are found in a directory"""

    def __init__(self, directory_path):
        self.directory_path = directory_path
        super().__init__(f"No supported files found in directory: {directory_path}")
