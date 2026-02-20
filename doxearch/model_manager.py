"""Manage spaCy models: download, install, and load."""

import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

import spacy


class ModelManager:
    """Download, install, and load spaCy models."""

    # Fallback model URLs (used if API query fails)
    _FALLBACK_MODEL_URLS = {
        "en_core_web_sm": {
            "url": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
            "version": "3.8.0",
            "size_mb": 13,
            "language": "English",
            "description": "Small English model for general text processing",
        }
    }

    # Language code to full name mapping
    _LANGUAGE_MAP = {
        "ca": "Catalan",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fi": "Finnish",
        "fr": "French",
        "hr": "Croatian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "lt": "Lithuanian",
        "mk": "Macedonian",
        "nb": "Norwegian Bokmål",
        "nl": "Dutch",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sl": "Slovenian",
        "sv": "Swedish",
        "uk": "Ukrainian",
        "xx": "Multi-language",
        "zh": "Chinese",
    }

    # Model size suffix to description mapping
    _SIZE_MAP = {
        "sm": "Small",
        "md": "Medium",
        "lg": "Large",
        "trf": "Transformer",
    }

    # Model size suffix to estimated size in MB
    _SIZE_ESTIMATES = {
        "sm": 13,
        "md": 50,
        "lg": 550,
        "trf": 450,
    }

    # Common model patterns for fallback detection
    _COMMON_MODEL_PATTERNS = [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_web_trf",
        "lt_core_news_sm",
        "lt_core_news_md",
        "lt_core_news_lg",
        "de_core_news_sm",
        "de_core_news_md",
        "de_core_news_lg",
        "es_core_news_sm",
        "es_core_news_md",
        "es_core_news_lg",
        "fr_core_news_sm",
        "fr_core_news_md",
        "fr_core_news_lg",
        "it_core_news_sm",
        "it_core_news_md",
        "it_core_news_lg",
        "nl_core_news_sm",
        "nl_core_news_md",
        "nl_core_news_lg",
        "pl_core_news_sm",
        "pl_core_news_md",
        "pl_core_news_lg",
        "pt_core_news_sm",
        "pt_core_news_md",
        "pt_core_news_lg",
        "ru_core_news_sm",
        "ru_core_news_md",
        "ru_core_news_lg",
        "zh_core_web_sm",
        "zh_core_web_md",
        "zh_core_web_lg",
        "ja_core_news_sm",
        "ja_core_news_md",
        "ja_core_news_lg",
    ]

    # Required files for a valid model directory
    _REQUIRED_MODEL_FILES = ["meta.json", "config.cfg"]

    def __init__(self, models_dir: Path):
        """
        Initialize the model manager.

        Args:
            models_dir: Directory where models will be stored
        """
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.MODEL_URLS = {}

    @classmethod
    def fetch_available_models(cls, spacy_version: str = "3.8") -> dict:
        """
        Fetch available spaCy models from spaCy's compatibility data.

        Args:
            spacy_version: spaCy version to fetch models for

        Returns:
            Dictionary of model information
        """
        model_urls = {}

        try:
            # Query spaCy's compatibility JSON
            compat_url = "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json"

            with urllib.request.urlopen(compat_url, timeout=10) as response:
                compat_data = json.loads(response.read().decode())

                # Get models for our spaCy version
                spacy_models = compat_data.get("spacy", {}).get(spacy_version, {})

                for model_name, versions in spacy_models.items():
                    # Get the latest version (first in the list)
                    if not versions:
                        continue

                    version = versions[0]

                    # Extract language from model name (e.g., "en" from "en_core_web_sm")
                    language_code = model_name.split("_")[0]
                    language = cls._LANGUAGE_MAP.get(
                        language_code, language_code.upper()
                    )

                    # Generate description based on model name
                    size_suffix = model_name.split("_")[-1]
                    size_desc = cls._SIZE_MAP.get(size_suffix, size_suffix.upper())

                    # Determine model type from name
                    if "core" in model_name:
                        model_type = "core"
                    elif "dep" in model_name:
                        model_type = "dependency"
                    elif "ent" in model_name:
                        model_type = "entity"
                    elif "sent" in model_name:
                        model_type = "sentence"
                    else:
                        model_type = "general"

                    description = f"{size_desc} {language} {model_type} model"

                    # Construct download URL
                    wheel_name = f"{model_name}-{version}-py3-none-any.whl"
                    download_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{version}/{wheel_name}"

                    # Estimate size based on model size suffix
                    estimated_size = cls._SIZE_ESTIMATES.get(size_suffix, 13)

                    model_urls[model_name] = {
                        "url": download_url,
                        "version": version,
                        "size_mb": estimated_size,
                        "language": language,
                        "description": description,
                    }

        except Exception:
            # Fall back to hardcoded URLs
            return cls._FALLBACK_MODEL_URLS.copy()

        # If no models were fetched, use fallbacks
        if not model_urls:
            return cls._FALLBACK_MODEL_URLS.copy()

        return model_urls

    # === Model Information Methods ===

    def is_model_in_downloads(self, model_name: str) -> bool:
        """
        Check if a model exists in the downloaded models directory.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is in the downloaded models directory
        """
        model_dir = self.models_dir / model_name
        if not model_dir.exists():
            return False

        required_files = ["meta.json", "config.cfg"]
        return all((model_dir / f).exists() for f in required_files)

    def is_model_installed(self, model_name: str) -> bool:
        """
        Check if a model is available from any source.

        Checks in order:
        1. Bundled models (in frozen executable)
        2. Downloaded models (in models_dir)
        3. System-wide installed models

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available from any source
        """
        # Check bundled models (frozen executable)
        if getattr(sys, "frozen", False):
            base_path = Path(sys._MEIPASS)
            bundled_model_path = base_path / model_name

            if bundled_model_path.exists():
                required_files = ["meta.json", "config.cfg"]
                if all((bundled_model_path / f).exists() for f in required_files):
                    return True

        # Check downloaded models
        if self.is_model_in_downloads(model_name):
            return True

        # Check system-wide installed models
        try:
            spacy.load(model_name, disable=["parser", "ner", "tagger", "lemmatizer"])
            return True
        except OSError:
            return False

    def is_model_available(self, model_name: str) -> bool:
        """
        Alias for is_model_installed for backward compatibility.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is available from any source
        """
        return self.is_model_installed(model_name)

    def get_model_info(self, model_name: str) -> dict:
        """
        Get information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        # Language code to full name mapping
        language_map = self._LANGUAGE_MAP

        # Start with base info from MODEL_URLS if available
        if model_name in self.MODEL_URLS:
            info = self.MODEL_URLS[model_name].copy()
        else:
            # Create default info for models not in MODEL_URLS
            info = {
                "url": "N/A",
                "version": "Unknown",
                "size_mb": 0,
                "language": "Unknown",
                "description": "Installed model (not in registry)",
            }

        info["name"] = model_name

        # Determine model location
        location = None
        if getattr(sys, "frozen", False):
            base_path = Path(sys._MEIPASS)
            bundled_model_path = base_path / model_name
            if bundled_model_path.exists():
                required_files = ["meta.json", "config.cfg"]
                if all((bundled_model_path / f).exists() for f in required_files):
                    location = "bundled"

        if not location and self.is_model_in_downloads(model_name):
            location = "downloaded"

            # Try to get actual size and version from downloaded model
            model_dir = self.models_dir / model_name
            meta_file = model_dir / "meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta_data = json.load(f)
                        info["version"] = meta_data.get("version", info["version"])

                        # Map language code to full name
                        lang_code = meta_data.get("lang", "")
                        info["language"] = language_map.get(
                            lang_code, lang_code.upper()
                        )

                        if "description" not in self.MODEL_URLS.get(model_name, {}):
                            info["description"] = meta_data.get(
                                "description", info["description"]
                            )
                except Exception:
                    pass

            # Calculate actual size
            model_size = 0
            for item in model_dir.rglob("*"):
                if item.is_file():
                    model_size += item.stat().st_size
            info["size_mb"] = model_size / (1024 * 1024)

        if not location:
            try:
                nlp = spacy.load(
                    model_name, disable=["parser", "ner", "tagger", "lemmatizer"]
                )
                location = "system"

                # Get metadata from loaded model
                meta = nlp.meta
                info["version"] = meta.get("version", info["version"])

                # Map language code to full name
                lang_code = meta.get("lang", "")
                info["language"] = language_map.get(lang_code, lang_code.upper())

                if "description" not in self.MODEL_URLS.get(model_name, {}):
                    info["description"] = meta.get("description", info["description"])
            except OSError:
                pass

        info["location"] = location
        info["installed"] = location is not None

        return info

    def get_all_models_info(self) -> list[dict]:
        """
        Get information about all available models.

        Returns models from:
        1. MODEL_URLS (registry)
        2. Downloaded models not in registry
        3. System-wide installed models not in registry
        """
        all_model_names = set(self.MODEL_URLS.keys())

        # Add downloaded models
        if self.models_dir.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir():
                    required_files = ["meta.json", "config.cfg"]
                    if all((item / f).exists() for f in required_files):
                        all_model_names.add(item.name)

        # Add bundled models (in frozen executable)
        if getattr(sys, "frozen", False):
            base_path = Path(sys._MEIPASS)
            if base_path.exists():
                for item in base_path.iterdir():
                    if item.is_dir():
                        required_files = ["meta.json", "config.cfg"]
                        if all((item / f).exists() for f in required_files):
                            all_model_names.add(item.name)

        # Add system-wide installed models using spaCy's utility
        try:
            import spacy.util

            # Get all installed models from spaCy
            installed_models = spacy.util.get_installed_models()
            all_model_names.update(installed_models)
        except Exception:
            # Fallback: check common model patterns if spaCy utility fails
            common_model_patterns = self._COMMON_MODEL_PATTERNS

            for model_name in common_model_patterns:
                if model_name not in all_model_names:
                    try:
                        # Quick check without loading the full model
                        spacy.util.get_package_path(model_name)
                        all_model_names.add(model_name)
                    except Exception:
                        pass

        return [self.get_model_info(name) for name in sorted(all_model_names)]

    def get_models_directory_size(self) -> int:
        """Get total size of models directory in bytes."""
        total_size = 0
        for item in self.models_dir.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        return total_size

    # === Model Download Methods ===

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[bool, str]:
        """
        Download and extract a spaCy model.

        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback(downloaded_bytes, total_bytes, status_message)

        Returns:
            Tuple of (success, message)
        """
        if model_name not in self.MODEL_URLS:
            return False, f"Unknown model: {model_name}"

        # Check if already in downloads directory
        if self.is_model_in_downloads(model_name):
            return True, f"Model {model_name} is already in the downloads directory"

        model_info = self.MODEL_URLS[model_name]
        url = model_info["url"]
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".whl") as temp_file:
                temp_path = Path(temp_file.name)

            if progress_callback:
                progress_callback(0, 100, f"Downloading {model_name}...")

            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if progress_callback and total_size > 0:
                    progress_callback(
                        downloaded, total_size, f"Downloading {model_name}..."
                    )

            urllib.request.urlretrieve(url, temp_path, reporthook=download_progress)

            if progress_callback:
                progress_callback(0, 100, f"Extracting {model_name}...")

            model_dir = self.models_dir / model_name

            # Remove existing directory if it exists
            if model_dir.exists():
                shutil.rmtree(model_dir)

            model_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as temp_extract_dir:
                temp_extract_path = Path(temp_extract_dir)

                # Extract the wheel file
                with zipfile.ZipFile(temp_path, "r") as zip_ref:
                    zip_ref.extractall(temp_extract_path)

                # Find the directory containing config.cfg (the actual model data)
                source_dir = None

                for item in temp_extract_path.rglob("config.cfg"):
                    if item.is_file():
                        # The parent of config.cfg is the model directory
                        source_dir = item.parent
                        break

                if not source_dir:
                    # List what we found for debugging
                    found_items = []
                    for item in temp_extract_path.rglob("*"):
                        found_items.append(str(item.relative_to(temp_extract_path)))
                    return False, (
                        f"Could not find model files in downloaded package. "
                        f"Found items: {found_items[:20]}"  # Limit to first 20 items
                    )

                # Copy all files and directories from source to destination
                for item in source_dir.iterdir():
                    dest_item = model_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest_item)
                    elif item.is_dir():
                        if dest_item.exists():
                            shutil.rmtree(dest_item)
                        shutil.copytree(item, dest_item)

            # Clean up temporary file
            if temp_path and temp_path.exists():
                temp_path.unlink()

            # Verify installation
            if self.is_model_in_downloads(model_name):
                if progress_callback:
                    progress_callback(
                        100, 100, f"✓ Successfully installed {model_name}"
                    )
                return True, f"Successfully installed {model_name}"
            else:
                # Check what files we actually have
                if model_dir.exists():
                    files = list(model_dir.iterdir())
                    return False, (
                        f"Installation verification failed for {model_name}. "
                        f"Files in directory: {[f.name for f in files]}"
                    )
                else:
                    return (
                        False,
                        f"Installation verification failed for {model_name}. Directory not created.",
                    )

        except Exception as e:
            if temp_path and temp_path.exists():
                temp_path.unlink()

            error_msg = f"Failed to download {model_name}: {str(e)}"
            if progress_callback:
                progress_callback(0, 100, f"✗ {error_msg}")
            return False, error_msg

    def delete_model(self, model_name: str) -> tuple[bool, str]:
        """
        Delete a downloaded model.

        Args:
            model_name: Name of the model to delete

        Returns:
            Tuple of (success, message)
        """
        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            return False, f"Model {model_name} is not installed"

        try:
            shutil.rmtree(model_dir)
            return True, f"Successfully deleted {model_name}"
        except Exception as e:
            return False, f"Failed to delete {model_name}: {str(e)}"

    # === Model Loading Methods ===

    def load_model(self, model_name: str, disable: Optional[list[str]] = None):
        """
        Load a spaCy model from various sources.

        Priority:
        1. Bundled models (in frozen executable)
        2. Downloaded models (in models_dir)
        3. Installed models (system-wide)

        Args:
            model_name: Name of the model to load
            disable: Pipeline components to disable

        Returns:
            Loaded spaCy model

        Raises:
            ValueError: If model cannot be found or loaded
        """
        if disable is None:
            disable = ["parser", "ner"]

        # Try bundled models (frozen executable)
        if getattr(sys, "frozen", False):
            base_path = Path(sys._MEIPASS)
            bundled_model_path = base_path / model_name

            if bundled_model_path.exists():
                try:
                    return spacy.load(bundled_model_path, disable=disable)
                except Exception:
                    pass  # Try next option

        # Try downloaded models
        downloaded_model_path = self.models_dir / model_name

        if downloaded_model_path.exists():
            try:
                return spacy.load(downloaded_model_path, disable=disable)
            except Exception:
                pass  # Try next option

        # Try system-wide installed models
        try:
            return spacy.load(model_name, disable=disable)
        except OSError:
            pass

        # Model not found anywhere
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Please download it using the Model Manager."
        )
