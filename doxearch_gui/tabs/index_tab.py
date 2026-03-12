from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex
from doxearch.doxearch import Doxearch
from doxearch.exceptions import DirectoryAlreadyIndexedError
from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer
from doxearch.utils.general import get_db_path_for_directory
from doxearch_gui.workers.index_worker import IndexWorker


class IndexTab(QWidget):
    """Index tab for indexing documents."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.index_worker: Optional[IndexWorker] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder to index...")
        folder_layout.addWidget(self.folder_input)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_button)

        layout.addLayout(folder_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.populate_model_combo()
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Tokenization options
        tokenization_label = QLabel("Tokenization Options:")
        tokenization_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(tokenization_label)

        # Lemmatization checkbox
        self.lemmatization_checkbox = QCheckBox("Enable Lemmatization")
        self.lemmatization_checkbox.setChecked(True)
        self.lemmatization_checkbox.setToolTip(
            "Convert words to their base form (e.g., 'running' → 'run')"
        )
        self.lemmatization_checkbox.stateChanged.connect(
            self.on_tokenization_option_changed
        )
        layout.addWidget(self.lemmatization_checkbox)

        # Stemming checkbox
        self.stemming_checkbox = QCheckBox("Enable Stemming")
        self.stemming_checkbox.setChecked(False)
        self.stemming_checkbox.setToolTip(
            "Apply stemming algorithm to reduce words to their root form"
        )
        self.stemming_checkbox.stateChanged.connect(self.on_tokenization_option_changed)
        layout.addWidget(self.stemming_checkbox)

        # Warning label for XOR constraint
        self.tokenization_warning_label = QLabel()
        self.tokenization_warning_label.setStyleSheet(
            "color: #FF9800; font-style: italic;"
        )
        self.tokenization_warning_label.setWordWrap(True)
        self.tokenization_warning_label.setVisible(False)
        layout.addWidget(self.tokenization_warning_label)

        # Force re-index checkbox
        self.force_checkbox = QCheckBox("Force re-indexing")
        layout.addWidget(self.force_checkbox)

        # Index button
        self.index_button = QPushButton("Start Indexing")
        self.index_button.clicked.connect(self.start_indexing)
        layout.addWidget(self.index_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status area
        layout.addWidget(QLabel("Status:"))
        self.index_status_text = QTextEdit()
        self.index_status_text.setReadOnly(True)
        layout.addWidget(self.index_status_text)

        layout.addStretch()

    def populate_model_combo(self):
        """Populate the model combo box with available models."""
        # Clear existing items
        self.model_combo.clear()

        available_models = [
            model_info["name"]
            for model_info in self.parent.model_manager.get_all_models_info()
            if self.parent.model_manager.is_model_available(model_info["name"])
        ]

        if available_models:
            self.model_combo.addItems(available_models)
            self.model_combo.setEnabled(True)
        else:
            self.model_combo.addItem("No models available")
            self.model_combo.setEnabled(False)

    def on_tokenization_option_changed(self):
        """Handle changes to tokenization options (enforce XOR constraint)."""
        lemmatization_enabled = self.lemmatization_checkbox.isChecked()
        stemming_enabled = self.stemming_checkbox.isChecked()

        # If both are checked, show warning and uncheck the other one
        if lemmatization_enabled and stemming_enabled:
            # Determine which one was just checked
            sender = self.sender()
            if sender == self.lemmatization_checkbox:
                # Lemmatization was just checked, uncheck stemming
                self.stemming_checkbox.blockSignals(True)
                self.stemming_checkbox.setChecked(False)
                self.stemming_checkbox.blockSignals(False)
                self.tokenization_warning_label.setText(
                    "⚠ Lemmatization and stemming are mutually exclusive. Stemming has been disabled."
                )
            else:
                # Stemming was just checked, uncheck lemmatization
                self.lemmatization_checkbox.blockSignals(True)
                self.lemmatization_checkbox.setChecked(False)
                self.lemmatization_checkbox.blockSignals(False)
                self.tokenization_warning_label.setText(
                    "⚠ Lemmatization and stemming are mutually exclusive. Lemmatization has been disabled."
                )
            self.tokenization_warning_label.setVisible(True)
        else:
            self.tokenization_warning_label.setVisible(False)

    def browse_folder(self):
        """Open folder browser dialog."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Index", str(Path.home())
        )

        if folder:
            self.folder_input.setText(folder)

    def start_indexing(self):
        """Start the indexing process."""
        folder_path = self.folder_input.text().strip()

        if not folder_path:
            QMessageBox.warning(
                self, "No Folder Selected", "Please select a folder to index."
            )
            return

        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Folder",
                "The selected folder does not exist or is not a directory.",
            )
            return

        model = self.model_combo.currentText()

        if model == "No models available":
            QMessageBox.warning(
                self,
                "No Model Available",
                "Please download a model from the Models tab first.",
            )
            return

        # Check if model is available
        if not self.parent.model_manager.is_model_available(model):
            QMessageBox.warning(
                self,
                "Model Not Available",
                f"The selected model '{model}' is not available. Please download it from the Models tab.",
            )
            return

        force_reindex = self.force_checkbox.isChecked()
        lemmatization_enabled = self.lemmatization_checkbox.isChecked()
        stemming_enabled = self.stemming_checkbox.isChecked()

        # Get database path and convert to string
        db_path_str = str(
            get_db_path_for_directory(folder_path, self.parent.app_data_dir)
        )
        db_path = Path(db_path_str)

        # Check if directory is already indexed
        try:
            self.parent.context_manager.add_indexed_directory(
                directory_path=folder_path,
                db_path=db_path_str,
                tokenizer_model_name=model,
                lemmatization_enabled=lemmatization_enabled,
                stemming_enabled=stemming_enabled,
            )
            self.index_status_text.append(f"✓ Registered directory: {folder_path}")
            self.index_status_text.append(f"  Model: {model}")
            self.index_status_text.append(
                f"  Lemmatization: {'Enabled' if lemmatization_enabled else 'Disabled'}"
            )
            self.index_status_text.append(
                f"  Stemming: {'Enabled' if stemming_enabled else 'Disabled'}\n"
            )
        except DirectoryAlreadyIndexedError:
            existing_dir_info = self.parent.context_manager.get_directory_info(
                folder_path
            )

            if force_reindex:
                # Force re-indexing: wipe and rebuild with new settings
                self.index_status_text.append(
                    "🔄 Force re-indexing enabled - wiping existing index...\n"
                )

                # Remove the directory from context manager
                self.parent.context_manager.remove_indexed_directory(folder_path)

                # Delete the database file if it exists
                if db_path.exists():
                    try:
                        db_path.unlink()
                        self.index_status_text.append(
                            f"✓ Deleted existing database: {db_path.name}\n"
                        )
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Database Deletion Warning",
                            f"Could not delete existing database file:\n{e}",
                        )

                # Re-register the directory with new settings
                self.parent.context_manager.add_indexed_directory(
                    directory_path=folder_path,
                    db_path=db_path_str,
                    tokenizer_model_name=model,
                    lemmatization_enabled=lemmatization_enabled,
                    stemming_enabled=stemming_enabled,
                )
                self.index_status_text.append(
                    "✓ Re-registered directory with new settings"
                )
                self.index_status_text.append(f"  Model: {model}")
                self.index_status_text.append(
                    f"  Lemmatization: {'Enabled' if lemmatization_enabled else 'Disabled'}"
                )
                self.index_status_text.append(
                    f"  Stemming: {'Enabled' if stemming_enabled else 'Disabled'}\n"
                )
            else:
                # Normal re-indexing: ask user and use existing settings
                reply = QMessageBox.question(
                    self,
                    "Directory Already Indexed",
                    f"The directory '{folder_path}' is already indexed.\n\n"
                    "Do you want to re-index it with existing settings?\n\n"
                    f"Current settings:\n"
                    f"  Model: {existing_dir_info['tokenizer_model_name']}\n"
                    f"  Lemmatization: {'Enabled' if existing_dir_info.get('lemmatization_enabled', True) else 'Disabled'}\n"
                    f"  Stemming: {'Enabled' if existing_dir_info.get('stemming_enabled', False) else 'Disabled'}\n\n"
                    "💡 Tip: Check 'Force re-indexing' to rebuild with new settings.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.No:
                    return

                # Use existing settings for indexing
                self.index_status_text.append(
                    f"⚠ Directory already indexed: {folder_path}"
                )
                self.index_status_text.append(
                    "Updating index with existing settings..."
                )
                self.index_status_text.append(
                    f"  Model: {existing_dir_info['tokenizer_model_name']}"
                )
                self.index_status_text.append(
                    f"  Lemmatization: {'Enabled' if existing_dir_info.get('lemmatization_enabled', True) else 'Disabled'}"
                )
                self.index_status_text.append(
                    f"  Stemming: {'Enabled' if existing_dir_info.get('stemming_enabled', False) else 'Disabled'}"
                )
                self.index_status_text.append(
                    "💡 Tip: Use 'Force re-indexing' to rebuild with new settings\n"
                )

                # Override with existing settings
                model = existing_dir_info["tokenizer_model_name"]
                lemmatization_enabled = existing_dir_info.get(
                    "lemmatization_enabled", True
                )
                stemming_enabled = existing_dir_info.get("stemming_enabled", False)
                db_path_str = existing_dir_info["db_path"]

        # Create index and tokenizer
        try:
            index = SQLiteIndex(db_path=db_path_str)

            # Get model path from model manager
            model_path = None
            if self.parent.model_manager.is_model_in_downloads(model):
                model_path = str(self.parent.model_manager.models_dir / model)

            tokenizer = SpacyTokenizer(
                model=model,
                use_lemmatization=lemmatization_enabled,
                use_stemming=stemming_enabled,
                disable=["parser", "ner"],
                model_path=model_path,
            )

            doxearch = Doxearch(folder, index, tokenizer)

            # Disable UI during indexing
            self.index_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress

            # Start indexing in worker thread
            self.index_worker = IndexWorker(doxearch)
            self.index_worker.progress.connect(self.on_index_progress)
            self.index_worker.finished.connect(self.on_index_finished)
            self.index_worker.error.connect(self.on_index_error)
            self.index_worker.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Indexing Error", f"Failed to start indexing:\n{str(e)}"
            )

    def on_index_progress(self, message: str):
        """Handle indexing progress updates."""
        self.index_status_text.append(message)

    def on_index_finished(self, doc_count: int):
        """Handle indexing completion."""
        self.index_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.index_status_text.append(
            f"\n✓ Indexing completed successfully! Indexed {doc_count} documents."
        )

        # Set the newly indexed directory as active
        folder_path = self.folder_input.text().strip()
        self.parent.context_manager.set_active_directory(folder_path)

        # Refresh the main window
        self.parent.refresh_all_tabs()

        QMessageBox.information(
            self,
            "Indexing Complete",
            f"Successfully indexed {doc_count} documents.\n\n"
            f"The directory has been set as active.",
        )

    def on_index_error(self, error_message: str):
        """Handle indexing errors."""
        self.index_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.index_status_text.append(f"\n✗ Error: {error_message}")

        QMessageBox.critical(
            self,
            "Indexing Error",
            f"An error occurred during indexing:\n{error_message}",
        )
