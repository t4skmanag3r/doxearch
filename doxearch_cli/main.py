from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from doxearch.context_manager import DirectoryContextManager
from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex
from doxearch.doxearch import Doxearch
from doxearch.exceptions import DirectoryAlreadyIndexedError, DirectoryNotFoundError
from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer
from doxearch.utils.app_dir import get_app_data_dir
from doxearch.utils.general import get_db_path_for_directory

main = typer.Typer(
    name="doxearch",
    help="Local search engine for indexing documents (PDFs, docxs, etc.) using TF-IDF.",
    add_completion=False,
)
console = Console()


def get_context_manager() -> DirectoryContextManager:
    """Initialize and return a DirectoryContextManager instance."""
    app_data_dir = get_app_data_dir()
    context_db_path = app_data_dir / "context_manager.db"
    return DirectoryContextManager(db_path=str(context_db_path))


def get_doxearch_instance(
    folder: Path,
    db_path: str,
    model: str = "en_core_web_sm",
    use_lemmatization: bool = True,
    use_stemming: bool = False,
) -> Doxearch:
    """Create a Doxearch instance with the specified configuration."""
    index = SQLiteIndex(db_path=db_path)
    tokenizer = SpacyTokenizer(
        model=model,
        use_lemmatization=use_lemmatization,
        use_stemming=use_stemming,
        disable=["parser", "ner"],
    )
    return Doxearch(folder, index, tokenizer)


@main.command()
def index(
    folder: Path = typer.Argument(
        ...,
        help="Folder path to index",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    model: str = typer.Option(
        "en_core_web_sm",
        "--model",
        "-m",
        help="Spacy model to use for tokenization",
    ),
    batch_size: int = typer.Option(
        100, "--batch", "-b", help="Batch size for indexing"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-indexing of all documents"
    ),
    lemmatization: bool = typer.Option(
        True,
        "--lemmatization/--no-lemmatization",
        help="Enable or disable lemmatization",
    ),
    stemming: bool = typer.Option(
        False,
        "--stemming/--no-stemming",
        help="Enable or disable stemming",
    ),
):
    """Index documents in a folder."""
    try:
        # Validate tokenization options
        if lemmatization and stemming:
            console.print(
                "[bold red]✗ Error:[/bold red] Lemmatization and stemming cannot be enabled at the same time."
            )
            console.print("[cyan]Tip:[/cyan] Choose one or neither.")
            raise typer.Exit(1)

        context_manager = get_context_manager()
        folder_str = str(folder.resolve())

        # Prepare database path
        indexes_dir = context_manager.app_data_dir / "indexes"
        indexes_dir.mkdir(parents=True, exist_ok=True)
        db_path = get_db_path_for_directory(folder_str, context_manager.app_data_dir)

        # Register directory with tokenization settings
        try:
            context_manager.add_indexed_directory(
                folder_str,
                str(db_path),
                model,
                model_version=None,
                lemmatization_enabled=lemmatization,
                stemming_enabled=stemming,
            )
            console.print(f"[bold green]✓[/bold green] Registered directory: {folder}")
            console.print(f"  Model: {model}")
            console.print(
                f"  Lemmatization: {'Enabled' if lemmatization else 'Disabled'}"
            )
            console.print(f"  Stemming: {'Enabled' if stemming else 'Disabled'}")
        except DirectoryAlreadyIndexedError:
            if not force:
                console.print(f"[yellow]⚠[/yellow] Directory already indexed: {folder}")
                force_reindex = typer.confirm("Do you want to re-index?")
                if not force_reindex:
                    raise typer.Exit(0)

            console.print(f"[cyan]Re-indexing directory:[/cyan] {folder}")
            # Set as active to get the db_path
            active_info = context_manager.set_active_directory(folder_str)
            db_path = Path(active_info["db_path"])

        # Create Doxearch instance with tokenization settings
        doxearch = get_doxearch_instance(
            folder,
            str(db_path),
            model=model,
            use_lemmatization=lemmatization,
            use_stemming=stemming,
        )

        console.print(f"\n[bold cyan]Indexing folder:[/bold cyan] {folder}")
        console.print(f"[cyan]Batch size:[/cyan] {batch_size}\n")

        doxearch.index_folder(batch_size=batch_size)

        console.print("\n[bold green]✓ Indexing complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@main.command()
def search(
    query: str = typer.Argument(None, help="Search query"),
    folder: Path = typer.Argument(
        None,
        help="Folder to search in (optional if using active directory)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Spacy model to use for tokenization (uses active directory's model if not specified)",
    ),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results to show"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enter interactive search mode"
    ),
    no_fuzzy: bool = typer.Option(
        False,
        "--no-fuzzy",
        help="Disable fuzzy matching (exact matches only)",
    ),
):
    """Search indexed documents."""
    try:
        context_manager = get_context_manager()
        db_path = None

        # Determine folder, model, and db_path to use
        if folder is None:
            # Use active directory
            active_dir = context_manager.get_active_directory()
            if active_dir is None:
                console.print("[bold red]✗ Error:[/bold red] No active directory found")
                console.print(
                    "[cyan]Tip:[/cyan] Index a folder first or specify a folder path"
                )
                raise typer.Exit(1)

            folder = Path(active_dir["directory_path"])
            db_path = active_dir["db_path"]
            if model is None:
                model = active_dir["tokenizer_model_name"]

            console.print(f"[cyan]Using active directory:[/cyan] {folder}")
            console.print(f"[cyan]Using database:[/cyan] {db_path}")
            console.print(f"[cyan]Using model:[/cyan] {model}")
        else:
            # Folder specified, set it as active
            folder_str = str(folder.resolve())
            try:
                active_info = context_manager.set_active_directory(folder_str)
                db_path = active_info["db_path"]

                # Use model from active directory if not specified
                if model is None:
                    model = active_info["tokenizer_model_name"]
                    console.print(f"[cyan]Using model from directory:[/cyan] {model}")

                console.print(f"[cyan]Using database:[/cyan] {db_path}")
            except DirectoryNotFoundError:
                console.print(f"[yellow]⚠[/yellow] Directory not indexed: {folder}")
                console.print(
                    "[cyan]Tip:[/cyan] Index the folder first with 'doxearch index'"
                )
                raise typer.Exit(1)

        # Default model if still None
        if model is None:
            model = "en_core_web_sm"
            console.print(f"[cyan]Using default model:[/cyan] {model}")

        # Show fuzzy matching status
        use_fuzzy = not no_fuzzy
        fuzzy_status = "enabled" if use_fuzzy else "disabled"
        console.print(f"[cyan]Fuzzy matching:[/cyan] {fuzzy_status}")

        doxearch = get_doxearch_instance(folder, db_path, model=model)

        if interactive:
            _interactive_search(doxearch, use_fuzzy=use_fuzzy)
            return

        # Validate query is provided for non-interactive mode
        if not query:
            console.print(
                "[bold red]✗ Error:[/bold red] Query is required for non-interactive search"
            )
            console.print(
                "[cyan]Tip:[/cyan] Use --interactive flag for interactive mode"
            )
            raise typer.Exit(1)

        _perform_search(doxearch, query, top_k, use_fuzzy=use_fuzzy)

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@main.command()
def list_directories():
    """List all indexed directories."""
    try:
        context_manager = get_context_manager()

        with context_manager.get_session() as session:
            from doxearch.context_manager import IndexedDirectory

            directories = session.query(IndexedDirectory).all()

            if not directories:
                console.print("[yellow]No indexed directories found[/yellow]")
                console.print(
                    "[cyan]Tip:[/cyan] Index a folder with 'doxearch index <folder>'"
                )
                return

            # Create table
            table = Table(title="Indexed Directories")
            table.add_column("Status", style="cyan", width=8)
            table.add_column("Directory", style="green")
            table.add_column("Model", style="blue")
            table.add_column("Version", style="magenta")
            table.add_column("Lemma", style="yellow", width=6)
            table.add_column("Stem", style="yellow", width=6)
            table.add_column("Database", style="dim")

            for directory in directories:
                status = "✓ Active" if directory.is_active else "Inactive"
                status_style = "green" if directory.is_active else "dim"

                # Show relative database path for readability
                db_path = Path(directory.db_path)
                db_display = db_path.name if db_path.exists() else directory.db_path

                # Format boolean fields
                lemma_display = "✓" if directory.lemmatization_enabled else "✗"
                stem_display = "✓" if directory.stemming_enabled else "✗"

                table.add_row(
                    f"[{status_style}]{status}[/{status_style}]",
                    directory.directory_path,
                    directory.tokenizer_model_name,
                    directory.tokenizer_model_version or "N/A",
                    lemma_display,
                    stem_display,
                    db_display,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@main.command()
def set_active(
    folder: Path = typer.Argument(
        ...,
        help="Folder to set as active",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
):
    """Set a directory as the active directory for searching."""
    try:
        context_manager = get_context_manager()
        folder_str = str(folder.resolve())

        active_info = context_manager.set_active_directory(folder_str)

        console.print(f"[green]✓[/green] Set active directory: {folder_str}")
        console.print(f"[cyan]Model:[/cyan] {active_info['tokenizer_model_name']}")
        console.print(f"[cyan]Database:[/cyan] {active_info['db_path']}")

    except DirectoryNotFoundError:
        console.print(f"[bold red]✗ Error:[/bold red] Directory not indexed: {folder}")
        console.print("[cyan]Tip:[/cyan] Index the folder first with 'doxearch index'")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


def _perform_search(doxearch: Doxearch, query: str, top_k: int, use_fuzzy: bool = True):
    """Perform a single search and display results."""
    console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")

    results = doxearch.search(query, top_k=top_k, use_fuzzy=use_fuzzy)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(results)} result(s):[/bold green]\n")

    for i, result in enumerate(results, 1):
        # Handle both dict and object results
        if isinstance(result, dict):
            filename = result.get("filename", "Unknown")
            score = result.get("score", 0.0)
            filepath = result.get("filepath", "Unknown")
        else:
            filename = result.filename
            score = result.score
            filepath = result.filepath

        console.print(f"[bold]{i}. {filename}[/bold]")
        console.print(f"   Score: {score:.4f}")
        console.print(f"   Path: {filepath}")
        console.print()


def _interactive_search(doxearch: Doxearch, use_fuzzy: bool = True):
    """Run interactive search mode."""
    console.print("[bold cyan]Interactive Search Mode[/bold cyan]")
    fuzzy_status = "enabled" if use_fuzzy else "disabled"
    console.print(f"[cyan]Fuzzy matching: {fuzzy_status}[/cyan]")
    console.print("Type your query and press Enter. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            query = typer.prompt("Search", default="")

            if query.lower() in ["exit", "quit", ""]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            _perform_search(doxearch, query, top_k=10, use_fuzzy=use_fuzzy)

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
