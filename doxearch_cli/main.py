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
    folder: Path, db_path: str, model: str = "en_core_web_sm"
) -> Doxearch:
    """Initialize and return a Doxearch instance."""
    index = SQLiteIndex(db_path=db_path)

    try:
        tokenizer = SpacyTokenizer(model=model, disable=["parser", "ner"])
    except ValueError as e:
        console.print(f"[bold red]✗ Model Error:[/bold red] {e}")
        console.print("\n[yellow]Available models:[/yellow]")
        console.print("  • en_core_web_sm (English)")
        console.print("  • lt_core_news_sm (Lithuanian)")
        console.print("\n[cyan]Install with:[/cyan]")
        console.print(f"  uv run python -m spacy download {model}")
        raise typer.Exit(1)

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
):
    """Index documents in a folder."""
    console.print(f"[bold blue]Indexing folder:[/bold blue] {folder}")

    try:
        # Initialize context manager
        context_manager = get_context_manager()
        app_data_dir = get_app_data_dir()

        # Ensure indexes directory exists
        indexes_dir = app_data_dir / "indexes"
        indexes_dir.mkdir(parents=True, exist_ok=True)

        # Check if directory is already indexed
        folder_str = str(folder.resolve())

        # Get or create database path for this directory
        db_path = get_db_path_for_directory(folder_str, app_data_dir)

        try:
            # Try to add the directory to context manager
            context_manager.add_indexed_directory(
                folder_str,
                str(db_path),
                model,
                model_version=None,  # TODO: Extract version from tokenizer
            )
            console.print(f"[green]✓[/green] Registered directory in context manager")
            console.print(f"[green]✓[/green] Database: {db_path}")
        except DirectoryAlreadyIndexedError:
            if not force:
                console.print(f"[yellow]⚠[/yellow] Directory already indexed")
                console.print("[cyan]Tip:[/cyan] Use --force to re-index")

                # Set as active directory and get its db_path
                active_info = context_manager.set_active_directory(folder_str)
                db_path = Path(active_info["db_path"])
                console.print(f"[green]✓[/green] Set as active directory")
                console.print(f"[green]✓[/green] Using database: {db_path}")
            else:
                console.print(f"[yellow]⚠[/yellow] Force re-indexing...")
                # Set as active directory for re-indexing
                active_info = context_manager.set_active_directory(folder_str)
                db_path = Path(active_info["db_path"])

        doxearch = get_doxearch_instance(folder, str(db_path), model=model)
        console.print(f"[green]✓[/green] Using model: {model}")

        # Check for documents
        pdf_files = list(folder.rglob("*.pdf"))
        docx_files = list(folder.rglob("*.docx"))
        total_files = len(pdf_files) + len(docx_files)

        if total_files == 0:
            console.print("[yellow]⚠[/yellow] No PDF or DOCX files found")
            raise typer.Exit(1)

        console.print(f"[cyan]Found {total_files} document(s)[/cyan]")

        # Index with progress
        with console.status("[bold green]Indexing documents..."):
            doxearch.index_folder()

        console.print("[bold green]✓ Indexing completed![/bold green]")

        # Show statistics
        final_count = doxearch.index.get_document_count()
        console.print(f"[cyan]Total documents indexed: {final_count}[/cyan]")

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

        doxearch = get_doxearch_instance(folder, db_path, model=model)

        if interactive:
            _interactive_search(doxearch)
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

        _perform_search(doxearch, query, top_k)

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
            table.add_column("Database", style="yellow")

            for directory in directories:
                status = "✓ Active" if directory.is_active else "Inactive"
                status_style = "green" if directory.is_active else "dim"

                # Show relative database path for readability
                db_path = Path(directory.db_path)
                db_display = db_path.name if db_path.exists() else directory.db_path

                table.add_row(
                    f"[{status_style}]{status}[/{status_style}]",
                    directory.directory_path,
                    directory.tokenizer_model_name,
                    directory.tokenizer_model_version or "N/A",
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


@main.command()
def remove_directory(
    folder: Path = typer.Argument(
        ...,
        help="Folder to remove from index",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    delete_db: bool = typer.Option(
        False,
        "--delete-db",
        "-d",
        help="Also delete the database file",
    ),
):
    """Remove a directory from the index."""
    try:
        context_manager = get_context_manager()
        folder_str = str(folder.resolve())

        # Get directory info before removing
        dir_info = context_manager.get_directory_info(folder_str)
        if not dir_info:
            console.print(f"[yellow]⚠[/yellow] Directory not indexed: {folder}")
            raise typer.Exit(1)

        db_path = Path(dir_info["db_path"])

        # Remove from context manager
        context_manager.remove_indexed_directory(folder_str)
        console.print(f"[green]✓[/green] Removed directory from index: {folder_str}")

        # Optionally delete the database file
        if delete_db and db_path.exists():
            db_path.unlink()
            console.print(f"[green]✓[/green] Deleted database file: {db_path}")
        elif db_path.exists():
            console.print(f"[cyan]Database file preserved:[/cyan] {db_path}")
            console.print(
                "[cyan]Tip:[/cyan] Use --delete-db flag to also delete the database"
            )

    except DirectoryNotFoundError:
        console.print(f"[bold red]✗ Error:[/bold red] Directory not indexed: {folder}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


def _perform_search(doxearch: Doxearch, query: str, top_k: int):
    """Perform a single search and display results."""
    console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")

    results = doxearch.search(query, top_k=top_k)

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


def _interactive_search(doxearch: Doxearch):
    """Run interactive search mode."""
    console.print("[bold cyan]Interactive Search Mode[/bold cyan]")
    console.print("Type your query and press Enter. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            query = typer.prompt("Search", default="")

            if query.lower() in ["exit", "quit", ""]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            _perform_search(doxearch, query, top_k=10)

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
