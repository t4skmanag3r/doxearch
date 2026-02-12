from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from doxearch.doc_index.sqlite_index.sqlite_index import SQLiteIndex
from doxearch.doxearch import Doxearch, get_app_data_dir
from doxearch.tokenizer.spacy_tokenizer.spacy_tokenizer import SpacyTokenizer

app = typer.Typer(
    name="doxearch",
    help="Local search engine for indexing documents (PDFs, docxs, etc.) using TF-IDF.",
    add_completion=False,
)
console = Console()


def get_doxearch_instance(folder: Path, model: str = "en_core_web_sm") -> Doxearch:
    """Initialize and return a Doxearch instance."""
    app_data_dir = get_app_data_dir()
    db_path = app_data_dir / "doxearch.db"
    index = SQLiteIndex(db_path=str(db_path))

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


@app.command()
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
        doxearch = get_doxearch_instance(folder, model=model)
        console.print(f"[green]✓[/green] Database: {doxearch.index.engine.url}")
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


@app.command()
def search(
    folder: Path = typer.Argument(
        ...,
        help="Folder to search in",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    query: str = typer.Argument(None, help="Search query"),
    model: str = typer.Option(
        "en_core_web_sm",
        "--model",
        "-m",
        help="Spacy model to use for tokenization",
    ),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of results to show"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enter interactive search mode"
    ),
):
    """Search indexed documents."""
    try:
        doxearch = get_doxearch_instance(folder, model=model)

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


def _perform_search(doxearch: Doxearch, query: str, top_k: int):
    """Perform a single search and display results."""
    results = doxearch.search(query, top_k=top_k)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Create results table
    table = Table(title=f"Search Results for: '{query}'")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Filename", style="green")
    table.add_column("Path", style="blue")
    table.add_column("Score", style="magenta", width=10)

    for rank, result in enumerate(results, 1):
        filepath = Path(result["filepath"])

        # Get relative path from the indexed folder
        try:
            relative_path = filepath.relative_to(doxearch.folder_path)
            relative_dir = relative_path.parent
            display_path = str(relative_dir) if str(relative_dir) != "." else "."
        except ValueError:
            # If file is not relative to folder, show full path
            display_path = str(filepath.parent)

        # Create clickable link to parent directory
        parent_url = filepath.parent.as_uri()
        clickable_path = f"[link={parent_url}]{display_path}[/link]"

        # Create clickable filename that opens the file
        file_url = filepath.as_uri()
        clickable_filename = f"[link={file_url}]{result['filename']}[/link]"

        table.add_row(
            str(rank), clickable_filename, clickable_path, f"{result['score']:.4f}"
        )

    console.print(table)


def _interactive_search(doxearch: Doxearch):
    """Interactive search mode."""
    console.print("\n[bold cyan]Interactive Search Mode[/bold cyan]")
    console.print("Enter your queries (or 'quit' to exit)\n")

    while True:
        try:
            query = typer.prompt("Search", default="").strip()

            if query.lower() in ["quit", "exit", "q"]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            if not query:
                continue

            _perform_search(doxearch, query, top_k=10)
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Goodbye![/cyan]")
            break


if __name__ == "__main__":
    app()
