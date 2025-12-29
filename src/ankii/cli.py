"""Anki Card Reviewer CLI."""

import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt

from ankii.anki_connect import AnkiConnect, AnkiConnectError
from ankii.llm import OpenRouterClient, LLMError
from ankii.evaluator import CardInfo, display_card, display_evaluation


app = typer.Typer(
    name="ankii",
    help="Anki card reviewer with LLM-powered analysis",
    no_args_is_help=True,
)
console = Console()


@app.command()
def decks():
    """List all available decks."""
    try:
        anki = AnkiConnect()
        deck_names = anki.get_decks()
    except AnkiConnectError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    console.print()
    console.print("[bold]Available Decks:[/bold]")
    console.print()
    
    for name in sorted(deck_names):
        # Skip default deck if empty description
        if name == "Default" and len(deck_names) > 1:
            continue
        console.print(f"  • {name}")
    
    console.print()
    console.print(f"[dim]Total: {len(deck_names)} decks[/dim]")


@app.command()
def review(
    deck: str = typer.Argument(..., help="Deck name to review"),
    filter: str = typer.Option("", "--filter", "-f", help="Additional Anki search filter"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum cards to review"),
    model: str = typer.Option("anthropic/claude-3.5-sonnet", "--model", "-m", help="LLM model to use"),
):
    """Review and evaluate cards from a deck."""
    try:
        anki = AnkiConnect()
    except AnkiConnectError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    try:
        llm = OpenRouterClient(model=model)
    except LLMError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    # Build query
    query = f'deck:"{deck}"'
    if filter:
        query = f"{query} {filter}"
    
    console.print(f"\n[dim]Searching: {query}[/dim]\n")
    
    try:
        card_ids = anki.find_cards(query)
    except AnkiConnectError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    if not card_ids:
        console.print("[yellow]No cards found matching the query.[/yellow]")
        raise typer.Exit(0)
    
    console.print(f"Found [bold]{len(card_ids)}[/bold] cards")
    
    # Limit cards
    if len(card_ids) > limit:
        console.print(f"[dim]Reviewing first {limit} cards (use --limit to change)[/dim]")
        card_ids = card_ids[:limit]
    
    console.print()
    
    # Get card info
    try:
        cards_info = anki.get_cards_info(card_ids)
    except AnkiConnectError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    # Review each card
    for i, info in enumerate(cards_info, 1):
        card = CardInfo.from_anki_info(info)
        
        console.rule(f"[bold]Card {i}/{len(cards_info)}[/bold]")
        console.print()
        
        display_card(card, console)
        
        # Ask if user wants to evaluate
        if not Confirm.ask("\nEvaluate this card?", default=True):
            continue
        
        console.print("[dim]Analyzing with LLM...[/dim]")
        
        try:
            evaluation = llm.evaluate_card(card.front, card.back, card.model_name)
            display_evaluation(evaluation, console)
        except LLMError as e:
            console.print(f"[bold red]LLM Error:[/bold red] {e}")
            continue
        
        # Ask about applying changes
        improved_front = evaluation.get("improved_front")
        improved_back = evaluation.get("improved_back")
        
        if improved_front or improved_back:
            if Confirm.ask("\nApply suggested improvements?", default=False):
                # Find the field names
                field_names = list(card.fields.keys())
                if len(field_names) >= 2:
                    front_field = field_names[0]
                    back_field = field_names[1]
                    
                    updates = {}
                    if improved_front and improved_front != "null":
                        updates[front_field] = str(improved_front)
                    if improved_back and improved_back != "null":
                        updates[back_field] = str(improved_back)
                    
                    if updates:
                        try:
                            anki.update_note_fields(card.note_id, updates)
                            console.print("[bold green]✓ Card updated![/bold green]")
                        except AnkiConnectError as e:
                            console.print(f"[bold red]Failed to update:[/bold red] {e}")
        
        console.print()
        
        # Continue?
        if i < len(cards_info):
            if not Confirm.ask("Continue to next card?", default=True):
                break
    
    console.print("\n[bold green]Review complete![/bold green]\n")


@app.command()
def stats(deck: str = typer.Argument(..., help="Deck name")):
    """Show statistics for a deck."""
    try:
        anki = AnkiConnect()
        stats = anki.get_deck_stats(deck)
    except AnkiConnectError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    
    if not stats:
        console.print(f"[yellow]Deck '{deck}' not found.[/yellow]")
        raise typer.Exit(1)
    
    table = Table(title=f"Stats: {deck}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    
    table.add_row("New cards", str(stats.get("new_count", 0)))
    table.add_row("Learning", str(stats.get("learn_count", 0)))
    table.add_row("Review", str(stats.get("review_count", 0)))
    table.add_row("Total in deck", str(stats.get("total_in_deck", 0)))
    
    console.print()
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
