"""Card evaluation logic and result formatting."""

import re
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class CardInfo:
    """Simplified card information."""
    card_id: int
    note_id: int
    deck_name: str
    model_name: str
    front: str
    back: str
    lapses: int
    reps: int
    fields: dict[str, str]
    
    @classmethod
    def from_anki_info(cls, info: dict[str, Any]) -> "CardInfo":
        """Create from AnkiConnect cardsInfo response."""
        # Extract text from HTML
        front = strip_html(info.get("question", ""))
        back = strip_html(info.get("answer", ""))
        
        # Clean up the back - it often contains the front
        if front and back.startswith(front):
            back = back[len(front):].strip()
        
        return cls(
            card_id=info["cardId"],
            note_id=info["note"],
            deck_name=info["deckName"],
            model_name=info["modelName"],
            front=front,
            back=back,
            lapses=info.get("lapses", 0),
            reps=info.get("reps", 0),
            fields={name: val["value"] for name, val in info.get("fields", {}).items()},
        )


def strip_html(html: str) -> str:
    """Remove HTML tags and clean up text."""
    # Remove style tags and their contents
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove script tags and their contents
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove CSS-like content that may have leaked (e.g., ".card { ... }")
    text = re.sub(r'\.[a-zA-Z][a-zA-Z0-9_-]*\s*\{[^}]*\}', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    return text.strip()


def display_card(card: CardInfo, console: Console) -> None:
    """Display a card nicely."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="bold cyan")
    table.add_column("Value")
    
    table.add_row("Deck", card.deck_name)
    table.add_row("Type", card.model_name)
    table.add_row("Lapses", str(card.lapses))
    table.add_row("Reviews", str(card.reps))
    
    console.print(table)
    console.print()
    console.print(Panel(card.front, title="Front", border_style="green"))
    console.print(Panel(card.back, title="Back", border_style="blue"))


def display_evaluation(evaluation: dict[str, Any], console: Console) -> None:
    """Display evaluation results."""
    score = evaluation.get("overall_score", "?")
    
    # Score color
    if isinstance(score, int):
        if score >= 8:
            score_style = "bold green"
        elif score >= 5:
            score_style = "bold yellow"
        else:
            score_style = "bold red"
    else:
        score_style = "bold white"
    
    console.print()
    console.print(f"[{score_style}]Overall Score: {score}/10[/{score_style}]")
    console.print()
    
    # Criteria scores
    for criterion in ["clarity", "atomicity", "answer_quality"]:
        if criterion in evaluation:
            data = evaluation[criterion]
            if isinstance(data, dict):
                c_score = data.get("score", "?")
                comment = data.get("comment", "")
                console.print(f"  {criterion.title()}: {c_score}/10 - {comment}")
    
    console.print()
    
    # Issues
    issues = evaluation.get("issues", [])
    if issues:
        console.print("[bold red]Issues:[/bold red]")
        for issue in issues:
            console.print(f"  • {issue}")
        console.print()
    
    # Suggestions
    suggestions = evaluation.get("suggestions", [])
    if suggestions:
        console.print("[bold yellow]Suggestions:[/bold yellow]")
        for suggestion in suggestions:
            console.print(f"  • {suggestion}")
        console.print()
    
    # Improved versions
    improved_front = evaluation.get("improved_front")
    if improved_front and improved_front != "null":
        console.print(Panel(str(improved_front), title="Suggested Front", border_style="green"))
    
    improved_back = evaluation.get("improved_back")
    if improved_back and improved_back != "null":
        console.print(Panel(str(improved_back), title="Suggested Back", border_style="blue"))
