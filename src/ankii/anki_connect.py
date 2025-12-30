"""AnkiConnect API client.

Communicates with Anki via the AnkiConnect plugin.
AnkiConnect must be installed and Anki must be running.
API documentation: https://foosoft.net/projects/anki-connect/
"""

import httpx
from typing import Any


ANKI_CONNECT_URL = "http://localhost:8765"


class AnkiConnectError(Exception):
    """Raised when AnkiConnect returns an error."""
    pass


class AnkiConnect:
    """Client for the AnkiConnect API."""
    
    def __init__(self, url: str = ANKI_CONNECT_URL):
        self.url = url
        self.version = 6  # AnkiConnect API version
    
    def _invoke(self, action: str, **params: Any) -> Any:
        """Make a request to AnkiConnect."""
        payload = {
            "action": action,
            "version": self.version,
            "params": params,
        }
        
        try:
            response = httpx.post(self.url, json=payload, timeout=10.0)
            response.raise_for_status()
        except httpx.ConnectError:
            raise AnkiConnectError(
                "Could not connect to Anki. Make sure Anki is running "
                "and AnkiConnect is installed."
            )
        except httpx.HTTPError as e:
            raise AnkiConnectError(f"HTTP error: {e}")
        
        result = response.json()
        
        if result.get("error"):
            raise AnkiConnectError(result["error"])
        
        return result.get("result")
    
    def get_decks(self) -> list[str]:
        """Get list of all deck names."""
        return self._invoke("deckNames")
    
    def find_cards(self, query: str) -> list[int]:
        """Find cards matching a query.
        
        Query uses Anki's search syntax:
        - "deck:DeckName" - cards in a specific deck
        - "is:due" - cards that are due
        - "is:new" - new cards
        - "is:learn" - cards in learning
        - "is:review" - review cards
        - "is:suspended" - suspended cards
        - "prop:lapses>3" - cards with more than 3 lapses
        
        Returns list of card IDs.
        """
        return self._invoke("findCards", query=query)
    
    def get_cards_info(self, card_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed info for cards.
        
        Returns info including:
        - cardId: Card ID
        - noteId: Note ID
        - deckName: Deck name
        - modelName: Note type name
        - fields: Dict of field names to values
        - question: Rendered question HTML
        - answer: Rendered answer HTML
        - lapses: Number of times card was forgotten
        - reps: Total number of reviews
        """
        return self._invoke("cardsInfo", cards=card_ids)
    
    def get_notes_info(self, note_ids: list[int]) -> list[dict[str, Any]]:
        """Get detailed info for notes."""
        return self._invoke("notesInfo", notes=note_ids)
    
    def update_note_fields(self, note_id: int, fields: dict[str, str]) -> None:
        """Update the fields of a note.
        
        Args:
            note_id: The note ID to update
            fields: Dict mapping field names to new values
        """
        self._invoke("updateNoteFields", note={
            "id": note_id,
            "fields": fields,
        })
    
    def get_deck_stats(self, deck_name: str) -> dict[str, Any]:
        """Get statistics for a deck."""
        decks = self._invoke("getDeckStats", decks=[deck_name])
        if decks:
            return list(decks.values())[0]
        return {}

    def add_tags(self, note_ids: list[int], tags: str) -> None:
        """Add tags to notes.

        Args:
            note_ids: List of note IDs to tag
            tags: Space-separated tags to add
        """
        self._invoke("addTags", notes=note_ids, tags=tags)

    def remove_tags(self, note_ids: list[int], tags: str) -> None:
        """Remove tags from notes.

        Args:
            note_ids: List of note IDs to untag
            tags: Space-separated tags to remove
        """
        self._invoke("removeTags", notes=note_ids, tags=tags)

    def suspend_cards(self, card_ids: list[int]) -> None:
        """Suspend cards."""
        self._invoke("suspend", cards=card_ids)

    def unsuspend_cards(self, card_ids: list[int]) -> None:
        """Unsuspend cards."""
        self._invoke("unsuspend", cards=card_ids)

    def delete_notes(self, note_ids: list[int]) -> None:
        """Delete notes (and all their cards).

        WARNING: This is irreversible!

        Args:
            note_ids: List of note IDs to delete
        """
        self._invoke("deleteNotes", notes=note_ids)
