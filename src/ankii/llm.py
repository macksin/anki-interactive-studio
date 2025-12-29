"""OpenRouter LLM client for card evaluation."""

import os
import httpx
from typing import Any

from dotenv import load_dotenv


# Load .env file
load_dotenv()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "z-ai/glm-4.7"


class LLMError(Exception):
    """Raised when LLM API returns an error."""
    pass


class OpenRouterClient:
    """Client for the OpenRouter API."""
    
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key."
            )
        self.model = model
    
    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Make a chat completion request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ankii",
            "X-Title": "Anki Card Reviewer",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        
        try:
            response = httpx.post(
                OPENROUTER_URL, 
                json=payload, 
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"OpenRouter API error: {e}")
        
        result = response.json()
        
        if "error" in result:
            raise LLMError(f"OpenRouter error: {result['error']}")
        
        return result["choices"][0]["message"]["content"]
    
    def evaluate_card(self, front: str, back: str, card_type: str = "basic") -> dict[str, Any]:
        """Evaluate a flashcard's quality.
        
        Args:
            front: The front (question) of the card
            back: The back (answer) of the card
            card_type: Type of card (basic, cloze, etc.)
            
        Returns:
            Dict with evaluation results including:
            - overall_score: 1-10 rating
            - clarity: Assessment of question clarity
            - atomicity: Whether it tests one concept
            - issues: List of identified problems
            - suggestions: Suggested improvements
            - improved_front: Suggested improved front (optional)
            - improved_back: Suggested improved back (optional)
        """
        prompt = f"""You are an expert on spaced repetition and flashcard design. 
Evaluate this Anki flashcard based on best practices for effective learning.

**Card Type:** {card_type}

**Front (Question):**
{front}

**Back (Answer):**
{back}

Evaluate the card on these criteria:
1. **Clarity** - Is the question clear and unambiguous?
2. **Atomicity** - Does it test exactly one concept?
3. **Answer Quality** - Is the answer correct, concise, and sufficient?
4. **Formatting** - Is formatting used effectively?
5. **Learnability** - Will this be easy to remember long-term?

Respond in this exact JSON format:
{{
    "overall_score": <1-10>,
    "clarity": {{
        "score": <1-10>,
        "comment": "<brief assessment>"
    }},
    "atomicity": {{
        "score": <1-10>,
        "comment": "<brief assessment>"
    }},
    "answer_quality": {{
        "score": <1-10>,
        "comment": "<brief assessment>"
    }},
    "issues": ["<issue 1>", "<issue 2>"],
    "suggestions": ["<suggestion 1>", "<suggestion 2>"],
    "improved_front": "<suggested improved front, or null if good>",
    "improved_back": "<suggested improved back, or null if good>"
}}

Only output valid JSON, no other text."""

        response = self._chat([{"role": "user", "content": prompt}])
        
        # Parse JSON response
        import json
        try:
            # Handle potential markdown code blocks
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "overall_score": 0,
                "issues": ["Failed to parse LLM response"],
                "suggestions": [],
                "raw_response": response,
            }
    
    def analyze_card_group(
        self, 
        cards: list[dict[str, str]], 
        filter_applied: str = "",
        group_type: str = "cluster"
    ) -> str:
        """Analyze a group of related cards.
        
        Args:
            cards: List of dicts with 'front' and 'back' keys
            filter_applied: Description of filter used (e.g., "leech cards")
            group_type: "cluster" or "similar" to describe grouping method
            
        Returns:
            Analysis text from LLM
        """
        cards_text = "\n\n".join([
            f"**Card {i+1}:**\n- Front: {c['front']}\n- Back: {c['back']}"
            for i, c in enumerate(cards)
        ])
        
        context = ""
        if filter_applied:
            context = f"\n\nThese cards were filtered by: **{filter_applied}**"
        
        if group_type == "similar":
            grouping_desc = "These cards were grouped because they are semantically similar (high cosine similarity between their embeddings)."
        else:
            grouping_desc = "These cards were grouped together by clustering algorithm based on their semantic similarity."
        
        prompt = f"""Analyze this group of Anki flashcards and provide a brief summary.

{grouping_desc}{context}

**Cards in this group:**

{cards_text}

Please provide:
1. **Topic**: What subject/concept do these cards cover?
2. **Summary**: Brief description of what these cards are testing
3. **Observations**: Any patterns, redundancies, or issues you notice (e.g., near-duplicates, overlapping content)
4. **Recommendation**: Should any cards be merged, deleted, or revised?

Keep your response concise and actionable."""

        return self._chat([{"role": "user", "content": prompt}])
