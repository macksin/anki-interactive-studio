"""OpenRouter LLM client for card evaluation."""

import os
import httpx
from typing import Any
from dataclasses import dataclass, field

from dotenv import load_dotenv


# Load .env file
load_dotenv()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

# Pricing per 1M tokens (input, output) - update as needed
# Source: https://openrouter.ai/models
MODEL_PRICING = {
    # Google Gemini models
    "google/gemini-2.0-flash-001": (0.1, 0.4),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-3-flash-preview": (0.50, 3.0),
    # Z.AI GLM models (formerly ZhipuAI)
    "z-ai/glm-4.7": (0.40, 1.50),
    "z-ai/glm-4.6": (0.20, 0.80),
    # OpenAI models
    "openai/gpt-4o-mini": (0.15, 0.6),
    "openai/gpt-4o": (2.5, 10.0),
    # Anthropic models
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "anthropic/claude-3.5-sonnet": (3.0, 15.0),
    "anthropic/claude-3-opus": (15.0, 75.0),
    # Open source models
    "meta-llama/llama-3.3-70b-instruct": (0.3, 0.3),
    "mistralai/mistral-7b-instruct": (0.06, 0.06),
    "deepseek/deepseek-chat-v3": (0.14, 0.28),
}

# Available models for UI selection (name -> model_id)
AVAILABLE_MODELS = {
    "Gemini 3 Flash Preview (newest)": "google/gemini-3-flash-preview",
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Gemini 2.0 Flash (fast, cheap)": "google/gemini-2.0-flash-001",
    "GLM-4.7 (Z.AI flagship)": "z-ai/glm-4.7",
    "GLM-4.6": "z-ai/glm-4.6",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "GPT-4o": "openai/gpt-4o",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "Llama 3.3 70B": "meta-llama/llama-3.3-70b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat-v3",
}

# Default pricing if model not in list
DEFAULT_PRICING = (1.0, 2.0)  # $1/M input, $2/M output


@dataclass
class UsageStats:
    """Track API usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    request_count: int = 0

    def add_usage(self, prompt: int, completion: int, cost: float) -> None:
        """Add usage from a request."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.total_cost += cost
        self.request_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "request_count": self.request_count,
        }

    def format_cost(self) -> str:
        """Format cost as string."""
        if self.total_cost < 0.01:
            return f"${self.total_cost:.4f}"
        return f"${self.total_cost:.2f}"


class LLMError(Exception):
    """Raised when LLM API returns an error."""
    pass


# Global usage tracker for session
_session_usage = UsageStats()


def get_session_usage() -> UsageStats:
    """Get the global session usage stats."""
    return _session_usage


def reset_session_usage() -> None:
    """Reset the global session usage stats."""
    global _session_usage
    _session_usage = UsageStats()


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
        self.last_usage: dict[str, Any] | None = None

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage."""
        input_price, output_price = MODEL_PRICING.get(self.model, DEFAULT_PRICING)
        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (completion_tokens / 1_000_000) * output_price
        return input_cost + output_cost

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

        # Track usage
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = self._calculate_cost(prompt_tokens, completion_tokens)

        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            "model": self.model,
        }

        # Add to global session tracker
        _session_usage.add_usage(prompt_tokens, completion_tokens, cost)

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

    def evaluate_and_merge_cards(
        self,
        cards: list[dict[str, str]],
        context: str = "",
    ) -> dict[str, Any]:
        """Multi-agent evaluation: analyze cards and propose actions.

        This uses a chain-of-thought approach to:
        1. Evaluate each card for quality and accuracy
        2. Identify duplicates and overlapping content
        3. Propose merging, deletion, or improvement actions
        4. Generate new optimized cards

        Args:
            cards: List of dicts with 'front', 'back', and optionally 'card_id', 'note_id'
            context: Additional context about the cards (deck name, filter, etc.)

        Returns:
            Dict with:
            - analysis: Overall analysis text
            - cards_to_delete: List of indices of cards to mark for deletion
            - cards_to_keep: List of indices of cards to keep as-is
            - new_cards: List of new card dicts with 'front' and 'back'
            - reasoning: Explanation for each decision
        """
        import json

        cards_text = "\n\n".join([
            f"**Card {i+1}** (ID: {c.get('card_id', 'N/A')}):\n- Front: {c['front']}\n- Back: {c['back']}"
            for i, c in enumerate(cards)
        ])

        prompt = f"""You are an expert flashcard optimizer using spaced repetition best practices.

## Your Task
Analyze the following group of similar Anki flashcards and determine the optimal action for each.

## Context
{context if context else "These cards were grouped by semantic similarity."}

## Cards to Analyze

{cards_text}

## Evaluation Criteria

For each card, evaluate:
1. **Factual Accuracy**: Is the information correct?
2. **Atomicity**: Does it test exactly ONE concept? (Cards should be atomic)
3. **Clarity**: Is the question unambiguous?
4. **Redundancy**: Does it duplicate another card's content?
5. **Effectiveness**: Will this help long-term retention?

## Best Practices for Flashcards
- One concept per card (atomic)
- Clear, unambiguous questions
- Concise but complete answers
- No trick questions
- Test understanding, not memorization of exact wording

## Actions You Can Take

For each card, choose ONE action:
- **KEEP**: Card is good as-is
- **DELETE**: Card is redundant, incorrect, or low quality
- **MERGE**: Combine with other cards into a better version

If cards should be MERGED or improved, create NEW cards that:
- Follow all best practices
- Preserve the important information
- Are better than the originals

## Response Format

Respond with ONLY valid JSON in this exact format:
```json
{{
    "analysis": "<2-3 sentence summary of what you found>",
    "card_decisions": [
        {{
            "card_index": 0,
            "action": "DELETE|KEEP|MERGE",
            "reason": "<brief explanation>"
        }}
    ],
    "new_cards": [
        {{
            "front": "<question>",
            "back": "<answer>",
            "source_cards": [0, 1],
            "reason": "<why this new card is better>"
        }}
    ],
    "summary": {{
        "total_reviewed": <number>,
        "to_delete": <number>,
        "to_keep": <number>,
        "new_cards_created": <number>
    }}
}}
```

Important:
- card_index is 0-based (first card is 0)
- source_cards lists which original cards this new card replaces/combines
- If no new cards needed, use empty array: "new_cards": []
- Be conservative: only DELETE truly bad/redundant cards
- Be helpful: create better cards when merging is beneficial"""

        response = self._chat([{"role": "user", "content": prompt}])

        # Parse JSON response
        try:
            text = response.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())

            # Validate and normalize response
            if "card_decisions" not in result:
                result["card_decisions"] = []
            if "new_cards" not in result:
                result["new_cards"] = []
            if "analysis" not in result:
                result["analysis"] = "Analysis completed."
            if "summary" not in result:
                result["summary"] = {
                    "total_reviewed": len(cards),
                    "to_delete": len([d for d in result["card_decisions"] if d.get("action") == "DELETE"]),
                    "to_keep": len([d for d in result["card_decisions"] if d.get("action") == "KEEP"]),
                    "new_cards_created": len(result["new_cards"]),
                }

            return result

        except json.JSONDecodeError as e:
            return {
                "analysis": "Failed to parse LLM response. Please try again.",
                "card_decisions": [],
                "new_cards": [],
                "summary": {"total_reviewed": len(cards), "to_delete": 0, "to_keep": len(cards), "new_cards_created": 0},
                "error": str(e),
                "raw_response": response,
            }
