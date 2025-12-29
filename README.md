# Anki Interactive Studio

A tool for visualizing and improving Anki flashcards using AI.

## Features

- **Card Clustering**: Visualize similar cards using AI embeddings
- **Interactive Scatter Plot**: Explore card groups in 2D space
- **CLI Tools**: List decks, review cards with LLM analysis

## Installation

1. Make sure [AnkiConnect](https://ankiweb.net/shared/info/2055492159) is installed in Anki
2. Clone this repo and install:

```bash
uv sync
```

3. Copy `.env.example` to `.env` and add your OpenRouter API key:
```bash
cp .env.example .env
```

## Usage

### Streamlit App (Card Clusters)

```bash
uv run streamlit run src/ankii/app.py
```

### CLI Commands

```bash
# List all decks
uv run ankii decks

# Review cards from a deck
uv run ankii review "Your Deck Name"

# Filter by card status
uv run ankii review "Your Deck Name" --filter "tag:leech"
```

## Requirements

- Anki with AnkiConnect plugin running
- OpenRouter API key
