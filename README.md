# Project Crease 🏏

A cricket match analysis agent built with LangChain and LangGraph.

## What it does

Paste any cricket scorecard and the agent:
1. Extracts structured match data (result, top performer, key moment, bowlers, collapse)
2. Validates extraction quality with a confidence check loop
3. Generates styled commentary — Neutral, Dramatic, or Yoda

## Architecture

```
[Extract] → [Confidence Check] → [Commentary] → [Report]
                ↑_______low confidence_________|
```

Built with LangGraph StateGraph — the confidence check node loops back to extraction if quality is low, max 2 retries.

## Tech Stack

- Python 3.12
- LangChain
- LangGraph
- Groq (llama-3.1-8b-instant)
- Pydantic

## Setup

```bash
git clone https://github.com/someshktiwari/project-crease.git
cd project-crease
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

## Usage

```python
from src.pipeline import analyse

result = analyse(scorecard=your_scorecard_text, style="dramatic")
```

## Styles

- `neutral` — factual and calm
- `dramatic` — emotional and intense
- `yoda` — speak like Yoda, you must

## Author

Somesh Kant Tiwari  
[GitHub](https://github.com/someshktiwari) · [LinkedIn](https://linkedin.com/in/someshkanttiwari)