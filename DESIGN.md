# Project Crease — Design Document

## What is Project Crease?

A cricket match analysis agent that takes a raw scorecard as input and returns structured insights and styled commentary. Built with LangChain and LangGraph.

---

## Core Concepts

### 1. What is a StateGraph?

A `StateGraph` is LangGraph's core abstraction for building agentic workflows. It models a pipeline as a directed graph where **nodes** are functions that read and update a shared state dict, and **edges** define the execution order between them. In Project Crease, `build_graph()` creates a `StateGraph(MatchState)` with four nodes (`extract → confidence → commentary → report`). The graph is compiled into a runnable app, and each node receives the current `MatchState` and returns a partial dict to merge back into it.

### 2. What is a Conditional Edge?

A conditional edge is an edge whose destination is decided **at runtime** by a routing function, rather than being hardcoded. In Project Crease, `add_conditional_edges("confidence", route_confidence)` uses the `route_confidence` function to inspect `state["confidence"]` — if it is `"low"`, execution loops back to the `extract` node for another attempt; if `"high"`, it moves forward to `commentary`. This is what enables the retry loop in the pipeline.

### 3. What does `with_structured_output` do?

`with_structured_output` wraps an LLM so that its response is automatically parsed into a given schema — in this case, the `CricketAnalysis` Pydantic model. Instead of receiving a raw text string, the caller gets back a validated Python object with typed fields like `match_result`, `top_performer`, etc. Under the hood it instructs the model to return JSON conforming to the schema and handles deserialization and validation.

---

## Design Decisions

### 4. Why LangGraph instead of a simple LangChain chain?

A plain LangChain chain (`prompt | llm | parser`) is strictly linear — there is no way to loop back or branch. Project Crease needs a **conditional retry loop**: if the extraction quality is low, the pipeline must circle back to the extract node and try again. `StateGraph` supports conditional edges and cycles, making this possible. It also cleanly separates each step into its own node, making the pipeline easier to debug, extend, and visualize.

### 5. Why does the confidence check loop exist?

LLM outputs are non-deterministic — a single call can produce vague or incomplete extractions (e.g., a one-line turning point or an empty top performer). The `confidence_node` acts as a **quality gate**: it inspects the extracted fields against minimum thresholds (character length, non-empty checks) and routes back to `extract_node` if the output is not good enough. This self-healing loop gives the model additional chances to produce better results without any human intervention, capped at 2 retries to avoid runaway costs.

### 6. Why Pydantic for structured output?

Pydantic provides **schema definition, type validation, and serialization** in one package. By defining `CricketAnalysis` as a `BaseModel`, the project gets: (1) a clear contract of exactly which fields the LLM must return, (2) automatic validation that all fields are present and correctly typed, and (3) easy conversion to a plain dict via `model_dump()` for storing in `MatchState`. It also plugs directly into LangChain's `with_structured_output`, which uses the Pydantic schema to constrain the LLM's JSON output.

---

## Pipeline Architecture

```
[Extract] → [Confidence Check] → [Commentary] → [Report]
                ↑_______low confidence_________|
```

### Nodes

| Node | Responsibility |
|------|---------------|
| `extract_node` | Calls LLM with raw scorecard, returns structured CricketAnalysis object |
| `confidence_node` | Validates extraction quality, routes back to extract if vague |
| `commentary_node` | Generates styled commentary based on analysis and chosen style |
| `report_node` | Prints final match report with styled commentary |

### State

```python
class MatchState(TypedDict):
    scorecard: str           # Raw input from user
    analysis: dict           # Extracted structured data
    confidence: str          # "high" or "low"
    retry_count: int         # Max 2 retries to prevent infinite loop
    style: str               # "neutral" | "dramatic" | "yoda"
    styled_commentary: str   # Final commentary output
```

---

## Key Design Choices

### Why max 2 retries?

The `confidence_node` loops back to `extract_node` on low confidence, creating a cycle in the StateGraph. Without a cap, a consistently vague scorecard (or a model having a bad day) would cause an **infinite loop** — burning API credits and never reaching the commentary/report nodes. Two retries strikes a balance: the LLM gets **3 total attempts** (1 initial + 2 retries), which is enough for stochastic variance to produce a better extraction, but not so many that you waste time and money on a fundamentally bad input. After 2 retries, the pipeline accepts whatever it has and moves on — a "best effort" graceful degradation.

### Why flat Pydantic fields instead of nested lists?

`CricketAnalysis` uses flat `str` fields (`top_bowler_1`, `top_bowler_2`, `top_bowler_3`) instead of something like `top_bowlers: list[Bowler]`. This is because **`with_structured_output` on smaller models (like Llama 3.1 8B) is far more reliable with flat, scalar fields**. Nested objects and lists increase the chance the model produces malformed JSON — missing brackets, wrong nesting, or incorrect array lengths. Flat fields give the LLM a simpler schema to conform to, improving extraction success rates and reducing retries. It is a pragmatic trade-off: less elegant schema, but significantly more robust structured output from a compact model.

---

## What I would add in v2

- Streamlit UI for scorecard input and style selection
- Cricinfo API integration to fetch live scorecards
- Support for T20, ODI, and Test match type detection
- RAG layer for player career stats comparison