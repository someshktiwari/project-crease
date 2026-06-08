"""
Project Crease - Cricket Match Analysis Agent
LangChain + LangGraph pipeline for structured match analysis and styled commentary.
"""

import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


# ── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
)


# ── Pydantic schema ───────────────────────────────────────────────────────────

class CricketAnalysis(BaseModel):
    match_result: str = Field(description="Who won and by how much")
    top_performer: str = Field(description="Player of the match and why")
    key_turning_point: str = Field(description="The single moment that decided the match")
    top_bowler_1: str = Field(description="Best bowler: name, wickets, runs conceded")
    top_bowler_2: str = Field(description="Second best bowler: name, wickets, runs conceded")
    top_bowler_3: str = Field(description="Third best bowler: name, wickets, runs conceded")
    batting_collapse: str = Field(description="Worst batting collapse in the match")
    commentary: str = Field(description="Two sentence match commentary")


# ── Extraction chain ──────────────────────────────────────────────────────────

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert cricket analyst. Extract detailed insights from the scorecard."),
    ("human", "{scorecard}"),
])

extraction_chain = extraction_prompt | llm.with_structured_output(CricketAnalysis)


# ── State ─────────────────────────────────────────────────────────────────────

class MatchState(TypedDict):
    scorecard: str
    analysis: dict
    confidence: str
    retry_count: int
    style: str                    # "neutral" | "dramatic" | "yoda"
    styled_commentary: Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def extract_node(state: MatchState) -> dict:
    """Extract structured data from raw scorecard text."""
    result = extraction_chain.invoke({"scorecard": state["scorecard"]})
    return {"analysis": result.model_dump()}


def confidence_node(state: MatchState) -> dict:
    """
    Validate extraction quality.
    Returns confidence=low and increments retry_count if key fields are vague.
    Max 2 retries to prevent infinite loops.
    """
    analysis = state["analysis"]
    issues = []

    if len(analysis["key_turning_point"]) < 20:
        issues.append("turning point too vague")
    if len(analysis["commentary"]) < 50:
        issues.append("commentary too short")
    if not analysis["top_performer"]:
        issues.append("missing top performer")

    if issues and state["retry_count"] < 2:
        print(f"[confidence_node] Low confidence — {issues}. Retrying...")
        return {"confidence": "low", "retry_count": state["retry_count"] + 1}

    return {"confidence": "high"}


def commentary_node(state: MatchState) -> dict:
    """Generate styled match commentary based on extracted analysis."""
    analysis = state["analysis"]

    commentary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a cricket commentator. Generate match commentary in the following style: {style}. "
            "Neutral means factual and calm. Dramatic means emotional and intense. "
            "Yoda means speak like Yoda from Star Wars.",
        ),
        (
            "human",
            "Match result: {match_result}\n"
            "Top performer: {top_performer}\n"
            "Key moment: {key_turning_point}\n"
            "Collapse: {batting_collapse}",
        ),
    ])

    commentary_chain = commentary_prompt | llm
    response = commentary_chain.invoke({
        "style": state["style"],
        "match_result": analysis["match_result"],
        "top_performer": analysis["top_performer"],
        "key_turning_point": analysis["key_turning_point"],
        "batting_collapse": analysis["batting_collapse"],
    })

    return {"styled_commentary": response.content}


def report_node(state: MatchState) -> dict:
    """Print the final match report."""
    analysis = state["analysis"]
    print(f"""
╔══════════════════════════════════════════╗
║           PROJECT CREASE REPORT          ║
╚══════════════════════════════════════════╝
Result        : {analysis['match_result']}
Top Performer : {analysis['top_performer']}
Turning Point : {analysis['key_turning_point']}
Top Bowlers   : {analysis['top_bowler_1']} | {analysis['top_bowler_2']} | {analysis['top_bowler_3']}
Collapse      : {analysis['batting_collapse']}
Confidence    : {state['confidence']}

── Styled Commentary ({state['style'].upper()}) ──
{state['styled_commentary']}
""")
    return {}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_confidence(state: MatchState) -> str:
    """Route back to extract on low confidence, forward to commentary on high."""
    return "extract" if state["confidence"] == "low" else "commentary"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(MatchState)

    graph.add_node("extract", extract_node)
    graph.add_node("confidence", confidence_node)
    graph.add_node("commentary", commentary_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "confidence")
    graph.add_conditional_edges("confidence", route_confidence)
    graph.add_edge("commentary", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def analyse(scorecard: str, style: str = "neutral") -> dict:
    """
    Run the full Project Crease pipeline.

    Args:
        scorecard: Raw scorecard text pasted by the user.
        style: Commentary style — "neutral", "dramatic", or "yoda".

    Returns:
        Final state dict with analysis and styled_commentary.
    """
    app = build_graph()
    return app.invoke({
        "scorecard": scorecard,
        "analysis": {},
        "confidence": "",
        "retry_count": 0,
        "style": style,
        "styled_commentary": None,
    })


if __name__ == "__main__":
    sample_scorecard = "Paste scorecard here"
    analyse(sample_scorecard, style="dramatic")