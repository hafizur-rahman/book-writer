"""LangGraph agentic authoring loop for a single chapter."""
from __future__ import annotations
import json, operator
from typing import List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from chapters import Chapter

class WritingState(TypedDict):
    chapter: Chapter
    research: str
    outline: str
    draft: str
    review: str
    final: str
    iterations: int
    decision: str                 # "improve" | "pass"
    messages: list                # reduce-combiner: Annotated[list, operator.add]

# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------
RESEARCH_SYS = ("You are a research assistant for a graduate textbook on deep "
                "learning and LLMs. Produce concise, accurate research notes "
                "with the key facts, formulas, and canonical code APIs for the "
                "topic. Be technically precise but keep an intuitive thread.")

OUTLINE_SYS = ("You are a technical author. Produce a tight chapter outline as a "
               "Markdown skeleton with numbered sections and bullet points.")

DRAFT_SYS = (
    "You are a world-class professor writing an authoritative textbook chapter "
    "on deep learning and large language models. Write a FULL chapter in "
    "Markdown following these rules:\n"
    "1. Open with an INTUITIVE explanation (analogies, 'the big idea', "
    "intuition before math).\n"
    "2. Then the TECHNICAL DEPTH: precise definitions, the core math "
    "(LaTeX $...$), and how the breakthrough actually works.\n"
    "3. Include a WORKING CODE EXAMPLE in PyTorch/LangGraph that a reader can run.\n"
    "4. Add 'Breakthrough' and 'Historical Context' framing — where this sits in "
    "the progression from earlier work.\n"
    "5. End with a short 'Explain it like I'm 5' summary and key takeaways.\n"
    "6. Keep prose clear and correct; never invent facts or formulas."
)

REVIEW_SYS = (
    "You are a rigorous technical reviewer. Read the chapter draft and return ONLY "
    "valid JSON with keys: score (0-10 integer), pass (bool), "
    "feedback (list of concrete fixes). Pass only if math is correct, code is "
    "runnable, and intuition + technical depth are both present."
)

EDIT_SYS = (
    "You are the author. Revise the draft using ONLY the reviewer feedback. "
    "Return the full corrected Markdown chapter."
)


def _build(state: dict) -> str:
    c: Chapter = state["chapter"]
    return (
        f"# {c.num}. {c.title}\n\nAbstract: {c.abstract}\n"
        f"Breakthrough: {c.breakthrough}\n\nSubtopics: {' | '.join(c.subtopics)}\n\n"
        f"Signature code to teach: `{c.key_code}`"
    )


def build_workflow(cpu_model, gpu_model):
    """Compile the agentic authoring graph bound to a local LLM."""

    def _research(state: dict) -> dict:
        sys = SystemMessage(RESEARCH_SYS)
        user = HumanMessage(f"Chapter:\n{_build(state)}\n\nResearch notes:")

        print(sys)
        print(user)

        resp = gpu_model.invoke(f"{RESEARCH_SYS}\n\n{_build(state)}").content

        print(resp)

        return {"research": resp,
                "messages": [sys, user, AIMessage(resp)]}

    def _outline(state: dict) -> dict:
        sys = SystemMessage(OUTLINE_SYS)
        prompt = f"{OUTLINE_SYS}\n\n{_build(state)}\n\nResearch notes:\n{state['research']}"
        
        print(sys)
        print(prompt)

        resp = gpu_model.invoke(prompt).content
        
        print(resp)
        
        return {"outline": resp, "messages": [sys, AIMessage(resp)]}

    def _draft(state: dict) -> dict:
        sys = SystemMessage(DRAFT_SYS)
        prompt = (
            f"{DRAFT_SYS}\n\n{state['outline'] or _build(state)}\n\n"
            f"Notes:\n{state['research']}\n\n"
            f"Write the full chapter now."
        )

        print(sys)
        print(prompt)

        resp = cpu_model.invoke(prompt).content
        
        print(resp)
        
        return {"draft": resp, "messages": [sys, AIMessage(resp)]}

    def _review(state: dict) -> dict:
        prompt = f"{REVIEW_SYS}\n\nDRAFT:\n{state['draft']}"
        print(prompt)

        resp = gpu_model.invoke(prompt).content        
        print(resp)

        try:
            data = json.loads(resp.split("```")[0].strip().lstrip("```json"))
        except Exception:
            data = {"score": 5, "pass": False, "feedback": ["Could not parse JSON"]}
        decision = "pass" if data.get("pass") and data.get("score", 0) >= 7 else "improve"
        return {"review": resp, "decision": decision}

    def _edit(state: dict) -> dict:
        sys = SystemMessage(EDIT_SYS)
        prompt = f"{EDIT_SYS}\n\nDraft:\n{state['draft']}\n\nFeedback:\n{state['review']}"
        
        print(sys)
        print(prompt)
        
        resp = cpu_model.invoke(prompt).content
        print(resp)
        
        return {"draft": resp,
                "iterations": state.get("iterations", 0) + 1,
                "messages": [sys, AIMessage(resp)]}

    def _finalize(state: dict) -> dict:
        return {"final": state["draft"], "decision": "pass"}

    g = StateGraph(WritingState)
    g.add_node("research", _research)
    g.add_node("outline", _outline)
    g.add_node("draft", _draft)
    g.add_node("review", _review)
    g.add_node("edit", _edit)
    g.add_node("finalize", _finalize)

    g.add_edge(START, "research")
    g.add_edge("research", "outline")
    g.add_edge("outline", "draft")
    g.add_edge("draft", "review")
    # Self-critic loop: improve until judge passes or max iterations hit.
    g.add_conditional_edges(
        "review", lambda s: s["decision"],
        {"improve": "edit", "pass": "finalize"})
    g.add_edge("edit", "draft")
    g.add_edge("finalize", END)

    return g.compile()