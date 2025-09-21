# nodes.py
from typing import TypedDict, List, Dict, Any
from collections import defaultdict
import random

# ---------------------------------------------------------
# State TypedDict for LangGraph usage (simple dict for demo)
# ---------------------------------------------------------
class DebateState(TypedDict, total=False):
    topic: str
    round: int
    transcript: List[Dict[str, Any]]        # list of {"round":1,"agent":"Scientist","text":"..."}
    memory_scientist: List[str]            # summaries / bullet points relevant to Scientist
    memory_philosopher: List[str]          # summaries / bullet points relevant to Philosopher
    used_args: List[str]                   # set/list to avoid repeats
    winner: str
    judge_summary: str
    # any other helper fields

# ---------------------------------------------------------
# Utility validators & helpers
# ---------------------------------------------------------
def check_turn(state: DebateState, expected_agent: str, logger=None):
    # expected_agent speaks on next round
    current_round = state.get("round", 0)
    # rounds are 1..8. round parity: Scientist starts (odd->Scientist)
    # But we receive expected_agent, so just ensure round parity aligns
    if current_round >= 8:
        raise ValueError("Max rounds already reached")
    if logger:
        logger.info(f"Validating turn: round {current_round+1} expected {expected_agent}")

def has_repeat(candidate: str, state: DebateState):
    used = set(state.get("used_args", []))
    s = candidate.strip().lower()
    return s in used

def mark_used(candidate: str, state: DebateState):
    used = state.setdefault("used_args", [])
    used.append(candidate.strip().lower())

def append_transcript(agent: str, text: str, state: DebateState):
    transcript = state.setdefault("transcript", [])
    r = state.setdefault("round", 0) + 1
    state["round"] = r
    transcript.append({"round": r, "agent": agent, "text": text})

def update_memory_for_agent(agent: str, text: str, state: DebateState):
    """Store a short bullet point for the agent to see (but not full transcript)."""
    bullet = make_bullet_from_text(text)
    if agent == "Scientist":
        mem = state.setdefault("memory_scientist", [])
        mem.append(bullet)
    else:
        mem = state.setdefault("memory_philosopher", [])
        mem.append(bullet)

def make_bullet_from_text(text: str) -> str:
    # naive summarizer / bullet generator (can be swapped with an LLM)
    # Keep 1-2 short sentences extracted heuristically.
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return text[:120]
    bullet = sentences[0]
    if len(sentences) > 1:
        bullet += ". " + sentences[1]
    return bullet[:250]

# ---------------------------------------------------------
# Simulated agent argument generators (deterministic + seeded)
# ---------------------------------------------------------
SCIENCE_TEMPLATES = [
    "Because {} creates demonstrable, measurable risks that can harm humans, AI should be regulated to ensure safety and accountability.",
    "High-risk applications like {} need oversight — regulation would require testing, monitoring, and rollback procedures.",
    "A risk-based framework (safety testing, transparent audits) is essential because {} can fail in unpredictable ways.",
    "Regulation encourages standards, certification, and responsible deployment, especially when {} impacts public safety.",
]

PHILOSOPHY_TEMPLATES = [
    "Regulating {} like medicine could limit intellectual freedom and slow valuable philosophical inquiry.",
    "Rigid regulation may entrench existing paradigms and prevent emergent, beneficial experimentation with {}.",
    "Regulation risks overreach: moral and ethical questions about {} may not be solvable by technical rules alone.",
    "Historical overregulation has sometimes delayed progress; careful, flexible approaches are better than medical-style regulation for {}.",
]

def agent_generate_argument(agent: str, topic: str, round_num: int, state: DebateState) -> str:
    # deterministic-ish selection based on topic + round to avoid repeats
    seed = (hash(topic) ^ round_num ^ (1 if agent=="Scientist" else 2)) & 0xFFFFFFFF
    random.seed(seed)
    if agent == "Scientist":
        templ = random.choice(SCIENCE_TEMPLATES)
        # choose a concrete subexample from topic if possible
        sub = pick_subexample(topic)
        return templ.format(sub)
    else:
        templ = random.choice(PHILOSOPHY_TEMPLATES)
        sub = pick_subexample(topic)
        return templ.format(sub)

def pick_subexample(topic: str) -> str:
    # very small heuristic: return a short phrase derived from topic
    t = topic.lower()
    candidates = []
    if "ai" in t or "artificial" in t:
        candidates = ["autonomous weapons", "medical diagnosis systems", "large-scale surveillance", "automated hiring systems"]
    elif "privacy" in t:
        candidates = ["data sharing", "user profiling", "behavioral tracking"]
    else:
        candidates = [topic, "high-risk applications", "societal-scale deployments"]
    return random.choice(candidates)

# ---------------------------------------------------------
# Node functions: these will be used in the graph
# Signature assumed: node(state: DebateState, config=None, runtime=None)
# ---------------------------------------------------------
def user_input_node(state: DebateState, config=None, runtime=None):
    # expects state['topic'] already set by CLI (main.py)
    topic = state.get("topic", "").strip()
    if not topic:
        raise ValueError("No topic provided")
    # initialize round, transcript, used_args
    state.setdefault("round", 0)
    state.setdefault("transcript", [])
    state.setdefault("used_args", [])
    # store initial memory lists
    state.setdefault("memory_scientist", [])
    state.setdefault("memory_philosopher", [])
    return {"status":"ok","topic":topic}

def agent_node_scientist(state: DebateState, config=None, runtime=None):
    logger = getattr(runtime, "logger", None)
    expected_agent = "Scientist"
    check_turn(state, expected_agent, logger=logger)
    # generate argument
    rnum = state.get("round", 0) + 1
    arg = agent_generate_argument("Scientist", state["topic"], rnum, state)
    if has_repeat(arg, state):
        # slight modification to avoid exact repeat
        arg += " (further clarification: " + pick_subexample(state["topic"]) + ")"
    mark_used(arg, state)
    append_transcript("Scientist", arg, state)
    update_memory_for_agent("Scientist", arg, state)
    if logger:
        logger.info(f"[Round {state['round']}] Scientist: {arg}")
    return {"text": arg}

def agent_node_philosopher(state: DebateState, config=None, runtime=None):
    logger = getattr(runtime, "logger", None)
    expected_agent = "Philosopher"
    check_turn(state, expected_agent, logger=logger)
    rnum = state.get("round", 0) + 1
    arg = agent_generate_argument("Philosopher", state["topic"], rnum, state)
    if has_repeat(arg, state):
        arg += " (added thought: " + pick_subexample(state["topic"]) + ")"
    mark_used(arg, state)
    append_transcript("Philosopher", arg, state)
    update_memory_for_agent("Philosopher", arg, state)
    if logger:
        logger.info(f"[Round {state['round']}] Philosopher: {arg}")
    return {"text": arg}

def memory_node(state: DebateState, config=None, runtime=None):
    # This node can optionally summarize memory for each agent (here already stored)
    # We produce concise summaries (join bullets)
    mem_sci = state.get("memory_scientist", [])
    mem_phil = state.get("memory_philosopher", [])
    # naive summarization: join bullets with semicolons
    state["summary_scientist"] = "; ".join(mem_sci[-3:])  # last 3
    state["summary_philosopher"] = "; ".join(mem_phil[-3:])
    if getattr(runtime, "logger", None):
        runtime.logger.info("Memory updated for both agents.")
    return {"ok": True}

def judge_node(state: DebateState, config=None, runtime=None):
    # Examine transcript + memory and produce final verdict after 8 rounds
    transcript = state.get("transcript", [])
    if len(transcript) < 8:
        raise ValueError("Debate incomplete; judge invoked too early")
    # Heuristics for deciding winner:
    # Count how many "risk", "safety", "standards" tokens for Scientist
    # vs "freedom", "progress", "philosophy" tokens for Philosopher
    sci_score = 0
    phil_score = 0
    for item in transcript:
        txt = item["text"].lower()
        if item["agent"] == "Scientist":
            if any(k in txt for k in ["risk", "safety", "standards", "audit", "testing", "accountab"]):
                sci_score += 2
            if any(k in txt for k in ["data", "medical", "surveil", "autonomous"]):
                sci_score += 1
        else:
            if any(k in txt for k in ["freedom", "progress", "ethical", "autonomy", "philosoph"]):
                phil_score += 2
            if any(k in txt for k in ["overregulate", "slow", "creativity", "experimen"]):
                phil_score += 1

    # also consider number of unique supporting bullets
    sci_score += len(set(state.get("memory_scientist", [])))
    phil_score += len(set(state.get("memory_philosopher", [])))

    # produce summary
    summary_lines = []
    summary_lines.append(f"Topic: {state.get('topic')}")
    summary_lines.append("Transcript summary (round by round):")
    for t in transcript:
        summary_lines.append(f"R{t['round']} {t['agent']}: {t['text']}")
    summary = "\n".join(summary_lines)
    state["judge_summary"] = summary

    if sci_score > phil_score:
        winner = "Scientist"
    elif phil_score > sci_score:
        winner = "Philosopher"
    else:
        winner = "Tie"

    justification = f"sci_score={sci_score}, phil_score={phil_score} -> winner: {winner}"
    state["winner"] = winner
    state["judge_justification"] = justification

    if getattr(runtime, "logger", None):
        runtime.logger.info("[Judge] Summary and verdict produced.")
        runtime.logger.info(f"[Judge] Winner: {winner}. Justification: {justification}")

    return {"winner": winner, "justification": justification, "summary": summary}
