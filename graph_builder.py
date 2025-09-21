# graph_builder.py
from langgraph.graph import StateGraph, START, END  # LangGraph APIs
from nodes import (
    DebateState,
    user_input_node,
    agent_node_scientist,
    agent_node_philosopher,
    memory_node,
    judge_node,
)
from logger_util import FileLogger, create_log_file

def build_graph(logger_path=None):
    # build graph and return compiled graph object
    graph = StateGraph(DebateState)

    # Add nodes
    graph.add_node("UserInput", user_input_node)
    graph.add_node("Scientist", agent_node_scientist)
    graph.add_node("Philosopher", agent_node_philosopher)
    graph.add_node("Memory", memory_node)
    graph.add_node("Judge", judge_node)

    # Edges: Start -> UserInput -> Scientist -> Memory -> Philosopher -> Memory -> Scientist ...
    graph.add_edge(START, "UserInput")
    graph.add_edge("UserInput", "Scientist")

    # we will create 4 alternating pairs for 8 rounds.
    # Implementation: after Scientist -> Memory -> Philosopher -> Memory, loop back until 8 rounds reached; but LangGraph needs a static graph.
    # A simple approach: unroll the 8 rounds explicitly by using multiple copies of scientist/philosopher nodes or use the same node but rely on state['round'] to stop.
    # For simplicity we will alternate Scientist -> Memory -> Philosopher -> Memory and let the runtime call edges repeatedly until round==8 (runtime control).
    graph.add_edge("Scientist", "Memory")
    graph.add_edge("Memory", "Philosopher")
    graph.add_edge("Philosopher", "Memory")
    # After memory, either go to Scientist again or to Judge if rounds done:
    graph.add_edge("Memory", "Scientist")
    # Finally, when rounds>=8 we route to Judge (the runtime will invoke Judge once condition met).
    graph.add_edge("Memory", "Judge")
    graph.add_edge("Judge", END)

    return graph
