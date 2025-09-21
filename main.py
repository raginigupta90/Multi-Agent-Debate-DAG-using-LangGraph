# main.py
import argparse
from datetime import datetime
from graph_builder import build_graph
from nodes import (
    DebateState,
    user_input_node,
    agent_node_scientist,
    agent_node_philosopher,
    memory_node,
    judge_node,
)
from logger_util import create_log_file, FileLogger
from dag_viz import generate_dag_image
import os
import time

class RuntimeStub:
    """Simple runtime object to carry logger and any other runtime helpers."""
    def __init__(self, logger):
        self.logger = logger

def run_debate(topic: str, verbose=True):
    # prepare logger file
    log_path = create_log_file()
    logger = FileLogger(log_path)
    runtime = RuntimeStub(logger)

    # Build (or load) graph (not actively invoked via langgraph runtime in this simple runner)
    graph = build_graph()

    # initialize state
    state = DebateState()
    state["topic"] = topic
    # call user input node
    logger.info(f"Topic set: {topic}")
    user_input_node(state, runtime=runtime)

    # Alternate 8 rounds: Scientist starts (round 1)
    print(f"Starting debate between Scientist and Philosopher on: {topic}")
    logger.info("Starting debate between Scientist and Philosopher")
    max_rounds = 8
    # ensure state['round'] starts at 0
    state["round"] = 0

    while state["round"] < max_rounds:
        next_round = state["round"] + 1
        # Scientist on odd rounds, Philosopher on even.
        if next_round % 2 == 1:
            # Scientist turn
            res = agent_node_scientist(state, runtime=runtime)
            print(f"[Round {state['round']}] Scientist: {res['text']}")
            logger.info(f"[Round {state['round']}] Scientist: {res['text']}")
        else:
            # Philosopher turn
            res = agent_node_philosopher(state, runtime=runtime)
            print(f"[Round {state['round']}] Philosopher: {res['text']}")
            logger.info(f"[Round {state['round']}] Philosopher: {res['text']}")

        # after each speaking turn, update memory node
        memory_node(state, runtime=runtime)
        logger.info(f"Memory state: sci({len(state.get('memory_scientist',[]))}) phil({len(state.get('memory_philosopher',[]))})")

        # small pause for better CLI readability
        time.sleep(0.15)

    # rounds done -> call judge
    judge_out = judge_node(state, runtime=runtime)
    print("\n[Judge] Summary of debate (short):")
    # print only a short extract; full summary is saved
    print(state["judge_summary"].split("\n")[0:6])  # just sample lines
    print(f"\n[Judge] Winner: {state['winner']}")
    print(f"Reason: {state['judge_justification']}")
    logger.info("Final judge output recorded.")
    # save full state dump to log for submission
    logger.info("Full state dump:")
    import json
    logger.info(json.dumps(state, indent=2, default=str))
    print(f"\nFull log written to: {log_path}")

    # generate DAG diagram
    try:
        dag_path = generate_dag_image()
        print(f"DAG diagram generated: {dag_path}")
        logger.info(f"DAG generated at {dag_path}")
    except Exception as e:
        logger.info(f"Failed to render DAG: {e}")
        print("Warning: graphviz DAG generation failed. Ensure graphviz installed (system) and Python graphviz package.")

    return state, log_path

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate CLI (LangGraph-style)")
    parser.add_argument("--topic", "-t", type=str, help="Topic for debate. If omitted, prompts the user.")
    args = parser.parse_args()
    topic = args.topic
    if not topic:
        topic = input("Enter topic for debate: ").strip()
    run_debate(topic)

if __name__ == "__main__":
    main()
