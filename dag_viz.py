# dag_viz.py
from graphviz import Digraph

def generate_dag_image(path="dag.png"):
    dot = Digraph(comment="Debate LangGraph DAG", format="png")
    dot.node("START")
    dot.node("UserInput", "UserInputNode\n(get topic)")
    dot.node("Scientist")
    dot.node("Memory")
    dot.node("Philosopher")
    dot.node("Judge")
    dot.node("END")

    dot.edge("START", "UserInput")
    dot.edge("UserInput", "Scientist")
    dot.edge("Scientist", "Memory")
    dot.edge("Memory", "Philosopher")
    dot.edge("Philosopher", "Memory")
    dot.edge("Memory", "Scientist", label="loop until round 8")
    dot.edge("Memory", "Judge", label="if rounds == 8")
    dot.edge("Judge", "END")

    dot.render(filename=path, cleanup=True)
    return path + ".png"
