import numpy as np
import random
from rich.console import Console
from rich.table import Table
from rich import box
from rich.prompt import Prompt

console = Console()

# -----------------------------------------------------
# Helper: manual unicode bar (compatible with all Rich versions)
# -----------------------------------------------------
def make_bar(value, max_value, width=20):
    ratio = value / max_value if max_value > 0 else 0
    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "░" * empty

# -----------------------------------------------------
# User-defined components
# -----------------------------------------------------
z0 = np.array([0.2, 0.5, 0.1, 0.15, 0.05])

S = np.array([
    [0.2, 0.5, 0.3, 0.0, 0.0],
    [0.4, 0.2, 0.4, 0.0, 0.0],
    [0.0, 0.3, 0.4, 0.3, 0.0],
    [0.0, 0.0, 0.4, 0.3, 0.3],
    [0.0, 0.0, 0.0, 0.5, 0.5]
])

alpha = 0.8
n_nodes = len(z0)

# -----------------------------------------------------
# Initialization
# -----------------------------------------------------
current = np.random.choice(n_nodes, p=z0)
visits = np.zeros(n_nodes, dtype=int)
step = 0

console.print("[bold green]Random Walk With Restart Simulation[/bold green]")
console.print(f"α = {alpha} (walk probability), 1-α = {1-alpha} (restart probability)\n")
console.print("Press Enter for next step, Ctrl+C to stop.\n")

# -----------------------------------------------------
# Main interactive loop
# -----------------------------------------------------
while True:
    visits[current] += 1
    step += 1
    freqs = visits / step

    # Create table
    table = Table(
        title=f"Step {step} — Current Node: [bold yellow]{current+1}[/bold yellow]",
        title_style="bold cyan",
        box=box.ROUNDED
    )

    table.add_column("Node", justify="center")
    table.add_column("Visits", justify="center")
    table.add_column("Frequency", justify="center")
    table.add_column("Histogram", justify="left")

    max_visits = max(visits) if max(visits) > 0 else 1

    for i in range(n_nodes):
        bar = make_bar(visits[i], max_visits, width=20)
        table.add_row(
            f"{i+1}",
            str(visits[i]),
            f"{freqs[i]:.3f}",
            bar
        )

    console.clear()
    console.print(table)

    Prompt.ask("Press [enter] for next step", default="")

    # WALK step
    if random.random() < alpha:
        current = np.random.choice(n_nodes, p=S[current])
    else:
        current = np.random.choice(n_nodes, p=z0)
