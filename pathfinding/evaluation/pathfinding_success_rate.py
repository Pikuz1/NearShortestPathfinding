import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from grid_based_abstraction import generate_udg, map_udg_to_grid_static, map_grid_to_udg_path

# Define Pathfinding Algorithms
def dijkstra_path(graph, src, dest):
    try:
        return nx.shortest_path(graph, src, dest, method='dijkstra')
    except nx.NetworkXNoPath:
        return None

def astar_path(graph, src, dest):
    try:
        return nx.astar_path(graph, src, dest)
    except nx.NetworkXNoPath:
        return None

def bellman_ford_path(graph, src, dest):
    try:
        return nx.shortest_path(graph, src, dest, method='bellman-ford')
    except nx.NetworkXNoPath:
        return None

# ✅ Function to ensure valid source and destination
def get_valid_src_dest(udg, node_mapping):
    """
    Selects a valid src and dest where:
    - A path exists in UDG.
    - The mapped nodes also exist in the grid.
    """
    while True:
        src, dest = random.sample(list(udg.nodes), 2)
        if nx.has_path(udg, src, dest) and node_mapping.get(src) and node_mapping.get(dest):
            return src, dest

# ✅ Function to Evaluate Success Rate
def evaluate_success_rate(num_runs=1000, num_nodes=50):
    algorithms = {
        'Dijkstra': dijkstra_path,
        'A*': astar_path,
        'Bellman-Ford': bellman_ford_path
    }
    success_counts = {algo: {'UDG': 0, 'Mapped Grid': 0} for algo in algorithms}

    for _ in range(num_runs):
        # Step 1: Generate UDG and Grid
        udg, udg_positions = generate_udg(num_nodes)
        grid, grid_positions, node_mapping = map_udg_to_grid_static(udg, udg_positions)

        # Step 2: Choose valid `src` and `dest`
        src, dest = get_valid_src_dest(udg, node_mapping)

        for algo_name, algo_func in algorithms.items():
            # Step 3: Check if a valid path exists in UDG
            if algo_func(udg, src, dest):
                success_counts[algo_name]['UDG'] += 1  # ✅ Increase UDG success count

            # Step 4: Compute Mapped Grid to UDG Path
            src_grid, dest_grid = node_mapping[src], node_mapping[dest]
            grid_path = algo_func(grid, src_grid, dest_grid)

            if grid_path:
                mapped_udg_path = map_grid_to_udg_path(grid_path, node_mapping, udg, src, dest)
                if mapped_udg_path:
                    success_counts[algo_name]['Mapped Grid'] += 1  # ✅ Increase Mapped Grid success count

    # ✅ Convert to Success Rate (%)
    success_rates = {
        algo: {
            'UDG': (success_counts[algo]['UDG'] / num_runs) * 100,
            'Mapped Grid': (success_counts[algo]['Mapped Grid'] / num_runs) * 100
        } for algo in algorithms
    }

    return success_rates

# ✅ Visualization Function
def visualize_success_rate(success_rates):
    """
    Generates a bar chart comparing success rates for UDG and Mapped Grid paths.
    """
    df = pd.DataFrame(success_rates).T  # Convert dictionary to DataFrame

    df.plot(kind='bar', figsize=(10, 6), rot=0, color=['blue', 'orange'])

    plt.xlabel('Algorithm')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison: UDG vs. Mapped Grid to UDG')
    plt.ylim(0, 100)  # Ensure y-axis is from 0 to 100%
    plt.legend(title='Graph Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(plt.gca().patches):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.show()

# ✅ Main Execution
success_rates = evaluate_success_rate(num_runs=1000, num_nodes=50)
visualize_success_rate(success_rates)
