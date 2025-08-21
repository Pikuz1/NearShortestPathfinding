import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
from grid_based_abstraction import generate_udg, map_udg_to_grid_static, map_grid_to_udg_path

# ✅ Define Pathfinding Algorithms
def dijkstra_path(graph, src, dest):
    return nx.shortest_path(graph, src, dest, method='dijkstra')

def astar_path(graph, src, dest):
    return nx.astar_path(graph, src, dest)

def bellman_ford_path(graph, src, dest):
    return nx.shortest_path(graph, src, dest, method='bellman-ford')

# ✅ Function to ensure valid src and dest
def get_valid_src_dest(udg, node_mapping):
    """Find a valid source and destination where paths exist in both UDG and Grid."""
    while True:
        src, dest = random.sample(list(udg.nodes), 2)
        if nx.has_path(udg, src, dest) and node_mapping.get(src) and node_mapping.get(dest):
            return src, dest

# ✅ Compute Confidence Interval
def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for computational times."""
    data = np.array([x for x in data if x is not None])  # Remove None values
    if len(data) == 0:
        return None, None

    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))  # Standard Error
    margin = std_err * 1.96  # 95% Confidence Interval
    return mean, margin

# ✅ Evaluate Computation Time for Each Algorithm (FAIR COMPARISON)
def evaluate_computation_time(num_runs=1000, num_nodes=50):
    algorithms = {
        'Dijkstra': dijkstra_path,
        'A*': astar_path,
        'Bellman-Ford': bellman_ford_path
    }
    results = {algo: {'UDG': [], 'Grid': [], 'Mapped Grid': []} for algo in algorithms}

    for _ in range(num_runs):
        # Step 1: Generate UDG and Grid
        udg, udg_positions = generate_udg(num_nodes)
        grid, grid_positions, node_mapping = map_udg_to_grid_static(udg, udg_positions)

        # Step 2: Get a valid `src` and `dest`
        src, dest = get_valid_src_dest(udg, node_mapping)

        for algo_name, algo_func in algorithms.items():
            # Compute paths inside the same loop for fair timing
            try:
                start_time = time.time()
                udg_path = algo_func(udg, src, dest)
                udg_time = time.time() - start_time
            except nx.NetworkXNoPath:
                udg_path = None
                udg_time = None

            try:
                src_grid, dest_grid = node_mapping[src], node_mapping[dest]
                start_time = time.time()
                grid_path = algo_func(grid, src_grid, dest_grid)
                grid_time = time.time() - start_time
            except nx.NetworkXNoPath:
                grid_path = None
                grid_time = None

            try:
                start_time = time.time()
                mapped_udg_path = map_grid_to_udg_path(grid_path, node_mapping, udg, src, dest)
                mapped_udg_time = time.time() - start_time
            except nx.NetworkXNoPath:
                mapped_udg_path = None
                mapped_udg_time = None

            results[algo_name]['UDG'].append(udg_time)
            results[algo_name]['Grid'].append(grid_time)
            results[algo_name]['Mapped Grid'].append(mapped_udg_time)

    return results

# ✅ Visualization: Confidence Interval Bar Graph
def visualize_computation_time(results):
    plt.figure(figsize=(12, 6))

    algorithms = list(results.keys())
    categories = ['UDG', 'Grid', 'Mapped Grid']
    colors = ['blue', 'green', 'orange']

    means = {category: [] for category in categories}
    errors = {category: [] for category in categories}

    for category in categories:
        for algo in algorithms:
            mean, margin = compute_confidence_interval(results[algo][category])
            means[category].append(mean)
            errors[category].append(margin)

    # Plot bars for each category with confidence intervals
    bar_width = 0.2
    x = np.arange(len(algorithms))

    for i, category in enumerate(categories):
        plt.bar(x + i * bar_width, means[category], yerr=errors[category], capsize=8,
                color=colors[i], width=bar_width, alpha=0.7, label=category)

    plt.xlabel('Pathfinding Algorithm')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison (Confidence Interval)')
    plt.xticks(x + bar_width, algorithms)
    plt.ylim(0, 0.01)
    plt.legend()
    plt.grid(axis='y')

    # Display values on bars
    for i, algo in enumerate(algorithms):
        for j, category in enumerate(categories):
            mean, margin = compute_confidence_interval(results[algo][category])
            if mean is not None:
                plt.text(x[i] + j * bar_width, mean + (errors[category][i] if errors[category][i] else 0.002),
                         f'{mean:.4f}', ha='center', fontsize=10)

    plt.show()

# ✅ Main Execution
results = evaluate_computation_time(num_runs=1000, num_nodes=50)
visualize_computation_time(results)
