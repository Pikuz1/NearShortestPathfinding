import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
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
    """ Selects a valid src and dest where:
    - A path exists in UDG.
    - The mapped nodes also exist in the grid.
    """
    while True:
        src, dest = random.sample(list(udg.nodes), 2)
        if nx.has_path(udg, src, dest) and node_mapping.get(src) and node_mapping.get(dest):
            return src, dest

# ✅ Evaluate Edge Count for UDG Path vs. Grid Path vs. Mapped Grid to UDG Path
def evaluate_edge_count(num_runs=1000, num_nodes=50):
    algorithms = {
        'Dijkstra': dijkstra_path,
        'A*': astar_path,
        'Bellman-Ford': bellman_ford_path
    }
    results = {algo: {'Run': [], 'UDG Path': [], 'Grid Path': [], 'Mapped Grid to UDG Path': []} for algo in algorithms}

    for run in range(1, num_runs + 1):
        # Step 1: Generate UDG and Grid
        udg, udg_positions = generate_udg(num_nodes)
        grid, grid_positions, node_mapping = map_udg_to_grid_static(udg, udg_positions)

        # Step 2: Get a valid `src` and `dest`
        src, dest = get_valid_src_dest(udg, node_mapping)

        # Store edge counts
        for algo_name, algo_func in algorithms.items():
            try:
                # Step 3: Compute Path in UDG
                udg_path = algo_func(udg, src, dest)
                udg_path_edges = len(udg_path) - 1 if udg_path else None
            except nx.NetworkXNoPath:
                udg_path_edges = None

            try:
                # Step 4: Compute Path in Grid
                src_grid, dest_grid = node_mapping[src], node_mapping[dest]
                grid_path = algo_func(grid, src_grid, dest_grid)
                grid_path_edges = len(grid_path) - 1 if grid_path else None
            except nx.NetworkXNoPath:
                grid_path_edges = None

            try:
                # Step 5: Convert Grid Path to UDG Path
                mapped_udg_path = map_grid_to_udg_path(grid_path, node_mapping, udg, src, dest)
                mapped_udg_edges = len(mapped_udg_path) - 1 if mapped_udg_path else None
            except (nx.NetworkXNoPath, KeyError, TypeError):
                mapped_udg_edges = None

            results[algo_name]['Run'].append(run)
            results[algo_name]['UDG Path'].append(udg_path_edges)
            results[algo_name]['Grid Path'].append(grid_path_edges)
            results[algo_name]['Mapped Grid to UDG Path'].append(mapped_udg_edges)

    return results

# ✅ Visualization Function: Line Graph for 1000 Runs with Rolling Average
def visualize_edge_count(results):
    for algo, data in results.items():
        df = pd.DataFrame(data)

        # Compute rolling average for trend smoothing
        df['UDG Path (Avg)'] = df['UDG Path'].rolling(window=50, min_periods=1).mean()
        df['Grid Path (Avg)'] = df['Grid Path'].rolling(window=50, min_periods=1).mean()
        df['Mapped Grid to UDG Path (Avg)'] = df['Mapped Grid to UDG Path'].rolling(window=50, min_periods=1).mean()

        plt.figure(figsize=(12, 6))

        # Original Data
        plt.plot(df['Run'], df['UDG Path'], label='UDG Path (Raw)', alpha=0.3, linestyle='-', color='blue')
        plt.plot(df['Run'], df['Grid Path'], label='Grid Path (Raw)', alpha=0.3, linestyle='--', color='green')
        plt.plot(df['Run'], df['Mapped Grid to UDG Path'], label='Mapped Grid to UDG Path (Raw)', alpha=0.3, linestyle='-.', color='red')

        # Rolling Averages for Trends
        plt.plot(df['Run'], df['UDG Path (Avg)'], label='UDG Path (Avg)', linewidth=2, linestyle='-', color='darkblue')
        plt.plot(df['Run'], df['Grid Path (Avg)'], label='Grid Path (Avg)', linewidth=2, linestyle='--', color='darkgreen')
        plt.plot(df['Run'], df['Mapped Grid to UDG Path (Avg)'], label='Mapped Grid to UDG Path (Avg)', linewidth=2, linestyle='-.', color='darkred')

        plt.xlabel('Run')
        plt.ylabel('Number of Edges')
        plt.title(f'Edge Count Comparison - {algo}')
        plt.legend()
        plt.grid(True)
        plt.show()

# ✅ Main Execution
results = evaluate_edge_count(num_runs=1000, num_nodes=50)
visualize_edge_count(results)
