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

# ✅ Compute Path Distance Based on Weights
def calculate_total_distance(graph, path):
    """
    Calculate total path weight for a given path in a weighted graph.
    Uses Euclidean distance if no explicit weight is available.
    """
    total_distance = 0
    if path and len(path) > 1:
        for u, v in zip(path[:-1], path[1:]):
            if graph.has_edge(u, v):
                total_distance += graph[u][v].get('weight', 1)  # Use weight if available
            else:
                print(f"Warning: Edge ({u}, {v}) not found in graph!")  # Debugging
    return total_distance

# ✅ Select Valid `src` and `dest`
def get_valid_src_dest(udg, node_mapping):
    """
    Finds a valid `src` and `dest` where:
    - A path exists in UDG.
    - Both nodes have mapped grid equivalents.
    """
    while True:
        src, dest = random.sample(list(udg.nodes), 2)
        if nx.has_path(udg, src, dest) and src in node_mapping and dest in node_mapping:
            return src, dest

# ✅ Evaluate Path Distance for Each Algorithm
def evaluate_path_distance(num_runs=1000, num_nodes=50):
    algorithms = {
        'Dijkstra': dijkstra_path,
        'A*': astar_path,
        'Bellman-Ford': bellman_ford_path
    }
    results = {algo: {'Run': [], 'UDG Path Distance': [], 'Mapped Grid to UDG Path Distance': []} for algo in algorithms}

    for run in range(1, num_runs + 1):
        # Step 1: Generate UDG and Grid
        udg, udg_positions = generate_udg(num_nodes)
        grid, grid_positions, node_mapping = map_udg_to_grid_static(udg, udg_positions)

        # Step 2: Get valid `src` and `dest`
        src, dest = get_valid_src_dest(udg, node_mapping)

        for algo_name, algo_func in algorithms.items():
            try:
                # Step 3: Compute Shortest Path in UDG
                udg_path = algo_func(udg, src, dest)
                udg_distance = calculate_total_distance(udg, udg_path)
            except nx.NetworkXNoPath:
                udg_distance = None

            try:
                # Step 4: Compute Shortest Path in Grid
                src_grid, dest_grid = node_mapping[src], node_mapping[dest]
                grid_path = algo_func(grid, src_grid, dest_grid)
            except nx.NetworkXNoPath:
                grid_path = None

            try:
                # Step 5: Convert Grid Path to Mapped UDG Path
                mapped_udg_path = map_grid_to_udg_path(grid_path, node_mapping, udg, src, dest)
                mapped_udg_distance = calculate_total_distance(udg, mapped_udg_path)
            except Exception:
                mapped_udg_distance = None

            # Step 6: Store Results
            results[algo_name]['Run'].append(run)
            results[algo_name]['UDG Path Distance'].append(udg_distance)
            results[algo_name]['Mapped Grid to UDG Path Distance'].append(mapped_udg_distance)

    return results

# ✅ Visualization: Line Graph with Rolling Averages
def visualize_path_distance(results):
    for algo, data in results.items():
        df = pd.DataFrame(data)

        # Compute rolling average for trend smoothing
        df['UDG Path (Avg)'] = df['UDG Path Distance'].rolling(window=50, min_periods=1).mean()
        df['Mapped Grid to UDG Path (Avg)'] = df['Mapped Grid to UDG Path Distance'].rolling(window=50, min_periods=1).mean()

        plt.figure(figsize=(12, 6))

        # Plot Raw Data
        plt.plot(df['Run'], df['UDG Path Distance'], label='UDG Path Distance (Raw)', alpha=0.3, linestyle='-', color='blue')
        plt.plot(df['Run'], df['Mapped Grid to UDG Path Distance'], label='Mapped Grid to UDG Path Distance (Raw)', alpha=0.3, linestyle='-.', color='red')

        # Plot Rolling Averages for Trend Visualization
        plt.plot(df['Run'], df['UDG Path (Avg)'], label='UDG Path Distance (Avg)', linewidth=2, linestyle='-', color='darkblue')
        plt.plot(df['Run'], df['Mapped Grid to UDG Path (Avg)'], label='Mapped Grid to UDG Path Distance (Avg)', linewidth=2, linestyle='-.', color='darkred')

        plt.xlabel('Run')
        plt.ylabel('Path Distance')
        plt.title(f'Path Distance Comparison - {algo}')
        plt.legend()
        plt.grid(True)
        plt.show()

# ✅ Main Execution
results = evaluate_path_distance(num_runs=1000, num_nodes=50)
visualize_path_distance(results)
