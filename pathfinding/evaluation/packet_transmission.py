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
    while True:
        src, dest = random.sample(list(udg.nodes), 2)
        if nx.has_path(udg, src, dest) and node_mapping.get(src) and node_mapping.get(dest):
            return src, dest

# ✅ Simulate Packet Transmission Success
def simulate_packet_transmission(path, graph, edge_success_prob=0.9):
    """
    Simulates packet transmission along a given path.
    Each edge has a probability of successful transmission.

    Parameters:
    - path: List of nodes representing the path.
    - graph: The NetworkX graph.
    - edge_success_prob: Can be:
        1. A float (default probability for all edges).
        2. A dictionary mapping edge tuples (u, v) -> probability.

    Returns:
    - True if packet transmission is successful, False otherwise.
    """
    if not path or len(path) < 2:
        print("⚠ Transmission failed: Invalid path.")
        return False

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        if not graph.has_edge(u, v):
            print(f"⚠ Transmission failed: Edge ({u}, {v}) does not exist.")
            return False  # Edge does not exist

        # Determine the success probability for this edge
        if isinstance(edge_success_prob, dict):  # If a dictionary is provided
            prob = edge_success_prob.get((u, v), 0.9)  # Default to 0.9 if edge not in dictionary
        else:
            prob = edge_success_prob  # Use given float probability

        if random.random() > prob:
            print(f"⚠ Packet lost at edge ({u}, {v})")
            return False  # Packet lost in transmission

    print("✅ Packet successfully transmitted!")
    return True  # Successfully transmitted


# ✅ Function to Evaluate Success Rate
def evaluate_success_rate(num_runs=1000, num_nodes=50, packet_success_prob=0.9):
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
            udg_path = algo_func(udg, src, dest)
            if udg_path and simulate_packet_transmission(udg_path, udg, packet_success_prob):
                success_counts[algo_name]['UDG'] += 1  # ✅ Increase UDG success count

            # Step 4: Compute Mapped Grid to UDG Path
            src_grid, dest_grid = node_mapping[src], node_mapping[dest]
            grid_path = algo_func(grid, src_grid, dest_grid)

            if grid_path:
                mapped_udg_path = map_grid_to_udg_path(grid_path, node_mapping, udg, src, dest)
                if mapped_udg_path and simulate_packet_transmission(mapped_udg_path, udg, packet_success_prob):
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
    df = pd.DataFrame(success_rates).T  # Convert dictionary to DataFrame
    df.plot(kind='bar', figsize=(10, 6), rot=0, color=['blue', 'orange'])

    plt.xlabel('Algorithm')
    plt.ylabel('Packet Transmission Success Rate (%)')
    plt.title('Packet Transmission Success Rate: UDG vs. Mapped Grid to UDG')
    plt.ylim(0, 100)  # Ensure y-axis is from 0 to 100%
    plt.legend(title='Graph Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(plt.gca().patches):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.show()

# ✅ Main Execution
success_rates = evaluate_success_rate(num_runs=1000, num_nodes=50, packet_success_prob=0.9)
visualize_success_rate(success_rates)
