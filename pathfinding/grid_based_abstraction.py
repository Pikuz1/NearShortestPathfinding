import networkx as nx
import matplotlib.pyplot as plt
import random
import time

DISTANCE = 8  # Distance for grid layout

# --- Generate UDG ---
def generate_udg(num_nodes, radius=10):
    """
    Generate a Unit Disk Graph (UDG) with random node positions.
    Nodes are connected if their Euclidean distance is less than a given radius.
    """
    udg = nx.Graph()
    udg_positions = {i: (random.randint(0, 50), random.randint(0, 50)) for i in range(num_nodes)}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x1, y1 = udg_positions[i]
            x2, y2 = udg_positions[j]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if distance <= radius:
                udg.add_edge(i, j, weight=distance)  # Assign weight = Euclidean distance

    return udg, udg_positions

def calculate_total_distance(graph, positions, path):
    """
    Calculate the total weighted distance of a given path in a graph.

    Args:
    - graph: The graph (UDG or Mapped Grid).
    - positions: Node positions (for Euclidean fallback).
    - path: The path as a list of nodes.

    Returns:
    - Total path distance.
    """
    total_distance = 0
    if path and len(path) > 1:
        for u, v in zip(path[:-1], path[1:]):
            if graph.has_edge(u, v):
                total_distance += graph[u][v].get('weight',
                    ((positions[u][0] - positions[v][0]) ** 2 +
                     (positions[u][1] - positions[v][1]) ** 2) ** 0.5)  # ✅ Use Euclidean distance as fallback
            else:
                print(f"⚠ Warning: Edge ({u}, {v}) not found in graph!")
    return total_distance

# ---Visualize UDG ---
def visualize_udg(udg, udg_positions, path_nodes=None, current_node=None):
    """
    Visualize the Unit Disk Graph (UDG) with optional path highlighting.
    """
    plt.figure(figsize=(8, 6))
    nx.draw(
        udg,
        pos=udg_positions,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )

    if path_nodes:
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
        nx.draw_networkx_edges(udg, pos=udg_positions, edgelist=path_edges, edge_color="red", width=2)
        nx.draw_networkx_nodes(udg, pos=udg_positions, nodelist=path_nodes, node_color="orange", node_size=700)

    if current_node is not None:
        nx.draw_networkx_nodes(udg, pos=udg_positions, nodelist=[current_node], node_color="green", node_size=800)

    plt.title("Unit Disk Graph (UDG) with Packet Transmission")
    plt.show()
    time.sleep(1)  # Pause for animation effect


def visualize_udg_with_grid_layout(udg, udg_positions, grid_width, grid_height):
    """
    Visualize the UDG with the grid layout overlay.
    """
    plt.figure(figsize=(8, 6))

    # Draw the UDG graph
    nx.draw(
        udg,
        pos=udg_positions,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )

    # Overlay grid lines
    x_max = max(pos[0] for pos in udg_positions.values()) + DISTANCE
    y_max = max(pos[1] for pos in udg_positions.values()) + DISTANCE

    # Draw vertical grid lines
    for i in range(grid_width + 1):
        plt.axvline(x=i * DISTANCE, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Draw horizontal grid lines
    for j in range(grid_height + 1):
        plt.axhline(y=j * DISTANCE, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.title("UDG with Grid Layout Overlay")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(False)  # Disable default grid
    plt.show()

def map_udg_to_grid_static(udg, udg_positions):
    """
    Map UDG to a grid graph where each grid cell has a representative node.
    Only edges from the UDG are preserved in the grid graph.

    Args:
        udg: The original UDG (Undirected Graph).
        udg_positions: Positions of the UDG nodes.

    Returns:
        grid_graph: A simplified grid graph with representative nodes.
        grid_positions: Positions of the representative nodes for visualization.
        node_mapping: Mapping of UDG nodes to representative nodes.
    """
    grid_graph = nx.Graph()
    grid_positions = {}
    node_mapping = {}
    representative_nodes = {}  # Track representative nodes for each grid cell

    # Step 1: Map UDG nodes to grid cells and assign representative nodes
    for node, (x, y) in udg_positions.items():
        # Determine the grid cell based on x and y coordinates
        grid_x = x // DISTANCE
        grid_y = y // DISTANCE
        grid_cell = (grid_x, grid_y)

        # Assign a representative node for the grid cell
        if grid_cell not in representative_nodes:
            representative_nodes[grid_cell] = len(representative_nodes)  # Unique representative node ID
            grid_positions[grid_cell] = (grid_x * DISTANCE, -grid_y * DISTANCE)  # Position for visualization
            grid_graph.add_node(grid_cell)  # Add representative node to grid graph

        # Map UDG node to the grid cell's representative node
        node_mapping[node] = grid_cell

    # Step 2: Add edges based on UDG connections
    for u, v in udg.edges():
        # Map UDG edge to grid graph edge via their respective grid cells
        if u in node_mapping and v in node_mapping:
            cell_u = node_mapping[u]
            cell_v = node_mapping[v]
            if cell_u != cell_v:  # Avoid self-loops
                grid_graph.add_edge(cell_u, cell_v)

    return grid_graph, grid_positions, node_mapping


def split_columns_into_subcolumns(grid, grid_positions, grid_width, grid_height):
    """
    Split each column into sub-columns based on connectivity and gaps.
    Identify the uppermost representative nodes for each subcolumn.

    Args:
    - grid: The grid graph.
    - grid_positions: Positions of the grid nodes.
    - grid_width: Width of the grid.
    - grid_height: Height of the grid.

    Returns:
    - columns: A dictionary of subcolumns, where each key is a subcolumn index and value is a list of nodes.
    - representatives: A dictionary where keys are subcolumn indices and values are representative nodes.
    """
    columns = {}
    representatives = {}
    column_index = 0

    for col in range(grid_width):
        # Get all nodes in the current column
        column_nodes = [node for node in grid.nodes if node[0] == col]

        # Sort nodes by their y-coordinate (topmost first)
        column_nodes.sort(key=lambda node: grid_positions[node][1], reverse=True)

        # Initialize variables for splitting
        subcolumn = []
        previous_node = None

        for node in column_nodes:
            if previous_node is not None:
                # Check if there's no vertical edge between `previous_node` and `node`
                if not grid.has_edge(previous_node, node):
                    # No edge: Finalize the current subcolumn and start a new one
                    if subcolumn:
                        columns[column_index] = subcolumn
                        representatives[column_index] = subcolumn[0]  # Uppermost node as representative
                        column_index += 1
                    subcolumn = []  # Start a new subcolumn

            # Add the current node to the subcolumn
            subcolumn.append(node)
            previous_node = node

        # Add the final subcolumn for this column
        if subcolumn:
            columns[column_index] = subcolumn
            representatives[column_index] = subcolumn[0]  # Uppermost node as representative
            column_index += 1

    return columns, representatives

# ---Visualize Grid Graph ---
def visualize_grid_graph(grid, grid_positions, path=None, node_mapping=None, representatives=None):
    """
    Visualize the Grid Graph with representative portals labeled only on the uppermost nodes.

    Args:
    - grid: The grid graph to visualize.
    - grid_positions: The positions of the nodes in the grid.
    - path: Optional list of nodes representing the path to highlight.
    - node_mapping: Mapping of UDG nodes to grid nodes.
    - representatives: Mapping of representative nodes to subcolumns.
    """
    plt.figure(figsize=(10, 10))

    # Flip the y-axis to correct the orientation of the grid graph
    flipped_positions = {node: (pos[0], -pos[1]) for node, pos in grid_positions.items()}

    # Draw the grid graph structure with flipped positions
    nx.draw(
        grid,
        pos=flipped_positions,
        with_labels=False,
        node_color="lightgray",
        edge_color="gray",
        node_size=600,
    )

    # Label representative nodes and grid positions
    for grid_node, position in flipped_positions.items():
        # Show grid position
        plt.text(position[0], position[1], f"({grid_node[0]}, {grid_node[1]})",
                 fontsize=8, color="black", ha="center", fontweight="bold")

        # Label representative nodes as P1, P2, etc., for uppermost nodes only
        if representatives and grid_node in representatives.values():
            portal_index = list(representatives.values()).index(grid_node)
            plt.text(position[0], position[1] - 4, f"T{portal_index}",
                     fontsize=12, color="green", ha="center", fontweight="bold")

        # Label UDG nodes mapped to the grid node
        udg_labels = [
            f"U{udg_node}" for udg_node, grid_cell in node_mapping.items() if grid_cell == grid_node
        ]
        if udg_labels:
            udg_text = ", ".join(udg_labels)  # Separate multiple UDG labels with commas
            plt.text(position[0], position[1] - 2, udg_text, fontsize=8, color="blue", ha="center")

    # Highlight the path if provided
    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(grid, pos=flipped_positions, edgelist=path_edges, edge_color="red", width=2)
        nx.draw_networkx_nodes(grid, pos=flipped_positions, nodelist=path, node_color="orange", node_size=700)

    plt.title("Grid Graph with Representative Portals (Correct Orientation)")
    plt.show()

# ---Visualize Path in UDG ---
def visualize_udg_path(udg, udg_positions, path_nodes):
    """
    Visualize the path in the UDG.
    """
    plt.figure(figsize=(10, 8))
    nx.draw(
        udg,
        pos=udg_positions,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=500,
        font_size=10,
    )

    # Highlight the path
    path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
    nx.draw_networkx_edges(udg, pos=udg_positions, edgelist=path_edges, edge_color="red", width=2)
    nx.draw_networkx_nodes(udg, pos=udg_positions, nodelist=path_nodes, node_color="orange", node_size=700)

    plt.title("UDG with Highlighted Path")
    plt.show()


def build_and_visualize_transition_graph(columns, representatives, path, grid):
    """
    Build and visualize a portal tree based on the representatives of each column.
    Highlight the path between source and destination in the portal tree.

    Args:
    - columns: Dictionary of sub-columns with their nodes.
    - representatives: Dictionary mapping sub-columns to their representative nodes.
    - path: The shortest path in the grid graph.
    - grid: The grid graph.
    """
    portal_tree = nx.Graph()  # Use undirected graph for bidirectional connections
    portal_labels = {}

    # Add nodes to the portal tree based on representatives
    for col_index, rep in representatives.items():
        portal_tree.add_node(f"T{col_index}")
        portal_labels[f"T{col_index }"] = f"T{col_index}"

    # Add edges between representative nodes based on connectivity in the grid
    for col_index, nodes in columns.items():
        if col_index in representatives:
            rep_node = representatives[col_index]
            for neighbor in grid.neighbors(rep_node):
                neighbor_col = next(
                    (idx for idx, col_nodes in columns.items() if neighbor in col_nodes), None
                )
                if neighbor_col is not None and neighbor_col in representatives:
                    portal_tree.add_edge(f"T{col_index}", f"T{neighbor_col}")

    # Highlight the path in the portal tree
    path_representatives = []
    for node in path:
        for col_index, col_nodes in columns.items():
            if node in col_nodes and col_index in representatives:
                portal_label = f"T{col_index}"
                if portal_label not in path_representatives:
                    path_representatives.append(portal_label)

    highlighted_edges = []
    for i in range(len(path_representatives) - 1):
        start = path_representatives[i]
        end = path_representatives[i + 1]
        if portal_tree.has_edge(start, end):
            highlighted_edges.append((start, end))

    # Visualize the portal tree
    pos = nx.drawing.nx_agraph.graphviz_layout(portal_tree, prog="dot")
    plt.figure(figsize=(10, 7))
    nx.draw(
        portal_tree,
        pos=pos,
        with_labels=True,
        node_color="orange",
        edge_color="blue",
        node_size=900,
        font_size=10,
        font_color="white",
    )
    if highlighted_edges:
        nx.draw_networkx_edges(portal_tree, pos=pos, edgelist=highlighted_edges, edge_color="red", width=2)
    nx.draw_networkx_labels(portal_tree, pos=pos, labels=portal_labels, font_color="black")
    plt.title("Transition Graph with Highlighted Path")
    plt.show()


def map_grid_to_udg_path(grid_path, node_mapping, udg, src_udg, dest_udg):
    """
    Optimized function to map the grid path back to UDG while ensuring:
    - Only one best UDG node per mapped grid node is selected.
    - The selected node has the **lowest total edge weight**.
    - The path follows only **real UDG edges** (no artificial connections).
    - The shortest possible path is created with minimal weight.

    Args:
    - grid_path: Path found in the grid graph.
    - node_mapping: Mapping of UDG nodes to grid nodes.
    - udg: Original UDG graph with weighted edges.
    - src_udg: Source node in the UDG.
    - dest_udg: Destination node in the UDG.

    Returns:
    - optimized_udg_path: The shortest valid path mapped from grid to UDG.
    """
    if not grid_path:
        return []

    optimized_udg_path = []
    visited_nodes = set()  # Track already added nodes

    # Step 1: Precompute node priorities for all UDG nodes
    node_priority_cache = {}
    for udg_node in udg.nodes:
        total_weight = sum(udg[udg_node][neighbor].get('weight', 1) for neighbor in udg.neighbors(udg_node))
        direct_edges = sum(1 for neighbor in udg.neighbors(udg_node) if neighbor in node_mapping.values())
        node_priority_cache[udg_node] = total_weight / (direct_edges + 1)  # +1 to avoid division by zero

    # Step 2: Select a single representative node per grid cell (with the lowest edge weight)
    selected_udg_nodes = {}
    for grid_node in grid_path:
        udg_nodes = [udg_node for udg_node, mapped_grid in node_mapping.items() if mapped_grid == grid_node]

        if not udg_nodes:
            continue

        # Use cached priorities to select the best UDG node
        best_udg_node = min(udg_nodes, key=lambda x: node_priority_cache[x])
        selected_udg_nodes[grid_node] = best_udg_node

    # Step 3: Precompute all-pairs shortest paths in UDG
    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra_path(udg, weight='weight'))

    # Step 4: Convert the grid path into a valid UDG path using only real edges
    selected_nodes_list = list(selected_udg_nodes.values())

    for i in range(len(selected_nodes_list) - 1):
        current_node = selected_nodes_list[i]
        next_node = selected_nodes_list[i + 1]

        # Direct edge exists → Use it
        if udg.has_edge(current_node, next_node):
            if current_node not in visited_nodes:
                optimized_udg_path.append(current_node)
                visited_nodes.add(current_node)
            if next_node not in visited_nodes:
                optimized_udg_path.append(next_node)
                visited_nodes.add(next_node)

        else:
            # No direct edge → Use precomputed shortest path
            try:
                shortest_path = all_pairs_shortest_paths[current_node][next_node]
                for node in shortest_path:
                    if node not in visited_nodes:
                        optimized_udg_path.append(node)
                        visited_nodes.add(node)
            except KeyError:
                print(f"⚠ Warning: No valid transition found between {current_node} and {next_node}. Skipping.")

    # Ensure source and destination are included
    if optimized_udg_path and optimized_udg_path[0] != src_udg:
        optimized_udg_path.insert(0, src_udg)
    if optimized_udg_path and optimized_udg_path[-1] != dest_udg:
        optimized_udg_path.append(dest_udg)

    return optimized_udg_path

def visualize_udg_path_nodemapping(udg, udg_positions, grid_to_udg_path, node_mapping):
    """
    Visualize the UDG with:
    - Grouped circles around mapped nodes.
    - A highlighted chosen node from each group.
    - Blurred edges and nodes not involved in the path.
    """
    plt.figure(figsize=(10, 8))

    # Identify unique groups from the node mapping
    unique_groups = set(node_mapping.values())
    colormap = plt.colormaps['tab20']  # Updated for Matplotlib 3.7+
    colors = {group: colormap(i % 20) for i, group in enumerate(unique_groups)}

    # Identify all nodes involved in the grid path
    path_related_nodes = set(grid_to_udg_path)

    # Blur unrelated nodes and edges
    blurred_nodes = [node for node in udg.nodes if node not in path_related_nodes]
    blurred_edges = [edge for edge in udg.edges if not (edge[0] in path_related_nodes and edge[1] in path_related_nodes)]

    # Draw blurred nodes
    nx.draw_networkx_nodes(
        udg,
        pos=udg_positions,
        nodelist=blurred_nodes,
        node_color="lightgray",
        alpha=0.2,  # Transparent nodes
        node_size=100
    )

    # Draw blurred edges
    nx.draw_networkx_edges(
        udg,
        pos=udg_positions,
        edgelist=blurred_edges,
        edge_color="gray",
        alpha=0.1,  # Transparent edges
        style='dashed'
    )

    # Draw grouped circles and highlight selected nodes
    for group, color in colors.items():
        # Nodes in the current group
        group_nodes = [node for node, mapped in node_mapping.items() if mapped == group]

        # Nodes in the group that are part of the path
        selected_nodes_in_group = [node for node in group_nodes if node in grid_to_udg_path]

        if selected_nodes_in_group:
            # Highlight the selected node (chosen from the group)
            selected_node = selected_nodes_in_group[0]  # Only one selected node
            nx.draw_networkx_nodes(
                udg,
                pos=udg_positions,
                nodelist=[selected_node],
                node_color=[colors[group]],
                node_size=700,
                edgecolors="black",
                linewidths=2,
                label=f"Selected Node {selected_node}"
            )

            # Draw other group nodes with a light circle
            other_nodes_in_group = [node for node in group_nodes if node != selected_node]
            nx.draw_networkx_nodes(
                udg,
                pos=udg_positions,
                nodelist=other_nodes_in_group,
                node_color="white",
                edgecolors=color,
                linewidths=1,
                node_size=500,
                alpha=0.5
            )

            # Draw a grouping circle around all nodes in the group
            x_vals = [udg_positions[n][0] for n in group_nodes]
            y_vals = [udg_positions[n][1] for n in group_nodes]
            center_x, center_y = sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)
            radius = max([((udg_positions[n][0] - center_x) ** 2 + (udg_positions[n][1] - center_y) ** 2) ** 0.5 for n in group_nodes]) + 2

            circle = plt.Circle((center_x, center_y), radius, color=color, alpha=0.2, linestyle='dashed', fill=False)
            plt.gca().add_patch(circle)

            # Annotate all nodes with their labels
            for node in group_nodes:
                x, y = udg_positions[node]
                plt.text(x, y, str(node), fontsize=8, fontweight='bold', ha='center', va='center', color='black')

    # Draw dotted or curved lines between selected nodes in the path
    path_edges = [(grid_to_udg_path[i], grid_to_udg_path[i + 1]) for i in range(len(grid_to_udg_path) - 1)]
    for u, v in path_edges:
        x_values = [udg_positions[u][0], udg_positions[v][0]]
        y_values = [udg_positions[u][1], udg_positions[v][1]]
        plt.plot(x_values, y_values, linestyle='dotted', color='red', linewidth=2, alpha=0.8)

    plt.title("UDG Visualization with Grouped Nodes and Highlighted Path")
    plt.axis("off")
    plt.show()

# --- Packet Transmission Simulation ---
def simulate_packet_transmission(udg, udg_positions, path_nodes):
    """
    Simulate packet transmission along the shortest path.
    Highlights the current node as the packet moves.
    """
    print("\n || Packet Transmission Start:")
    for i, node in enumerate(path_nodes):
        print(f"🔵 Packet at Node {node}")

        # Visualize packet movement step-by-step
        visualize_udg(udg, udg_positions, path_nodes, current_node=node)

        if i < len(path_nodes) - 1:
            print(f"➡ Moving to Node {path_nodes[i + 1]}")
        else:
            print("✅ Packet Reached Destination!\n")
        time.sleep(1)

# --- Main Function ---
def main():
    # Step 1: Generate UDG and visualize
    num_nodes = 100
    grid_width = 8
    grid_height = 8
    udg, udg_positions = generate_udg(num_nodes)
    visualize_udg(udg, udg_positions)

    # Show edge count for UDG
    print(f"Number of edges in UDG: {udg.number_of_edges()}")

    # Step 2: Map UDG to grid graph
    visualize_udg_with_grid_layout(udg, udg_positions, grid_width, grid_height)
    grid, grid_positions, node_mapping = map_udg_to_grid_static(udg, udg_positions)


    # Show edge count for Grid Graph
    print(f"Number of edges in Grid Graph: {grid.number_of_edges()}")
    #Visualize the mapped grid graph
    columns, representatives = split_columns_into_subcolumns(grid, grid_positions, grid_width, grid_height)
    visualize_grid_graph(grid, grid_positions, path=None, node_mapping=node_mapping, representatives=representatives)

    # Prompt the user to enter source and destination nodes
    print("\nNode Mapping (UDG to Grid Nodes):")
    for udg_node, grid_node in node_mapping.items():
        print(f"UDG Node {udg_node} -> Grid Node {grid_node}")

    while True:
        try:
            src_udg = int(input("\nEnter the source node (UDG node number): "))
            dest_udg = int(input("Enter the destination node (UDG node number): "))

            if src_udg not in node_mapping or dest_udg not in node_mapping:
                print("Invalid nodes. Please enter valid UDG node numbers.")
                continue
            break
        except ValueError:
            print("Please enter valid integer values for source and destination.")

    # Map UDG source and destination to grid graph
    src_grid = node_mapping[src_udg]
    dest_grid = node_mapping[dest_udg]

    print(f"\nSource Grid Node: {src_grid}")
    print(f"Destination Grid Node: {dest_grid}")

    # Step 3: Find the shortest path in UDG
    try:
        udg_path = nx.shortest_path(udg, source=src_udg, target=dest_udg, weight='weight', method='dijkstra')
        udg_distance = calculate_total_distance(udg, udg_positions, udg_path)
        print(f"Shortest Path in UDG: {udg_path}")
        print(f"Number of edges in UDG Path: {len(udg_path) - 1}")
        print(f"Distance of Path in UDG: {udg_distance}")
        # Step 3: Simulate Packet Transmission
        visualize_udg_path(udg, udg_positions, udg_path)
        # simulate_packet_transmission(udg, udg_positions, udg_path)

    except nx.NetworkXNoPath:
        print("No path exists in UDG.")

    # Step 4: Find the shortest path in the grid graph
    try:
        grid_path = nx.shortest_path(grid, source=src_grid, target=dest_grid, weight='weight', method='dijkstra')
        print(f"Shortest Path in Grid Graph: {grid_path}")
        print(f"Number of edges in Grid Path: {len(grid_path) - 1}")
        visualize_grid_graph(grid, grid_positions, grid_path, node_mapping, representatives)
        # Transition tree
        build_and_visualize_transition_graph(columns, representatives, grid_path, grid)


        # Map the grid path back to the UDG path using only the uppermost node
        grid_to_udg_path = map_grid_to_udg_path(grid_path, node_mapping, udg, src_udg, dest_udg)
        mapped_udg_distance = calculate_total_distance(udg, udg_positions, grid_to_udg_path)
        print(f"Distance of Path in Grid to UDG: {mapped_udg_distance}")

        # Ensure dest is included in the final UDG path
        if dest_udg not in grid_to_udg_path:
            grid_to_udg_path.append(dest_udg)
        print(f"Path in UDG Derived from Grid Path: {grid_to_udg_path}")
        visualize_udg_path_nodemapping(udg, udg_positions, grid_to_udg_path, node_mapping)
        simulate_packet_transmission(udg, udg_positions, grid_to_udg_path)


    except ValueError as e:
        print(e)



if __name__ == "__main__":
    main()
