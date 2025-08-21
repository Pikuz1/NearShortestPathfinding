# NearShortestPathfinding
Grid based abstraction technique for near shortest pathfinding in Hybrid Communication System
This project implements grid-based abstraction and pathfinding algorithms to compute near-shortest paths in hybrid communication networks.
This project simulates **packet transmission** over a **Unit Disk Graph (UDG)** and its corresponding **grid-mapped graph**, including visualization of paths, node mapping, and portal-based transitions to replicate hybrid communicatioin system. 
It is part of a masters thesis.

## 📂 Project Structure

```text
├── NearShortestPathfinding/              # Main source code
│   └── pathfinding/
│       ├── evaluation/                   # Scripts to evaluate performance
│       │   ├── computation_time.py       # Evaluate runtime of algorithms
│       │   ├── edge_count.py             # Evaluate number of edges in paths
│       │   ├── packet_transmission.py    # Evaluate packet transmission results
│       │   ├── path_distance.py          # Evaluate distance of computed paths
│       │   └── pathfinding_success_rate.py # Evaluate success rate of algorithms
│       │
│       ├── grid_based_abstraction.py     # Core module for grid-to-UDG mapping and pathfinding
│       └── __init__.py                   # Marks `pathfinding` as a package
│
├── pathfindingenv/                       # Python virtual environment
│
└── README.md                             # Project documentation
```
## ⚙️ Setup

Follow these steps to prepare the environment and make the package importable.

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd NearShortestPathfinding
```

2. **Create & activate a virtual environment**

Linux / macOS
```bash
python3 -m venv pathfindingenv
source pathfindingenv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the script**
```bash
python grid_based_abstraction.py
```

---

## Overview

The simulation performs the following steps:

1. Generates a **Unit Disk Graph (UDG)** with random node positions.
2. Visualizes the UDG and overlays a **grid layout**.
3. Maps the UDG to a **simplified grid graph** using representative nodes.
4. Finds the **shortest path** in both the UDG and the grid graph.
5. Maps the grid path back to the UDG, ensuring **optimal selection of nodes**.
6. Visualizes the **packet transmission** along the shortest path.

---