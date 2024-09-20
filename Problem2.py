import networkx as nx
import random
import numpy as np
import time
import matplotlib.pyplot as plt
random.seed(2024)
def create_directed_graph(num_nodes, num_links):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    # Add links to the graph with random costs and capacities
    for _ in range(num_links):
        u = random.randint(0, num_nodes-2)
        v = random.randint(u+1, num_nodes-1)
        cost = random.randint(1, 100)
        capacity = random.randint(1, 100)
        G.add_edge(u, v, weight=cost, capacity=capacity, flow=0)

    return G

def create_source_sink_graph(G, num_source_nodes, num_sink_nodes, total_capacity):
    # Create source nodes
    for _ in range(num_source_nodes):
        u = random.randint(0, len(G.nodes)-1)
        G.add_edge('s', u, weight=0, capacity=total_capacity / (num_source_nodes * 2), flow=0)

    # Create sink nodes
    for _ in range(num_sink_nodes):
        v = random.randint(0, len(G.nodes)-1)
        G.add_edge(v, 't', weight=0, capacity=total_capacity / (num_sink_nodes * 2), flow=0)

    return G

def create_graph():
    num_nodes = 50
    num_links = 900
    num_source_nodes = 5
    num_sink_nodes = 5
    total_capacity = 100

    # Create a directed graph with random costs and capacities
    G = create_directed_graph(num_nodes, num_links)

    # Add source and sink nodes to the graph
    G = create_source_sink_graph(G, num_source_nodes, num_sink_nodes, total_capacity)

    return G

def print_edges_with_flow(G):
    # Iterate through the edges of the graph
    for u, v, data in G.edges(data=True):
        # Check if the flow is greater than 0
        if data['flow'] > 0:
            # Print the edge with its flow
            print(f"{u} -> {v} with flow {data['flow']}")

def initialize_single_source(source, graph):
    distance = {vertex: float('inf') for vertex in graph.nodes()}
    distance[source] = 0

    pred = {vertex: None for vertex in graph.nodes()}

    return distance, pred

def relax(edge, distance, pred, graph):
    u = edge[0]
    v = edge[1]
    w = graph[u][v]['weight']
    if distance[v] > distance[u] + w:
        distance[v] = distance[u] + w
        pred[v] = u

    return distance, pred

def bellman_ford(graph, source):
    distance, pred = initialize_single_source(source, graph)

    for _ in range(len(graph.nodes()) - 1):
        for edge in graph.edges(data=True):
            distance, pred = relax(edge, distance, pred, graph)

    return distance, pred

def find_shortest_path(graph, source, sink):
    distance, pred = bellman_ford(graph, source)

    try:
        shortest_path = []
        path = sink

        while path is not None:
            shortest_path.append(path)
            path = pred[path]

        shortest_path.reverse()
        distance_to_sink = distance[sink]

        return shortest_path, distance_to_sink

    except nx.exception.NetworkXNoPath:
        return None, None

def graph_update(graph, path):
    INF = float('inf')
    flow = INF
    if path is not None:
        # Find the flow on the path
        for u, v in zip(path[:-1], path[1:]):
            flow = min(flow, graph[u][v]['capacity'])

        # Update the graph
        for u, v in zip(path[:-1], path[1:]):
            graph[u][v]['capacity'] -= flow
            graph[u][v]['flow'] = flow
            G.add_edge(v, u, weight=graph[u][v]['weight'], capacity=flow, flow=0)
            if graph[u][v]['capacity'] == 0:
                graph.remove_edge(u, v)
        return graph, flow

    else:
        return None, None


def copy_graph(G):
    # Create an empty directed graph
    G_copy = nx.DiGraph()

    # Add nodes to the new graph
    for node in G.nodes:
        G_copy.add_node(node)

    # Add links to the new graph with the same costs, capacities, and flows
    for u, v, attr_dict in G.edges(data=True):
        G_copy.add_edge(u, v, weight=attr_dict['weight'], capacity=attr_dict['capacity'], flow=attr_dict['flow'])

    return G_copy

def Sucessive_shortest_path(graph, source, sink):
    flow = 0
    add_flow = 0
    residual_graph = copy_graph(graph)
    new_graph = copy_graph(graph)
    while True:
        shortest_path, _ = find_shortest_path(residual_graph, source, sink)
        if len(shortest_path) > 1 or flow == 100:
            residual_graph, add_flow = graph_update(residual_graph, shortest_path)
            flow += add_flow
        else:
            break
    for u, v in new_graph.edges:
        if residual_graph.has_edge(u, v):
            new_graph[u][v]['flow'] = residual_graph[u][v]['flow']
            new_graph[u][v]['capacity'] -= residual_graph[u][v]['flow']
        else:
            new_graph[u][v]['flow'] = new_graph[u][v]['capacity']
            new_graph[u][v]['capacity'] = 0
    return new_graph
# Create the graph and display its structure
G = create_graph()

G = Sucessive_shortest_path(G, 's', 't')

# Function to run the algorithm, record runtime, and write to file
def Successive_Shortest_path(num_iterations, algorithm_name):
    runtimes = []

    for i in range(num_iterations):
        G = create_graph()

        start_time = time.time()
        G = Sucessive_shortest_path(G, 's', 't')
        print("Successive_Shortest_path: ", i, "Finished")
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)

        # Write the runtime to a file
        with open(f"{algorithm_name}_Running_time.txt", "a") as file:
            file.write(f"Iteration {i + 1}: {runtime} seconds\n")

    return runtimes

def Simplex(num_iterations, algorithm_name):
    runtimes = []

    for i in range(num_iterations):
        G = create_graph()

        start_time = time.time()
        G = nx.network_simplex(G, 's', 't')
        print("Simplex: ", i, "Finished")

        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)

        # Write the runtime to a file
        with open(f"{algorithm_name}_Running_time.txt", "a") as file:
            file.write(f"Iteration {i + 1}: {runtime} seconds\n")

    return runtimes

def Capacity_Scaled(num_iterations, algorithm_name):
    runtimes = []

    for i in range(num_iterations):
        G = create_graph()

        start_time = time.time()
        G = nx.capacity_scaling(G, 's', 't')
        print("Capacity_Scaled: ", i, "Finished")
        end_time = time.time()

        runtime = end_time - start_time
        runtimes.append(runtime)

        # Write the runtime to a file
        with open(f"{algorithm_name}_Running_time.txt", "a") as file:
            file.write(f"Iteration {i + 1}: {runtime} seconds\n")

    return runtimes

def plot_comparison(algorithm_names, avg_runtimes, total_runtimes):
    x = np.arange(len(algorithm_names))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots()
    rects2 = ax.bar(x + width/2, total_runtimes, width, label='Total')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Total Runtimes by Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_names)
    ax.legend()

    fig.tight_layout()

    plt.show()
def plot_runtimes(runtimes, algorithm_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(runtimes) + 1), runtimes, marker='o')
    plt.title(f'Runtime per Iteration for {algorithm_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.show()
def write_overall_performance(algorithm_name, all_runtimes):
    avg_runtime = sum(all_runtimes) / num_iterations
    total_runtime = sum(all_runtimes)

    # Write to "Overall.txt"
    with open("Overall.txt", "a") as file:
        file.write(f"{algorithm_name}:\n")
        file.write(f"Avg: {avg_runtime} seconds\n")
        file.write(f"Total: {total_runtime} seconds\n\n")

# Run the algorithms, record runtimes, and write to files
num_iterations = 1000
algorithm_names = ["Sucessive_shortest_path", "network_simplex", "capacity_scaling"]
avg_runtimes = []
total_runtimes = []
# Run and write for Sucessive_shortest_path
all_runtimes = Successive_Shortest_path(num_iterations, algorithm_names[0])
avg_runtimes.append(sum(all_runtimes) / num_iterations)
total_runtimes.append(sum(all_runtimes))
plot_runtimes(all_runtimes, algorithm_names[0])

# Run and write for network_simplex
another_runtimes = Simplex(num_iterations, algorithm_names[1])
avg_runtimes.append(sum(another_runtimes) / num_iterations)
total_runtimes.append(sum(another_runtimes))
plot_runtimes(another_runtimes, algorithm_names[1])

# Run and write for capacity_scaling
capacity_runtimes = Capacity_Scaled(num_iterations, algorithm_names[2])
avg_runtimes.append(sum(capacity_runtimes) / num_iterations)
total_runtimes.append(sum(capacity_runtimes))
plot_runtimes(capacity_runtimes, algorithm_names[2])

# Plot comparison
plot_comparison(algorithm_names, avg_runtimes, total_runtimes)