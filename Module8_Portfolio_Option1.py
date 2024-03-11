import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq
import numpy as np
from time import time
from tqdm import tqdm

class GeneticAlgorithm:
    def __init__(self, graph, population_size, generations, mutation_rate):
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    # Method to generate a random path between two nodes, start and end
    def random_path(self, start, end):
        # Initialize the path with the starting node
        path = [start]
        # Set the current node to the starting node
        current_node = start

        # Continue until the current node reaches the destination node
        while current_node != end:
            # Get the neighbors of the current node
            neighbors = list(self.graph.neighbors(current_node))

            # Check if there are no neighbors
            if not neighbors:
                # If there are no neighbors, backtrack by removing the last node from the path
                path.pop()
                # If the path becomes empty, break the loop
                if not path:
                    break
                # Update the current node to the previous node in the path
                current_node = path[-1]
            else:
                # Choose a random neighbor as the next node
                next_node = random.choice(neighbors)
                # Append the chosen next node to the path
                path.append(next_node)
                # Update the current node to the chosen next node
                current_node = next_node

        # Return the generated path
        return path

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Create a random path using the random_path method starting at the node labeled 0 to the last node
            start_node, end_node = 0, len(self.graph.nodes) - 1
            candidate_solution = self.random_path(start_node, end_node)
            population.append(candidate_solution)

        return population

    def evaluate_fitness(self, individual):
        # Initialize total_distance to 0
        total_distance = 0

        # Iterate over the indices of the individual
        for i in range(len(individual) - 1):
            # Get the current and next node in the individual
            current_node = individual[i]
            next_node = individual[i + 1]

            # Check if the edge exists
            if self.graph.has_edge(current_node, next_node):
                # Access the weight of the edge between the current and next node in the graph
                edge_weight = self.graph[current_node][next_node]['weight']
                # Add the edge weight to the total_distance
                total_distance += edge_weight
            else:
                # Handle the case where the edge does not exist (e.g., set a penalty)
                total_distance += 1000  # You can adjust the penalty as needed

        # Fitness is the negative of the total distance in the path
        fitness = -total_distance

        return fitness

    def crossover(self, parent1, parent2):
        common_nodes = set(parent1) & set(parent2)  # Find common nodes in both parents

        if not common_nodes:
            # If there are no common nodes, return one of the parents
            return random.choice([parent1, parent2])

        split = random.choice(list(common_nodes))

        parent1_genes = parent1[:parent1.index(split) + 1]
        parent2_genes = parent2[parent2.index(split) + 1:]

        child = parent1_genes + parent2_genes

        return child

    def mutate(self, individual):
        # Ensure start is smaller than end
        start, end = sorted(random.sample(range(len(individual)), 2))

        # Find a sub-path for mutation
        sub_path = self.random_path(start, end)

        # Apply the mutation to the individual
        individual[start:end + 1] = sub_path

        return individual

    def genetic_algorithm(self):
        # Initialize the population of individuals
        population = self.initialize_population()

        # Iterate through generations
        for generation in range(self.generations):
            # Evaluate the fitness of each individual in the population
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]

            # Select the top 50% based on fitness for reproduction
            selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
            selected_population = [population[selected_index] for selected_index in selected_indices]

            # Perform crossover and mutation to create the next generation
            new_population = []
            while len(new_population) < self.population_size:
                # Select two parents randomly from the top-performing individuals
                parent1, parent2 = random.sample(selected_population, 2)

                # Create a child by crossover of the two parents
                child = self.crossover(parent1, parent2)

                # Apply mutation to the child with a certain probability
                if random.uniform(0, 1) < self.mutation_rate:  # % chance of mutation
                    child = self.mutate(child)

                # Add the child to the new population
                new_population.append(child)

            # Update the population with the new generation
            population = new_population

        # Select the best individual from the final population
        best_individual = max(population, key=self.evaluate_fitness)
        best_fitness = self.evaluate_fitness(best_individual)

        # Return the best individual and its fitness
        return best_individual, -best_fitness


# Graph generator

def generate_weighted_connected_graph(nodes, min_edges=2, max_edges=3, weight_range=(1, 10)):
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(nodes))

    # Connect each node to at least one other node
    for i in range(nodes):
        # Ensure each node has at least one outgoing edge
        num_edges = max(1, random.randint(min_edges, max_edges))
        edges = random.sample([j for j in range(nodes) if j != i], num_edges)

        for edge in edges:
            weight = random.randint(*weight_range)
            G.add_edge(i, edge, weight=weight)

    return G

# Alternative Algorithms

def brute_force_shortest_path(graph, start, end, current_path=[], current_weight=0, shortest_path=None, shortest_weight=float('inf')):
    # Add the current node to the path
    current_path = current_path + [start]

    # Check if the current node is the destination
    if start == end:
        # Update the shortest path and weight if the current path is shorter
        if current_weight < shortest_weight:
            shortest_path = current_path
            shortest_weight = current_weight
    else:
        # Explore neighbors
        for neighbor in graph.neighbors(start):
            # Ensure not to revisit nodes in the current path to avoid cycles
            if neighbor not in current_path:
                # Calculate the weight of the edge from the current node to the neighbor
                edge_weight = graph[start][neighbor]['weight']
                # Recursively explore the path with the neighbor as the next node
                shortest_path, shortest_weight = brute_force_shortest_path(
                    graph, neighbor, end,
                    current_path, current_weight + edge_weight,
                    shortest_path, shortest_weight)

    return shortest_path, shortest_weight

def dijkstra_shortest_path(graph, start, end):
    # Initialize priority queue with a tuple (distance, node)
    priority_queue = [(0, start)]

    # Initialize distances with infinity for all nodes
    distances = {node: float('infinity') for node in graph.nodes}

    # Distance from the start node to itself is 0
    distances[start] = 0

    # Dijkstra's algorithm main loop
    while priority_queue:
        # Get the node with the minimum distance from the priority queue
        current_distance, current_node = heapq.heappop(priority_queue)

        # Skip if the distance to the current node is greater than the known distance
        if current_distance > distances[current_node]:
            continue

        # Explore neighbors of the current node
        for neighbor in graph.neighbors(current_node):
            # Calculate the total distance to the neighbor through the current node
            distance = current_distance + graph[current_node][neighbor]['weight']

            # If a shorter path is found, update the distance and push to the priority queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    # Reconstruct the shortest path from start to end
    path = [end]
    current_node = end
    while current_node != start:
        # Find neighbors with distances equal to the current distance minus the edge weight
        neighbors = [neighbor for neighbor in graph.neighbors(current_node) if distances[neighbor] == distances[current_node] - graph[current_node][neighbor]['weight']]
        current_node = neighbors[0]
        path.insert(0, current_node)

    return path

# Display

def display_weighted_graph(graph, shortest_path):
    pos = nx.spring_layout(graph)
    node_colors = ['skyblue' if node in shortest_path else 'white' for node in graph.nodes()]
    edge_labels = {(i, j): graph[i][j]['weight'] for i, j in graph.edges()}

    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=200, node_color=node_colors, font_color='black', font_size=8, edge_color='black')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(f"Shortest Path: {shortest_path}")
    plt.show()

# Test Example Using a Single Graph

node_count = 10
weighted_connected_graph = generate_weighted_connected_graph(node_count)


# Dijkstra's Algorithm:
shortest_path_dijkstra = dijkstra_shortest_path(weighted_connected_graph, 0, node_count - 1)
shortest_weight_dijkstra = sum(weighted_connected_graph[shortest_path_dijkstra[i]][shortest_path_dijkstra[i+1]]['weight'] for i in range(len(shortest_path_dijkstra)-1))
print("Dijkstra's Algorithm:")
print("Shortest Path:", shortest_path_dijkstra)
print("Shortest Weight:", shortest_weight_dijkstra)

# Genetic Approach
ga = GeneticAlgorithm(weighted_connected_graph, population_size=50, generations=100, mutation_rate=.001)
shortest_path_genetic, best_fitness = ga.genetic_algorithm()
genetic_weight = -best_fitness
print("\nGenetic Algorithm:")
print("Shortest Path:", shortest_path_genetic)
print("Shortest Weight:", -genetic_weight)

# Brute Force by calculating all possible paths
start_node, end_node = 0, node_count - 1
shortest_path_brute_force, shortest_weight_brute_force = brute_force_shortest_path(weighted_connected_graph, start_node, end_node)
print("\nBrute Force:")
print("Shortest Path:", shortest_path_brute_force)
print("Shortest Weight:", shortest_weight_brute_force)


# Display Dijkstra's Algorithm result
plt.figure()
plt.title("Dijkstra's Algorithm")
display_weighted_graph(weighted_connected_graph, shortest_path_dijkstra)
plt.show()

# Display Genetic Algorithm result
plt.figure()
plt.title("Genetic Algorithm")
display_weighted_graph(weighted_connected_graph, shortest_path_genetic)
plt.show()

# Display Brute Force result
plt.figure()
plt.title("Brute Force")
display_weighted_graph(weighted_connected_graph, shortest_path_brute_force)
plt.show()

# Performance Test by Creating Multiple Graphs of Increasing Node Size
node_sizes = list(range(10, 50))
dijkstra_execution_times = []
ga_execution_times = []

for size in tqdm(node_sizes, desc="Performance Test"):
    weighted_connected_graph = generate_weighted_connected_graph(size)

    # Dijkstra's Algorithm
    start_time_dijkstra = time()
    dijkstra_shortest_path(weighted_connected_graph, 0, size - 1)
    execution_time_dijkstra = time() - start_time_dijkstra
    dijkstra_execution_times.append(execution_time_dijkstra)

    # Genetic Algorithm
    start_time_ga = time()
    ga = GeneticAlgorithm(weighted_connected_graph, population_size=50, generations=100, mutation_rate=.001)
    ga.genetic_algorithm()
    execution_time_ga = time() - start_time_ga
    ga_execution_times.append(execution_time_ga)

# Plot the results
plt.plot(node_sizes, dijkstra_execution_times, marker='o', label='Dijkstra')
plt.plot(node_sizes, ga_execution_times, marker='o', label='Genetic Algorithm')
plt.xlabel('Node Size')
plt.ylabel('Execution Time (s)')
plt.title('Performance Test: Dijkstra vs Genetic Algorithm')
plt.legend()
plt.grid(True)
plt.show()
