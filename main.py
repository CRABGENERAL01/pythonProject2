import pandas as pd
import numpy as np
import random
import networkx as nx
import itertools
import time
import matplotlib.pyplot as plt

# Load distance matrix from data
distance_matrix = [
    [1, 606, 704, 429, 131, 539, 992, 317, 102, 796, 353, 606, 959, 988, 752, 310, 575, 1486, 1519, 557],
    [611, 1, 666, 195, 583, 100, 1097, 281, 597, 867, 179, 65.1, 808, 1352, 214, 494, 122, 1007, 1382, 239],
    [715, 661, 1, 589, 592, 572, 419, 475, 702, 188, 486, 618, 268, 677, 609, 399, 562, 781, 842, 438],
    [434, 195, 614, 1, 406, 220, 991, 105, 420, 759, 199, 253, 867, 1189, 403, 317, 291, 1075, 1440, 303],
    [131, 579, 574, 402, 1, 512, 863, 290, 118, 667, 426, 579, 829, 860, 757, 179, 548, 1303, 1391, 530],
    [538, 97, 574, 214, 510, 1, 997, 230, 525, 767, 86.2, 72.8, 708, 1255, 210, 422, 80.2, 908, 1281, 141],
    [1097, 1043, 425, 970, 941, 953, 1, 857, 1084, 237, 868, 1004, 419, 470, 1036, 781, 949, 626, 540, 825],
    [316, 276, 494, 98.8, 28, 230, 871, 1, 303, 639, 144, 297, 750, 1071, 434, 197, 266, 1021, 1323, 248],
    [101, 594, 706, 416, 119, 526, 980, 304, 1, 784, 441, 593, 961, 967, 740, 298, 563, 1317, 1507, 545],
    [863, 809, 191, 736, 719, 719, 234, 623, 850, 1, 634, 770, 215, 492, 802, 547, 715, 848, 752, 591],
    [452, 177, 475, 202, 424, 86.7, 851, 144, 439, 620, 1, 154, 669, 1109, 291, 336, 123, 878, 1242, 106],
    [601, 62.8, 612, 273, 573, 66.8, 1035, 293, 587, 805, 148, 1, 746, 1293, 151, 484, 59.5, 946, 1319, 179],
    [997, 810, 268, 871, 853, 709, 416, 757, 984, 277, 669, 752, 1, 653, 815, 681, 697, 518, 578, 573],
    [990, 1288, 651, 1197, 862, 1180, 477, 1083, 977, 463, 1094, 1231, 625, 1, 1262, 833, 1175, 1090, 1005, 1051],
    [746, 209, 642, 393, 718, 212, 1066, 438, 733, 835, 264, 147, 776, 1323, 1, 630, 154, 877, 1290, 261],
    [310, 488, 410, 311, 179, 421, 777, 197, 297, 546, 336, 488, 656, 833, 629, 1, 457, 1170, 1229, 440],
    [581, 122, 561, 287, 553, 81, 984, 273, 568, 754, 163, 59.8, 695, 1242, 158, 465, 1, 896, 1268, 128],
    [1328, 1010, 796, 1078, 1300, 909, 624, 1020, 1315, 849, 877, 952, 519, 1091, 880, 1184, 896, 1, 386, 776],
    [1623, 1386, 844, 1496, 1479, 1285, 538, 1382, 1655, 763, 1245, 1328, 579, 1005, 1294, 1306, 1273, 387, 1, 1149],
    [556, 238, 437, 306, 528, 138, 860, 248, 543, 630, 105, 181, 571, 1118, 265, 440, 125, 776, 1144, 1]
]

def nearest_neighbor_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    unvisited_cities = set(range(1, num_cities))
    tour = [0]

    current_city = 0
    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])
        tour.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    tour.append(0)
    return tour

def calculate_total_distance(tour, distance_matrix):
    total_distance = 0
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[i + 1]
        total_distance += distance_matrix[current_city][next_city]
    return total_distance

def pso_tsp(distance_matrix, num_particles=30, max_iterations=100, c1=2.0, c2=2.0):
    num_cities = len(distance_matrix)
    global_best_tour = None
    global_best_distance = float('inf')
    particles = []

    for _ in range(num_particles):
        tour = random.sample(range(num_cities), num_cities)
        tour.append(tour[0])  # Ensure the tour is a closed loop
        particle = {
            'tour': tour,
            'best_tour': tour,
            'best_distance': calculate_total_distance(tour, distance_matrix),
        }
        particles.append(particle)

    for _ in range(max_iterations):
        for particle in particles:
            tour = particle['tour']
            for _ in range(num_cities):
                if random.random() < 0.5:
                    # Perform local search
                    i, j = random.sample(range(1, num_cities), 2)
                    tour[i], tour[j] = tour[j], tour[i]
                else:
                    # Perform global search
                    new_tour = list(tour)
                    i, j = random.sample(range(1, num_cities), 2)
                    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                    new_distance = calculate_total_distance(new_tour, distance_matrix)
                    if new_distance < particle['best_distance']:
                        particle['best_tour'] = new_tour
                        particle['best_distance'] = new_distance
                        if new_distance < global_best_distance:
                            global_best_tour = new_tour
                            global_best_distance = new_distance
            particle['tour'] = particle['best_tour']

    return global_best_tour

def aco_tsp(distance_matrix, num_ants=20, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    num_cities = len(distance_matrix)
    pheromone = np.ones((num_cities, num_cities))
    best_tour = None
    best_distance = float('inf')

    for _ in range(num_iterations):
        ant_tours = []
        ant_distances = []

        for _ in range(num_ants):
            visited = set()
            tour = [0]
            current_city = 0

            while len(visited) < num_cities - 1:
                unvisited = set(range(num_cities)) - visited
                probabilities = []

                for city in unvisited:
                    pheromone_factor = pheromone[current_city][city] ** alpha
                    distance_factor = (1.0 / distance_matrix[current_city][city]) ** beta
                    total_factor = pheromone_factor * distance_factor
                    probabilities.append(total_factor)

                probabilities = [p / sum(probabilities) for p in probabilities]
                next_city = np.random.choice(list(unvisited), p=probabilities)
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city

            tour.append(0)  # Return to the starting city
            ant_tours.append(tour)
            ant_distances.append(calculate_total_distance(tour, distance_matrix))

            if ant_distances[-1] < best_distance:
                best_tour = tour
                best_distance = ant_distances[-1]

        # Update pheromone levels
        pheromone *= (1.0 - evaporation_rate)
        for i in range(num_ants):
            tour = ant_tours[i]
            distance = ant_distances[i]
            for j in range(len(tour) - 1):
                pheromone[tour[j]][tour[j + 1]] += (1.0 / distance)

    return best_tour

def held_karp_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    all_cities = set(range(num_cities))
    memo = {}

    def tsp_dp(mask, current_city):
        if mask == all_cities:
            return distance_matrix[current_city][0]
        if (tuple(mask), current_city) in memo:
            return memo[(tuple(mask), current_city)]

        min_distance = float('inf')
        for next_city in range(num_cities):
            if next_city != current_city and next_city not in mask:
                new_mask = mask | {next_city}
                subproblem_distance = distance_matrix[current_city][next_city] + tsp_dp(new_mask, next_city)
                min_distance = min(min_distance, subproblem_distance)

        memo[(tuple(mask), current_city)] = min_distance
        return min_distance

    best_tour = []
    mask = {0}
    current_city = 0

    for _ in range(1, num_cities):
        neighbors = [city for city in range(num_cities) if city != current_city]
        next_city = min(neighbors, key=lambda city: distance_matrix[current_city][city])
        best_tour.append(next_city)
        mask.add(next_city)
        current_city = next_city

    best_tour.insert(0, 0)
    best_tour.append(0)
    best_distance = tsp_dp(mask, current_city) + distance_matrix[current_city][0]

    return best_tour


def plot_tour(tour, distance_matrix):
    num_cities = len(distance_matrix)
    G = nx.Graph()

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            city1 = tour[i]
            city2 = tour[j]
            distance = distance_matrix[city1][city2]
            G.add_edge(city1, city2, weight=distance)

    pos = nx.spring_layout(G)
    labels = {city: city for city in G.nodes()}
    edge_labels = {(city1, city2): distance_matrix[city1][city2] for city1, city2 in G.edges()}
    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8, font_weight='bold')
    plt.title('TSP Tour')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    num_cities = len(distance_matrix)

    # Nearest Neighbor Algorithm
    nn_start_time = time.time()
    nn_solution = nearest_neighbor_tsp(distance_matrix)
    nn_execution_time = time.time() - nn_start_time
    nn_length = calculate_total_distance(nn_solution, distance_matrix)

    # Particle Swarm Optimization (PSO) Algorithm
    pso_start_time = time.time()
    pso_solution = pso_tsp(distance_matrix, num_particles=30, max_iterations=100, c1=2.0, c2=2.0)
    pso_execution_time = time.time() - pso_start_time
    pso_length = calculate_total_distance(pso_solution, distance_matrix)

    # Ant Colony Optimization (ACO) Algorithm
    aco_start_time = time.time()
    aco_solution = aco_tsp(distance_matrix, num_ants=20, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5)
    aco_execution_time = time.time() - aco_start_time
    aco_length = calculate_total_distance(aco_solution, distance_matrix)

    # Held-Karp Algorithm (Optimal)
    hk_start_time = time.time()
    hk_solution = held_karp_tsp(distance_matrix)
    hk_execution_time = time.time() - hk_start_time
    hk_length = calculate_total_distance(hk_solution, distance_matrix)

    # Print results
    print("Nearest Neighbor Algorithm:")
    print("Tour:", nn_solution)
    print("Tour Length:", nn_length)
    print("Execution Time:", nn_execution_time, "seconds")

    print("\nParticle Swarm Optimization (PSO) Algorithm:")
    print("Tour:", pso_solution)
    print("Tour Length:", pso_length)
    print("Execution Time:", pso_execution_time, "seconds")

    print("\nAnt Colony Optimization (ACO) Algorithm:")
    print("Tour:", aco_solution)
    print("Tour Length:", aco_length)
    print("Execution Time:", aco_execution_time, "seconds")

    print("\nHeld-Karp Algorithm (Optimal):")
    print("Tour:", hk_solution)
    print("Tour Length:", hk_length)
    print("Execution Time:", hk_execution_time, "seconds")

    # Plot the best tour found by each algorithm
    plot_tour(nn_solution, distance_matrix)
    plot_tour(pso_solution, distance_matrix)
    plot_tour(aco_solution, distance_matrix)
    plot_tour(hk_solution, distance_matrix)