import numpy as np
import random


def _map_global_node_to_master_local(chosen_node: int, s_grid):
    """将全局节点索引映射到 (master_id, local_node)。cloud 返回 None。
    cloud 索引 = sum(len(cpu_list) for 每组)
    """
    total_nodes = 0
    lengths = []
    for g in s_grid:
        n = len(g[2])
        lengths.append(n)
        total_nodes += n
    cloud_index = total_nodes
    if chosen_node == cloud_index:
        return None
    acc = 0
    for midx, n in enumerate(lengths):
        if chosen_node < acc + n:
            return midx, chosen_node - acc
        acc += n
    return None


def get_generic_act(s_grid, ava_node, context):
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    crossover_rate = 0.8

    def initialize_population():
        return [random.choice(ava_node[i]) for i in range(len(ava_node))]

    def fitness(individual):
        load_balance_score = 0
        utilization_score = 0

        # Calculate load balance score and utilization score
        for i, node in enumerate(individual):
            try:
                mapped = _map_global_node_to_master_local(node, s_grid)
                if mapped is not None:  # If not cloud computing
                    master, local_node = mapped
                    cpu_usage = s_grid[master][2][local_node][0] / s_grid[master][2][local_node][1]
                    mem_usage = s_grid[master][3][local_node][0] / s_grid[master][3][local_node][1]
                    utilization_score += cpu_usage + mem_usage
                    load_balance_score += abs(cpu_usage - mem_usage)
            except IndexError:
                print(f"IndexError for node {node} in individual {individual}")
                print(s_grid[master][2], s_grid[master][3])

        # Minimize load imbalance and maximize utilization
        return utilization_score - load_balance_score

    def selection(population, fitness_scores):
        try:
            total_fitness = sum(fitness_scores)
            probabilities = [f / total_fitness for f in fitness_scores]
        except ZeroDivisionError:
            probabilities = [1 / len(population)] * len(population)
        return population[np.random.choice(len(population), p=probabilities)]

    def crossover(parent1, parent2):
        if random.random() < crossover_rate:
            try:
                point = random.randint(1, len(parent1) - 2)
            except ValueError:
                return parent1, parent2
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1, parent2

    def mutate(individual):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = random.choice(ava_node[i])
        return individual

    # Initialize population
    population = [initialize_population() for _ in range(population_size)]

    for _ in range(generations):
        # Calculate fitness scores
        fitness_scores = [fitness(individual) for individual in population]

        # Create new population
        new_population = []
        for _ in range(population_size // 2):
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutate(offspring1))
            new_population.append(mutate(offspring2))

        population = new_population

    # Select the best individual
    best_individual = max(population, key=fitness)
    return best_individual, None, None, None, None, None, None
