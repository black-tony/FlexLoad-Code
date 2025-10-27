import random

# 假设服务器数量
NUM_SERVERS = 10

# 初始化服务器的负载情况
server_loads = [random.randint(0, 100) for _ in range(NUM_SERVERS)]
server_loads_change = []

# 客户端的服务需求
client_demand = 10

# 服务器的最大容量
MAX_CAPACITY = 100

# 遗传算法参数
POPULATION_SIZE = 20
GENERATIONS = 100
MUTATION_RATE = 0.1

def initialize_population():
    # 随机生成初始种群
    return [i % NUM_SERVERS for i in range(POPULATION_SIZE)]

def fitness(individual):
    # 适应度函数，目标是负载均衡且不超过最大容量
    server_id = individual
    if server_loads[server_id] + client_demand > MAX_CAPACITY:
        return 0  # 超过容量的个体适应度为0
    load_difference = abs(server_loads[server_id] + client_demand - sum(server_loads) / NUM_SERVERS)
    return 1 / (1 + load_difference)

def select(population):
    # 选择适应度高的个体 
    weights = [fitness(ind) for ind in population]
    if sum(weights) == 0:
        return random.choices(population, k=2)
    return random.choices(population, weights=weights, k=2)

def crossover(parent1, parent2):
    # 单点交叉
    return random.choice([parent1, parent2])

def mutate(individual):
    # 随机变异
    if random.random() < MUTATION_RATE:
        return random.randint(0, NUM_SERVERS - 1)
    return individual

def genetic_algorithm():
    population = initialize_population()
    for _ in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1, parent2 = select(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    
    # 返回最优解
    best_individual = max(population, key=fitness)
    # print(f"Best server selected: {best_individual} with load {server_loads[best_individual]}")
    return best_individual

def simulate_task_scheduling(num_tasks, init_server_loads=None):
    global server_loads
    global server_loads_change
    if init_server_loads is not None:
        server_loads = init_server_loads.copy()
    server_loads_change = []
    for _ in range(num_tasks):
        selected_server = genetic_algorithm()
        if server_loads[selected_server] + client_demand <= MAX_CAPACITY:
            server_loads[selected_server] += client_demand
            server_loads_change.append(server_loads.copy())
        else:
            return

# 调度N次任务
simulate_task_scheduling(10)
