import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_selected_locations(df, num_locations):
    unique_locations = df["LocationIDO"].unique()
    return random.sample(list(unique_locations), num_locations)

def generate_population(locations, pop_size):
    return [random.sample(locations, len(locations)) for _ in range(pop_size)]

def fitness(route, df, gamma):
    total_distance = 0
    total_time = 0
    n = len(route)
    
    for i in range(n):
        origin = route[i]
        dest = route[(i+1)%n]
        
        match = df[(df["LocationIDO"] == origin) & (df["LocationIDD"] == dest)]
        if match.empty:
            match = df[(df["LocationIDO"] == dest) & (df["LocationIDD"] == origin)]
            if match.empty:
                raise ValueError(f"Route not found: {origin}->{dest}")
        
        total_distance += match["Distance_km"].values[0]
        total_time += match["Travel_Time_min"].values[0]
    
    return 1 / (total_distance + gamma * total_time)

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None]*size
    child[start:end+1] = parent1[start:end+1]
    
    ptr = (end + 1) % size
    for gene in parent2:
        if gene not in child:
            child[ptr] = gene
            ptr = (ptr + 1) % size
    return child

def tournament_selection(population, scores, k=3):
    selected = []
    for _ in range(len(population)):
        contestants = random.sample(list(zip(population, scores)), k)
        winner = max(contestants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def adaptive_mutation(route, diversity):
    p_min = 0.1
    p_max = 0.5
    max_diversity = 0.5  # Valor de referencia para normalización
    
    adaptive_rate = p_min + (p_max - p_min) * (1 - diversity/max_diversity)
    
    if random.random() < adaptive_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def tabu_search(route, df, gamma, max_iter, tabu_size):
    best_route = route.copy()
    best_fitness = fitness(best_route, df, gamma)
    tabu_list = []
    
    for i in range(max_iter):
        neighbors = []
        for i in range(len(route)):
            for j in range(i+1, len(route)):
                new_route = route.copy()
                new_route[i], new_route[j] = new_route[j], new_route[i]
                if new_route not in tabu_list:
                    neighbors.append(new_route)
        
        if not neighbors:
            break
        
        neighbor_fitness = [(n, fitness(n, df, gamma)) for n in neighbors]
        neighbor_fitness.sort(key=lambda x: x[1], reverse=True)
        
        
        current_best = neighbor_fitness[0]
        
        if current_best[1] > best_fitness:
            best_route = current_best[0]
            best_fitness = current_best[1]
        
        tabu_list.append(current_best[0])
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        
    return best_route

def calculate_diversity(population):
    unique_routes = set(tuple(route) for route in population)
    return len(unique_routes) / len(population)

def plot_ga_progress(history):
    # Extraer datos del historial
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    diversity = [h['diversity'] for h in history]

    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Gráfico de Fitness
    ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
    ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=1.5)
    ax1.set_title('Fitness')
    ax1.set_ylabel('Fitness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    
    ax2.plot(generations, diversity, 'g-', label='Diversity', linewidth=2)
    ax2.set_title('Populatio Diversity')
    ax2.set_ylabel('Diversity')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(generations, best_fitness, 'm-', linewidth=2)
    ax3.set_title('Progreso del Mejor Fitness')
    ax3.set_xlabel('Generación')
    ax3.set_ylabel('Mejor Fitness')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def hybrid_genetic_algorithm(df, number_of_locations, generations, pop_size, gamma, 
                            elitism_ratio, tabu_iter, tabu_size, tabu_perctoimprove):
    
    locations = get_selected_locations(df, number_of_locations)
    population = generate_population(locations, pop_size)
    best_global = None
    best_fitness_global = -np.inf
    history = []
 
    
    for gen in range(generations):

        fitness_scores = [fitness(route, df, gamma) for route in population]
        
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness_global:
            best_global = population[current_best_idx]
            best_fitness_global = fitness_scores[current_best_idx]
        
        diversity = calculate_diversity(population)
        
        history.append({
            'generation': gen,
            'best_fitness': best_fitness_global,
            'avg_fitness': np.mean(fitness_scores),
            'diversity': diversity
        })
        
        selected = tournament_selection(population, fitness_scores)
        
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i+1)%pop_size]

            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            
            child1 = adaptive_mutation(child1, diversity) 
            child2 = adaptive_mutation(child2, diversity)
            
            offspring.extend([child1, child2])
        
        elite_size = int(elitism_ratio * pop_size)
        elite = sorted(zip(population, fitness_scores), 
                      key=lambda x: -x[1])[:elite_size]
        elite = [x[0] for x in elite]
        
        population = elite + offspring[:pop_size - elite_size]
        
        tabu_candidates = population[:int(tabu_perctoimprove * pop_size)]
        improved = [tabu_search(candidate, df, gamma, tabu_iter, tabu_size) 
                   for candidate in tabu_candidates]
        population[:len(improved)] = improved

        print("Generation ", gen , " best_fitness_global ",best_fitness_global)
    
    return best_global, best_fitness_global, history


NUM_LOCATIONS = 15
POPULATION_SIZE = 100
NUMBER_GENERATIONS = 50
MUTATION_RATE = 0.15
Gamma = 0.8
TABU_ITERACTIONS = 10
TABU_MAXSIZEOFLIST = 20
TABU_POPULATIONTOIMPROVE = 0

# Load Distance and Time Matrix  of Locations to find best route
distance_time_matrix = pd.read_csv('distance_time_matrix.csv')

best_route, best_score, history = hybrid_genetic_algorithm(
    distance_time_matrix,
    number_of_locations=NUM_LOCATIONS,
    generations=NUMBER_GENERATIONS,
    pop_size=POPULATION_SIZE,
    gamma=Gamma,
    elitism_ratio=0.15,
    tabu_iter=TABU_ITERACTIONS,
    tabu_size = TABU_MAXSIZEOFLIST,
    tabu_perctoimprove = TABU_POPULATIONTOIMPROVE
)

print(f"Mejor ruta encontrada: {best_route}")
print(f"Fitness: {best_score:.6f}")
total_distance = sum(df[df['LocationIDO'] == best_route[i]][df['LocationIDD'] == best_route[(i+1)%len(best_route)]]['Distance_km'].values[0]
                   for i in range(len(best_route)))
print(f"Distancia total: {total_distance:.2f} km")

plot_ga_progress(history)