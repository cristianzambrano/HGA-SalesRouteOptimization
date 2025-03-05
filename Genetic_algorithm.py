import time
import random
import pandas as pd
import numpy as np

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
    
    return 1 / (total_distance + gamma * total_time), total_distance, total_time

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
    max_diversity = 0.5  # 
    
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

def fixed_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def hybric_genetic_algorithm(df, locations, generations, pop_size, gamma,
               elitism_ratio, use_adaptive_mutation, use_tabu_search,
               fixed_mutation_rate, tabu_iter, tabu_size, tabu_perctoimprove):
    
    population = generate_population(locations, pop_size)
    history = []
    
    best_global = None
    best_fitness_global = -np.inf
    best_distance = np.inf
    best_time = np.inf

    for gen in range(generations):
        fitness_data = [fitness(route, df, gamma) for route in population]
        fitness_scores = [x[0] for x in fitness_data]
        distances = [x[1] for x in fitness_data]
        times = [x[2] for x in fitness_data]
        
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness_global:
            best_global = population[current_best_idx]
            best_fitness_global = fitness_scores[current_best_idx]
            best_distance = distances[current_best_idx]
            best_time = times[current_best_idx]

        diversity = calculate_diversity(population)
        
        history.append({
            'generation': gen,
            'best_fitness': best_fitness_global,
            'best_distance': best_distance,
            'best_time': best_time,
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
            
            if use_adaptive_mutation:
                child1 = adaptive_mutation(child1, diversity)
                child2 = adaptive_mutation(child2, diversity)
            elif fixed_mutation_rate is not None:
                child1 = fixed_mutation(child1, fixed_mutation_rate)
                child2 = fixed_mutation(child2, fixed_mutation_rate)
            
            offspring.extend([child1, child2])
        
        elite_size = int(elitism_ratio * pop_size)
        elite = sorted(zip(population, fitness_scores), key=lambda x: -x[1])[:elite_size]
        elite = [x[0] for x in elite]
        
        population = elite + offspring[:pop_size - elite_size]
        
        if use_tabu_search:
            tabu_candidates = population[:int(tabu_perctoimprove * pop_size)]
            improved = [tabu_search(candidate, df, gamma, tabu_iter, tabu_size) 
                       for candidate in tabu_candidates]
            population[:len(improved)] = improved
        
        print("Generation ", gen)

    return best_global, best_fitness_global, history

def run_algorithm_variant(df, locations, variant, params):
    start_time = time.time()
    
    config = {
        'use_adaptive_mutation': variant != 'Standard GA',
        'use_tabu_search': variant == 'GAAM-TS',
        'fixed_mutation_rate': 0.15 if variant == 'Standard GA' else None
    }
    
    best_route, best_score, history = hybric_genetic_algorithm(
        df, 
        locations,
        **{**params, **config}
    )
    
    computational_time = time.time() - start_time
    final_metrics = {
        'Total Distance (km)': history[-1]['best_distance'],
        'Total Delivery Time (min)': history[-1]['best_time'],
        'Computational Time (s)': computational_time,
        'Final Diversity': history[-1]['diversity']
    }
    
    return history, final_metrics, best_route

def compare_variants(folder, df, locations, params):
    variants = ['Standard GA', 'GA-AM', 'GAAM-TS']
    results = {}
    report_data = []

    for variant in variants:
        print(f"\nRunning {variant}...")
        history, metrics, best_route = run_algorithm_variant(df, locations, variant, params)
        report_data.append({
            'Variant': variant,
            'Distance (km)': metrics['Total Distance (km)'],
            'Time (min)': metrics['Total Delivery Time (min)'],
            'Comp. Time (s)': metrics['Computational Time (s)'],
            'Diversity': metrics['Final Diversity']
        })

        history_df = pd.DataFrame(history)
        history_df['Variant'] = variant
        filename = f"{folder}/history_{variant.replace(' ', '_')}.csv"
        history_df.to_csv(filename, index=False)

        best_route_df = pd.DataFrame(best_route)
        filename = f"{folder}/best_route_{variant.replace(' ', '_')}.csv"
        best_route_df.to_csv(filename, index=False)

        results[variant] = {
            'history': history,
            'metrics': metrics
        }

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(f"{folder}/comparative_report.csv", index=False)     
        
    return results, report_df



common_params = {
    'generations': 100,
    'pop_size': 100,
    'gamma': 0.8,
    'elitism_ratio': 0.15,
    'tabu_iter': 10,
    'tabu_size': 15,
    'tabu_perctoimprove': 0.01
}

distance_time_matrix = pd.read_csv('distance_time_matrix25.csv')
locations  = get_selected_locations(distance_time_matrix, 25)
results, report = compare_variants('results25', distance_time_matrix, locations, common_params)

