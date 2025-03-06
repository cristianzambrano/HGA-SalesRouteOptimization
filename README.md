# HGA-SalesRouteOptimization

##  Projct Description

This repository contains the source code and experimental results for the article **"Advanced Sales Route Optimization through Enhanced Genetic Algorithms and Real-Time Navigation Systems"**. This work improves the original approach proposed by Zambrano-Vega et al. (2019) by integrating a **Hybrid Genetic Algorithm (HGA)** combined with **Adaptive Mutation, Tabu Search, and Machine Learning techniques** to optimize sales routes efficiently.

The proposed method allows real-time route reoptimization based on traffic conditions, weather, and other external factors. Experimental results demonstrate a **20% reduction in total traveled distance and a 15% improvement in delivery time** compared to the original model.

## Features
- **Hybrid Genetic Algorithm (HGA)** for optimized sales route planning.
- **Adaptive Mutation Operator** to maintain diversity and avoid premature convergence.
- **Tabu Search** to refine solutions and escape local optima.
- **LSTM-based Machine Learning Model** for travel time prediction considering real-time traffic and weather conditions.
- **Integration with Google Maps API** for real-time navigation.


### Running the Hybrid Genetic Algorithm
1. **Preprocess the data**:
   ```bash
   python scripts/procesamiento.py
   ```
2. **Generate the optimization problem**:
   ```bash
   python scripts/problem_gen.py
   ```
3. **Train the LSTM model** (Optional, pretrained model available):
   ```bash
   python scripts/traininLSMT.py
   ```
4. **Run the Hybrid Genetic Algorithm**:
   ```bash
   python scripts/Genetic_algorithm.py
   ```

## 📊 Experimental Results
### Performance Evaluation
| Method | Total Distance (km) | Delivery Time (min) | Computation Time (s) | Diversity Score |
|--------|--------------------|------------------|----------------|----------------|
| Traditional GA | 500 | 480 | 120 | Low |
| GA + Adaptive Mutation | 450 (↓10%) | 432 (↓10%) | 130 | Medium |
| GA + Adaptive Mutation + Tabu Search | 427 (↓15%) | 408 (↓15%) | 150 | High |

The **Hybrid Genetic Algorithm with Adaptive Mutation and Tabu Search** demonstrates a significant improvement in route efficiency.
-
