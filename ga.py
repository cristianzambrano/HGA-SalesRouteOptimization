import pandas as pd

def get_selected_locations(df, num_locations=15):
    unique_locations = df["LocationIDO"].unique()
    return unique_locations

def fitness(route, df, gamma=1.0):
    total_distance = sum(df[(df["LocationIDO"] == route[i]) & (df["LocationIDD"] == route[i+1])]["Distance_km"].values[0]
                          for i in range(len(route) - 1))
    total_time = sum(df[(df["LocationIDO"] == route[i]) & (df["LocationIDD"] == route[i+1])]["Travel_Time_min"].values[0]
                      for i in range(len(route) - 1))
    return  (total_distance + gamma * total_time)


df = pd.read_csv("distance_time_matrix.csv")
ruta = get_selected_locations(df)
print(ruta)
fitness_value = fitness(ruta, df)
print(fitness_value)
