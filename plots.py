import pandas as pd
import matplotlib.pyplot as plt

# Cargar el dataset procesado con coordenadas
df_trips = pd.read_csv("nyc_taxi_trips_with_coordinates.csv")

# Convertir a tipo datetime las columnas de tiempo
df_trips["tpep_pickup_datetime"] = pd.to_datetime(df_trips["tpep_pickup_datetime"])
df_trips["tpep_dropoff_datetime"] = pd.to_datetime(df_trips["tpep_dropoff_datetime"])

# Crear una nueva columna con la duración del viaje en minutos
df_trips["trip_duration"] = (df_trips["tpep_dropoff_datetime"] - df_trips["tpep_pickup_datetime"]).dt.total_seconds() / 60

# Crear gráficos descriptivos

# """ # Definir nuevos bins para agrupar viajes mayores a 30 millas en un solo grupo
# bins = [0, 5, 10, 20, 30, df_trips["trip_distance"].max()]
# labels = ["0-5", "5-10", "10-20", "20-30", "30+"]

# df_trips["distance_group"] = pd.cut(df_trips["trip_distance"], bins=bins, labels=labels, include_lowest=True)
# distance_counts = df_trips["distance_group"].value_counts().sort_index()

# # Graficar la distribución de distancias agrupadas
# plt.figure(figsize=(8,5))
# bars = plt.bar(distance_counts.index, distance_counts, color="blue", edgecolor="black", alpha=0.7)
# # Agregar etiquetas de frecuencia en la parte superior de cada barra
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 200, int(yval), ha="center", va="bottom", fontsize=8, fontweight="bold")

# plt.xlabel("Trip Distance (miles)")
# plt.ylabel("Number of Trips")
# plt.title("Distribution of Trip Distances")
# plt.xticks()
# plt.grid(False)
# #plt.show()

# bins_duration = [0, 5, 10, 20, 30, 60, df_trips["trip_duration"].max()]
# labels_duration = ["0-5", "5-10", "10-20", "20-30", "30-60", "60+"]
# df_trips["duration_group"] = pd.cut(df_trips["trip_duration"], bins=bins_duration, labels=labels_duration, include_lowest=True)
# duration_counts = df_trips["duration_group"].value_counts().sort_index()

# plt.figure(figsize=(8,5))
# bars = plt.bar(duration_counts.index, duration_counts, color="green", edgecolor="black", alpha=0.7)

# # Agregar etiquetas de frecuencia en la parte superior de cada barra
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 100, int(yval), ha="center", va="bottom", fontsize=8, fontweight="bold")

# plt.xlabel("Trip Duration (minutes)")
# plt.ylabel("Number of Trips")
# plt.title("Distribution of Trip Durations")
# plt.xticks()
# plt.grid(False)
# plt.show() """

borough_counts = df_trips["Pickup_Borough"].value_counts()
plt.figure(figsize=(8,5))
bars = plt.bar(borough_counts.index, borough_counts, color="orange", edgecolor="black", alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, int(yval), ha="center", va="bottom", fontsize=8, fontweight="bold", color="black")

plt.xlabel("Borough")
plt.ylabel("Number of Trips")
plt.title("Number of Trips per Borough (Pickup)")
plt.xticks()
plt.grid(False)
plt.show()

