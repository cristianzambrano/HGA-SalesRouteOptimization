import pandas as pd
import numpy as np
import tensorflow as tf
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
import joblib

df_locations = pd.read_csv("taxi_locations.csv")

num_points = 35
hour_of_day = np.random.randint(0, 24)
day_of_week = np.random.randint(0, 7)
df_selected = df_locations.sample(n=num_points, random_state=42)

lstm_model = tf.keras.models.load_model("model/lstm_travel_time_model.h5", compile=False)
lstm_model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])

scaler = joblib.load("model/minmax_scaler.pkl")

def lstm_predict_time(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, trip_distance, hour_of_day, day_of_week):
    input_data = np.array([[pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, trip_distance, hour_of_day, day_of_week]])
    input_data_scaled = scaler.transform(input_data)
    input_data_reshaped = np.reshape(input_data_scaled, (1, 1, 7))  # Adding time step dimension
    predicted_time = lstm_model.predict(input_data_reshaped)
    return float(predicted_time[0, 0])  # Convert output tensor to scalar value

data = []
for i, row_i in df_selected.iterrows():
    for j, row_j in df_selected.iterrows():
        if i != j:  
            distance = geodesic((row_i["Latitude"], row_i["Longitude"]), (row_j["Latitude"], row_j["Longitude"])).km
            travel_time = lstm_predict_time(row_i["Latitude"], row_i["Longitude"], row_j["Latitude"], row_j["Longitude"], distance, hour_of_day, day_of_week)

            data.append([row_i["LocationID"], row_j["LocationID"], distance, travel_time])

df_problem = pd.DataFrame(data, columns=["LocationIDO", "LocationIDD", "Distance_km", "Travel_Time_min"])

df_problem.to_csv("distance_time_matrix35.csv", index=False)

print("Archivo 'distance_time_matrix35.csv' creado con éxito.")
