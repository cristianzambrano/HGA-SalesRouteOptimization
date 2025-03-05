import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib

file_path = "nyc_taxi_trips_with_coordinates.csv"
df_trips = pd.read_csv(file_path)
df_trips["tpep_pickup_datetime"] = pd.to_datetime(df_trips["tpep_pickup_datetime"])
df_trips["tpep_dropoff_datetime"] = pd.to_datetime(df_trips["tpep_dropoff_datetime"])
df_trips["trip_duration"] = (df_trips["tpep_dropoff_datetime"] - df_trips["tpep_pickup_datetime"]).dt.total_seconds() / 60

#Preprocesamiento de Datos
#Nulos
df_trips.dropna(subset=["Pickup_Lat", "Pickup_Lon", "Dropoff_Lat", "Dropoff_Lon", "trip_duration"], inplace=True)
# Filtrar viajes con duración mayor a 0 y menor a 120 minutos para evitar outliers extremos
df_trips = df_trips[(df_trips["trip_duration"] > 0) & (df_trips["trip_duration"] <= 120)]
# Extraer características temporales
df_trips["hour_of_day"] = df_trips["tpep_pickup_datetime"].dt.hour
df_trips["day_of_week"] = df_trips["tpep_pickup_datetime"].dt.dayofweek

# Seleccionar variables relevantes para el modelo
features = ["Pickup_Lat", "Pickup_Lon", "Dropoff_Lat", "Dropoff_Lon", "trip_distance", "hour_of_day", "day_of_week"]
target = "trip_duration"

scaler = MinMaxScaler()
df_trips[features] = scaler.fit_transform(df_trips[features])
joblib.dump(scaler, "model/minmax_scaler.pkl")

# Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(df_trips[features], df_trips[target], test_size=0.2, random_state=42)

# Convertir a formato 3D para LSTM (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Construir el modelo LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1) 
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_mae = model.evaluate(X_test, y_test)

# Guardar el modelo entrenado
model.save("model/lstm_travel_time_model.h5")

# Mostrar resultados
test_loss, test_mae


# Guardar historial en un archivo JSON
with open("model/lstm_training_history.json", "w") as f:
    json.dump(history.history, f)

history_dict = history.history

plt.figure(figsize=(8,5))
plt.plot(history_dict["loss"], label="Training Loss", color="blue", linewidth=2)
plt.plot(history_dict["val_loss"], label="Validation Loss", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history_dict["mae"], label="Training MAE", color="blue", linewidth=2)
plt.plot(history_dict["val_mae"], label="Validation MAE", color="red", linestyle="dashed", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Training and Validation MAE")
plt.legend()
plt.show()

y_pred = model.predict(X_test)

df_results = pd.DataFrame({
    "Actual Duration (min)": y_test.values,
    "Predicted Duration (min)": y_pred.flatten()
})
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
plt.xlabel("Actual Trip Duration (min)")
plt.ylabel("Predicted Trip Duration (min)")
plt.title("Actual vs. Predicted Trip Duration")
plt.axline((0, 0), slope=1, color="black", linestyle="dashed")  # Línea de referencia perfecta
plt.show()