import pandas as pd
import geopandas as gpd

df_trips = pd.read_parquet("data/data.parquet", engine="pyarrow")
taxi_zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")
taxi_zones = gpd.read_file("data/taxi_zones.shp")

print("Sistema de coordenadas original:", taxi_zones.crs)
#Convirtiendo coordenadas a WGS 84 (Lat/Lon)
taxi_zones = taxi_zones.to_crs(epsg=4326)
# Unir shapefile con el CSV de zonas
taxi_zones = taxi_zones.merge(taxi_zone_lookup, on="LocationID", how="left")

taxi_zones["centroid"] = taxi_zones.geometry.centroid
taxi_zones["Latitude"] = taxi_zones.centroid.y
taxi_zones["Longitude"] = taxi_zones.centroid.x

# Seleccionar solo las columnas necesarias
taxi_zones_coords = taxi_zones[["LocationID", "Borough", "Zone", "Latitude", "Longitude"]]


# Unir coordenadas de la zona de recogida (Pickup)
df_trips = df_trips.merge(taxi_zones_coords, left_on="PULocationID", right_on="LocationID", how="left")
df_trips = df_trips.rename(columns={"Latitude": "Pickup_Lat", "Longitude": "Pickup_Lon", "Borough": "Pickup_Borough", "Zone": "Pickup_Zone"})

# Unir coordenadas de la zona de destino (Dropoff)
df_trips = df_trips.merge(taxi_zones_coords, left_on="DOLocationID", right_on="LocationID", how="left", suffixes=("_PU", "_DO"))
df_trips = df_trips.rename(columns={"Latitude": "Dropoff_Lat", "Longitude": "Dropoff_Lon", "Borough": "Dropoff_Borough", "Zone": "Dropoff_Zone"})

df_trips.head(30000).to_csv("nyc_taxi_trips_with_coordinates.csv", index=False)
