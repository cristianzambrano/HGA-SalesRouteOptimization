import pandas as pd
import geopandas as gpd

# Cargar los datos
taxi_zone_lookup = pd.read_csv("data/taxi_zone_lookup.csv")  # Archivo con nombres y ID de zonas
taxi_zones = gpd.read_file("data/taxi_zones.shp")  # Shapefile con coordenadas

# Verificar sistema de coordenadas y convertir a lat/lon si es necesario
if taxi_zones.crs is None or taxi_zones.crs.to_epsg() != 4326:
    taxi_zones = taxi_zones.to_crs(epsg=4326)  # Convertir a WGS 84 si es necesario

# Extraer centroides de cada zona
taxi_zones["centroid"] = taxi_zones.geometry.centroid
taxi_zones["Latitude"] = taxi_zones.centroid.y
taxi_zones["Longitude"] = taxi_zones.centroid.x

# Seleccionar columnas de interés
taxi_zones = taxi_zones[["LocationID", "Latitude", "Longitude"]]

# Unir con taxi_zone_lookup para obtener nombres de zonas
df_locations = taxi_zone_lookup.merge(taxi_zones, on="LocationID", how="left")

# Guardar en CSV
df_locations.to_csv("taxi_locations.csv", index=False)

print("Archivo 'taxi_locations.csv' creado con éxito.")
