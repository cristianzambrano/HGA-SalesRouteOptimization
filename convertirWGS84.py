import geopandas as gpd

taxi_zones = gpd.read_file("data/taxi_zones.shp")
print("Sistema de coordenadas actual:", taxi_zones.crs)
taxi_zones = taxi_zones.to_crs(epsg=4326) # Convertir a WGS 84 (Lat/Lon)
taxi_zones["centroid"] = taxi_zones["geometry"].centroid
taxi_zones["Latitude"] = taxi_zones["centroid"].y
taxi_zones["Longitude"] = taxi_zones["centroid"].x
print(taxi_zones[["LocationID", "Latitude", "Longitude"]].head())