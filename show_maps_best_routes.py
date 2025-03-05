import os
import pandas as pd
import folium

variant = ""
folder = "results25"
taxi_locations = pd.read_csv("taxi_locations.csv")
variants = ['Standard_GA', 'GA-AM', 'GAAM-TS']

for variant in variants:
    best_route = pd.read_csv(f"{folder}/best_route_{variant}.csv", names=["LocationID"])
    route_locations = best_route.merge(taxi_locations, on="LocationID", how="left")
    route_coords = list(zip(route_locations["Latitude"], route_locations["Longitude"]))
    
    m = folium.Map(location=route_coords[0], zoom_start=12, tiles="CartoDB positron")
    for i, loc in enumerate(route_coords):
        folium.Marker(loc, 
                  popup=f'Stop {i+1}', 
                  icon=folium.DivIcon(html=f'<div style="font-size: 14pt; font-weight: bold; color: black;">{i+1}</div>')).add_to(m)
        folium.PolyLine(route_coords, color="red", weight=1.5, opacity=0.8).add_to(m)
        m.save(f"{folder}/best_route_map_{variant}.html")


