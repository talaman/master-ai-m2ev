import json
from sklearn.cluster import KMeans
import numpy as np
from math import radians, cos, sin, asin, sqrt
import googlemaps
import os

# ----data----

# Load JSON data
with open('museos.json') as f:
    data = json.load(f)

# Extract relevant information
museums = []
for item in data['@graph']:
    museum = {
        'id': item['id'],
        'title': item['title'],
        'latitude': item['location']['latitude'],
        'longitude': item['location']['longitude'],
        'address': item['address']['street-address'],
        'schedule': item['organization']['schedule'],
        'visit_time': item['visit_time']  
    }
    museums.append(museum)

# ----clustering----

# Extract coordinates
coordinates = np.array([(museum['latitude'], museum['longitude']) for museum in museums])

# Number of clusters (days)
num_days = int(input("Enter the number of days: "))  # Prompt for the number of days

# Perform clustering
kmeans = KMeans(n_clusters=num_days, random_state=0).fit(coordinates)
labels = kmeans.labels_

# Group museums by clusters
clusters = {i: [] for i in range(num_days)}
for label, museum in zip(labels, museums):
    clusters[label].append(museum)

# Reassign museums to clusters to balance the number of items
while True:
    max_cluster_size = max(len(museums) for museums in clusters.values())
    min_cluster_size = min(len(museums) for museums in clusters.values())
    if max_cluster_size - min_cluster_size <= 1:
        break
    max_cluster = max(clusters, key=lambda x: len(clusters[x]))
    min_cluster = min(clusters, key=lambda x: len(clusters[x]))
    museum_to_move = clusters[max_cluster].pop()
    clusters[min_cluster].append(museum_to_move)

# ----Reduce the number of museums to visit per day based on distance and visit time----

def calculate_distance(museum1, museum2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(radians, [museum1['latitude'], museum1['longitude']])
    lat2, lon2 = map(radians, [museum2['latitude'], museum2['longitude']])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def get_closest_museums(museums, max_distance, max_time):
    closest_museums = []
    total_time = 0
    for museum in museums:
        if all(calculate_distance(museum, other) <= max_distance for other in closest_museums):
            if total_time + museum['visit_time'] <= max_time:
                closest_museums.append(museum)
                total_time += museum['visit_time']
    return closest_museums

# Prompt for the maximum distance and time per day
max_distance = float(input("Enter the maximum distance per day (in kilometers): "))
max_time = int(input("Enter the maximum time per day (in hours): "))

# Reduce the number of museums to visit per day based on distance and visit time
for day, museums in clusters.items():
    clusters[day] = get_closest_museums(museums, max_distance=max_distance, max_time=max_time)

# ----Optimize route----

# Initialize Google Maps client
gmaps = googlemaps.Client(key='AIzaSyARvB2VPh9VAMXq6AmiXNbvhnf24YZ2ybk')  # Replace 'YOUR_API_KEY' with your actual API key

def get_route(museums):
    waypoints = [f"{museum['latitude']},{museum['longitude']}" for museum in museums]
    directions = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        direction = gmaps.directions(
            origin=start,
            destination=end,
            mode="transit"
        )
        # aggregate the from and to museums data to the direction
        direction[0]['from'] = museums[i]
        direction[0]['to'] = museums[i + 1]        
        directions.extend(direction)
    return directions

# Get optimized routes for each cluster
for day, museums in clusters.items():
    route = get_route(museums)
    print('\n**********************') 
    print(f"Day {day + 1}:")
    print('**********************')
    for i, leg in enumerate(route):
        from_museum = leg['from']['title']
        to_museum = leg['to']['title']
        print('\n--------------------------')
        print(f"From: {from_museum} to {to_museum}")
        print('--------------------------')
        for step in leg['legs'][0]['steps']:
            print(step['html_instructions'])
