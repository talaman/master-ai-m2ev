# ----data----


import json

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

# Print extracted data
# for museum in museums:
#     print(museum)


# ----clustering----


from sklearn.cluster import KMeans
import numpy as np

# Extract coordinates
coordinates = np.array([(museum['latitude'], museum['longitude']) for museum in museums])

# Number of clusters (days)
num_days = 3  # Adjust based on the number of days

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

# Print clusters
for day, museums in clusters.items():
    print(f"Day {day + 1}:")
    for museum in museums:
        print(f"  - {museum['title']} at {museum['address']}")

# reduce the number of museums to visit per day based on the visit_time (an int that 1 is an hour), the maximum hours per day is 8\

# for day, museums in clusters.items():
#     total_hours = 0
#     for museum in museums:
#         total_hours += museum['visit_time']
#     while total_hours > 4:
#         max_museum = max(museums, key=lambda x: x['visit_time'])
#         museums.remove(max_museum)
#         total_hours -= max_museum['visit_time']

from math import radians, cos, sin, asin, sqrt

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

# Reduce the number of museums to visit per day based on distance and visit time
for day, museums in clusters.items():
    clusters[day] = get_closest_museums(museums, max_distance=10, max_time=5)  # Replace 10 and 8 with your desired maximum distance and time

for day, museums in clusters.items():
    print(f"Day recalculated {day + 1}:")
    for museum in museums:
        print(f"  - {museum['title']} at {museum['address']}")

# ----Optimize route----

import googlemaps
import os
from sklearn.cluster import KMeans
import numpy as np

# Initialize Google Maps client
gmaps = googlemaps.Client(key='AIzaSyARvB2VPh9VAMXq6AmiXNbvhnf24YZ2ybk')

def get_route(museums):
    waypoints = [f"{museum['latitude']},{museum['longitude']}" for museum in museums]
    start = waypoints[0]
    end = waypoints[-1]
    waypoints = waypoints[1:-1]
    
    directions = gmaps.directions(
        origin=start,
        destination=end,
        waypoints=waypoints,
        mode="transit"
    )
    return directions

# Get optimized routes for each cluster
for day, museums in clusters.items():
    print(f"Day {day + 1}:")
    route = get_route(museums)
    for step in route[0]['legs'][0]['steps']:
        print(step['html_instructions'])


# TODO: Add the code to display the route on a map
# TODO: fix more than 2 museums per day
# TODO: priority

