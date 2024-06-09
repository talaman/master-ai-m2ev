# ----data----


import json

# Load JSON data
with open('201132-0-museos.json') as f:
    data = json.load(f)

# Extract relevant information
museums = []
for item in data['@graph'][13:17]:
    museum = {
        'id': item['id'],
        'title': item['title'],
        'latitude': item['location']['latitude'],
        'longitude': item['location']['longitude'],
        'address': item['address']['street-address'],
        'schedule': item['organization']['schedule']
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
num_days = 2  # Adjust based on the number of days

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
