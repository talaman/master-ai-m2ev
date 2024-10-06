"""
----Descripción----

Aplicación de planificación de visitas a museos en Madrid.

Esta aplicación permite planificar la visita a museos en Madrid en función de la distancia y el tiempo de visita.
Utiliza técnicas de clustering para agrupar los museos en diferentes días y optimiza la ruta de visita utilizando la API de Google Maps.

El objetivo de esta aplicación es ayudar a los usuarios a organizar su itinerario de visitas a museos de manera eficiente y optimizada.

----Estructura de archivos----

- app.py: Código fuente de la aplicación.
- requirements.txt: Archivo con las dependencias necesarias para ejecutar la aplicación.
- museos.json: Archivo JSON con la información de los museos en Madrid descargado del sitio web del Ministerio para la Transformación digital, en el enlace https://datos.gob.es/en/catalogo/l01280796-museos-de-la-ciudad-de-madrid, se ha añadido información adicional como el tiempo estimado de visita y la prioridad de visita.
- run-example.txt: Archivo con un ejemplo de ejecución de la aplicación.

Datos agregados a museos.json:
- visit_time: Tiempo estimado de visita en horas, por defecto 2, a ajustar en función de la realidad.
- priority: Prioridad de visita, donde 1 es la menor y 10 la mayor, a ajustar en función de las preferencias del usuario.

----Requisitos previos----

- Esto fue probado con Python 3.10, pero debería funcionar con otras versiones de Python 3.
- Instalar las librerías necesarias ejecutando el siguiente comando en la terminal:
  'pip install numpy scikit-learn googlemaps'
  O si tienes un archivo requirements.txt puedes instalar todas las dependencias con:
  'pip install -r requirements.txt'

----Instrucciones de uso----

1. Ejecutar el script 'app.py' en la terminal:
  'python app.py'
2. Ingresar el número de días que se desea planificar la visita a los museos.
3. Ingresar la distancia máxima en kilómetros que se desea recorrer por día.
4. Ingresar el tiempo máximo en horas que se desea dedicar a la visita de museos por día.

La aplicación mostrará la ruta optimizada para cada día de visita, indicando los museos a visitar y las instrucciones para llegar de un museo a otro.

----Valor diferenciador----

El valor diferenciador de esta aplicación es la combinación de técnicas de clustering para agrupar los museos en diferentes días y la optimización de la ruta de visita utilizando la API de Google Maps.
Esto permite a los usuarios planificar su itinerario de visitas de manera eficiente y optimizada, maximizando el tiempo y minimizando la distancia recorrida.

----

Autor: Daniel Antonio Tala de Dompierre de Chaufepie
"""



# ----Librerías----

import json
import numpy as np
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import KMeans
import googlemaps

print("\nBienvenido a la aplicación de planificación de visitas a museos en Madrid.\n")

# ----data----

print("Cargando datos de los museos en Madrid...\n")

# Cargar datos JSON
with open('museos.json') as f:
    data = json.load(f)

# Extraer información relevante
museums = []
for item in data['@graph']:
    museum = {
        'id': item['id'],
        'title': item['title'],
        'latitude': item['location']['latitude'],
        'longitude': item['location']['longitude'],
        'address': item['address']['street-address'],
        'schedule': item['organization']['schedule'],
        'visit_time': item['visit_time'],
        'priority': item['priority']
    }
    museums.append(museum)

# ----clustering----

# Extraer coordenadas
coordinates = np.array([(museum['latitude'], museum['longitude']) for museum in museums])

# Número de clusters (días)
num_days = int(input("Ingrese el número de días: "))  # Solicitar el número de días

# Realizar clustering
kmeans = KMeans(n_clusters=num_days, random_state=0).fit(coordinates)
labels = kmeans.labels_

# Agrupar museos por clusters
clusters = {i: [] for i in range(num_days)}
for label, museum in zip(labels, museums):
    clusters[label].append(museum)

# Reasignar museos a los clusters para equilibrar el número de elementos
while True:
    max_cluster_size = max(len(museums) for museums in clusters.values())
    min_cluster_size = min(len(museums) for museums in clusters.values())
    if max_cluster_size - min_cluster_size <= 1:
        break
    max_cluster = max(clusters, key=lambda x: len(clusters[x]))
    min_cluster = min(clusters, key=lambda x: len(clusters[x]))
    museum_to_move = clusters[max_cluster].pop()
    clusters[min_cluster].append(museum_to_move)

# ----Reducir el número de museos a visitar por día en función de la distancia y el tiempo de visita----

# Función para calcular la distancia entre dos museos
def calculate_distance(museum1, museum2):
    # Convertir latitud y longitud de grados a radianes
    lat1, lon1 = map(radians, [museum1['latitude'], museum1['longitude']])
    lat2, lon2 = map(radians, [museum2['latitude'], museum2['longitude']])

    # Fórmula de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radio de la Tierra en kilómetros
    return c * r

# Función para obtener los museos más cercanos
def get_closest_museums(museums, max_distance, max_time):
    closest_museums = []
    total_time = 0
    # Ordenar por prioridad y tiempo de visita
    museums.sort(key=lambda x: (x['priority'], x['visit_time']), reverse=True)
    for museum in museums:
        if all(calculate_distance(museum, other) <= max_distance for other in closest_museums):
            if total_time + museum['visit_time'] <= max_time:
                closest_museums.append(museum)
                total_time += museum['visit_time']
    return closest_museums

# Solicitar la distancia máxima y el tiempo máximo por día
max_distance = float(input("Ingrese la distancia máxima por día (en kilómetros): "))
max_time = int(input("Ingrese el tiempo máximo por día (en horas): "))

# Reducir el número de museos a visitar por día en función de la distancia y el tiempo de visita
for day, museums in clusters.items():
    clusters[day] = get_closest_museums(museums, max_distance=max_distance, max_time=max_time)

# ----Optimizar la ruta----

# Inicializar cliente de Google Maps
gmaps = googlemaps.Client(key='YOUR_KEY')  

def get_route(museums):
    waypoints = [f"{museum['latitude']},{museum['longitude']}" for museum in museums]
    directions = []
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        direction = gmaps.directions(
            origin=start,
            destination=end,
            mode="transit",
            language="es"
        )
        # Agregar los datos de los museos de origen y destino a la dirección
        direction[0]['from'] = museums[i]
        direction[0]['to'] = museums[i + 1]        
        directions.extend(direction)
    return directions

# Obtener rutas optimizadas para cada cluster
for day, museums in clusters.items():
    route = get_route(museums)
    print('\n**********************') 
    print(f"Día {day + 1}:")
    print('**********************')
    for i, leg in enumerate(route):
        if i== 0:
            print('\n--------------------------')
            print(f"Comienza en {leg['from']['title']}")
            print('--------------------------')
             

        from_museum = leg['from']['title']
        to_museum = leg['to']['title']
        print('\n--------------------------')
        print(f"Desde: {from_museum} hasta {to_museum}")
        print('--------------------------')
        for step in leg['legs'][0]['steps']:
            print(step['html_instructions'])
