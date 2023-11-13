import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# Cargar los datos desde el archivo CSV
data = pd.read_csv("iris2.csv")
selected_columns = data[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
data_array = selected_columns.to_numpy()

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Función para inicializar los centroides aleatoriamente
def initialize_centroids(data, k):
    num_samples, num_features = data.shape
    centroids = np.zeros((k, num_features))
    for i in range(k):
        centroid = data[random.randint(0, num_samples - 1)]
        centroids[i] = centroid
    return centroids

# Función para asignar puntos a clusters
def assign_to_clusters(data, centroids):
    num_samples = data.shape[0]
    distances = np.zeros((num_samples, len(centroids)))
    for i in range(len(centroids)):
        for j in range(num_samples):
            distances[j, i] = euclidean_distance(centroids[i], data[j])
    return np.argmin(distances, axis=1)

# Función para actualizar los centroides
def update_centroids(data, cluster_labels, k):
    num_samples, num_features = data.shape
    new_centroids = np.zeros((k, num_features))
    for i in range(k):
        points_in_cluster = data[cluster_labels == i]
        new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids

# Función para comprobar si los centroides han convergido
def has_converged(old_centroids, new_centroids, tolerance=1e-4):
    return np.all(np.abs(new_centroids - old_centroids) < tolerance)

# Función principal de K-Means
def k_means(data, k, max_iterations=100):
    num_samples = data.shape[0]
    num_features = data.shape[1]
    
    # Inicializar centroides aleatoriamente
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iterations):
        # Asignar puntos a clusters
        cluster_labels = assign_to_clusters(data, centroids)
        
        # Guardar los centroides antiguos para verificar la convergencia
        old_centroids = centroids
        
        # Actualizar los centroides
        centroids = update_centroids(data, cluster_labels, k)
        
        # Comprobar si los centroides han convergido
        if has_converged(old_centroids, centroids):
            print('\nLos centroides han convergido.\n')
            break

    return centroids, cluster_labels + 1

# Especificar el número de clusters (k)
k = 3

# Ejecutar el algoritmo K-Means
centroids, cluster_labels = k_means(data_array, k)

# Imprimir los centroides finales
print("Centroides finales:\n")
print(centroids)

# Imprimir las etiquetas de cluster para cada instancia
cluster_assignments = pd.DataFrame({'Instancia': range(1, len(cluster_labels)+1), 'Cluster No.': cluster_labels})
cluster_assignments['Data'] = [str(data_array[i]) for i in range(len(cluster_labels))]

print("\nEtiqueta de cluster para cada instancia:\n")
print(cluster_assignments.to_string(index=False))

