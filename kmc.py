import csv
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
random.seed(170)

# Function to read data from CSV file
def read_data(filename):
    """
    Reads data from a CSV file excluding the first column (assumed to be labels).

    Args:
    - filename (str): Name of the CSV file to read

    Returns:
    - data (list): List of data points read from the CSV file
    """
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(val) for val in row[1:]])  # Exclude the first column
    return data

# Function to initialize centroids randomly
def initialize_centroids(data, k):
    """
    Randomly initializes centroids from the data points.

    Args:
    - data (list): List of data points
    - k (int): Number of centroids

    Returns:
    - centroids (list): List of randomly initialized centroids
    """
    centroids = random.sample(data, k)
    return centroids

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
    - point1 (list): First point
    - point2 (list): Second point

    Returns:
    - distance (float): Euclidean distance between the points
    """
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

# Function to assign points to the nearest centroid
def assign_points(data, centroids):
    """
    Assigns data points to the nearest centroids.

    Args:
    - data (list): List of data points
    - centroids (list): List of centroids

    Returns:
    - clusters (dict): Dictionary with keys as centroid indices and values as lists of assigned points
    """
    clusters = {i: [] for i in range(len(centroids))}
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

# Function to update centroids based on mean of points in each cluster
def update_centroids(clusters):
    """
    Updates centroids based on the mean of points in each cluster.

    Args:
    - clusters (dict): Dictionary with keys as centroid indices and values as lists of assigned points

    Returns:
    - centroids (list): Updated list of centroids
    """
    centroids = []
    for cluster_index, points in clusters.items():
        if len(points) > 0:
            centroid = [sum(coords) / len(coords) for coords in zip(*points)]
            centroids.append(centroid)
        else:
            centroids.append([])
    return centroids

# Function to check convergence of centroids
def has_converged(old_centroids, new_centroids, epsilon):
    """
    Checks convergence of centroids based on a threshold epsilon.

    Args:
    - old_centroids (list): List of old centroids
    - new_centroids (list): List of new centroids
    - epsilon (float): Threshold for convergence

    Returns:
    - converged (bool): True if centroids have converged, False otherwise
    """
    for i in range(len(old_centroids)):
        if len(old_centroids[i]) > 0 and len(new_centroids[i]) > 0:
            if euclidean_distance(old_centroids[i], new_centroids[i]) > epsilon:
                return False
    return True

# K-Means Clustering Algorithm
def kmeans(data, k, max_iterations=100, epsilon=0.001):
    """
    Performs K-Means clustering on the given data.

    Args:
    - data (list): List of data points
    - k (int): Number of clusters
    - max_iterations (int): Maximum number of iterations
    - epsilon (float): Threshold for convergence

    Returns:
    - centroids (list): Final centroids after clustering
    - clusters (dict): Dictionary with keys as centroid indices and values as lists of assigned points
    """
    centroids = initialize_centroids(data, k)
    iteration = 0
    while True:
        clusters = assign_points(data, centroids)
        old_centroids = centroids.copy()
        centroids = update_centroids(clusters)
        iteration += 1
        if iteration >= max_iterations or has_converged(old_centroids, centroids, epsilon):
            break
    return centroids, clusters,iteration

# Function to read the first column from a CSV file
def read_first_column(filename):
    first_column = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            first_column.append(row[0])  # Extract the first column value
    return first_column

def read_second_column(filename):
    second_column = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            second_column[row[0]] = row[1]  # Store the second column value with first column as key
    return second_column

filename_players_stat = "players_stat.csv"
data = read_data(filename_players_stat)

# Read the first column from players_stat.csv
first_column = read_first_column(filename_players_stat)

filename_players_id = "players_id.csv"
second_column = read_second_column(filename_players_id)

# Ensure the length of first_column matches the data length
if len(first_column) != len(data):
    raise ValueError("Length mismatch between the first column and data")

# Perform K-Means Clustering
k = 6  # Define the number of clusters
final_centroids, clusters, num_iterations = kmeans(data, k)

# Collect cluster assignments while keeping original order
cluster_assignments = {}
for i, points in clusters.items():
    for point in points:
        index = data.index(point)
        cluster_assignments[first_column[index]] = i  # Store the first column value instead of index

# Write results to output.txt
with open('output.txt', 'w') as file:
    file.write("K-Means Clustering Output from players_stat.csv\n")
    file.write(f"k = {k}\n\n")
    file.write("Initial Centroids\n")
    initial_centroids = initialize_centroids(data, k)
    for centroid in initial_centroids:
        file.write(f"{tuple(centroid)}\n")
    file.write("\n")

    file.write("Final Centroids:\n")
    for i, centroid in enumerate(final_centroids):
        file.write(f"Cluster {i+1}: {centroid}\n")

    file.write("Clustered Data:\n")
    for index, point in enumerate(data):
        first_col_val = first_column[index]
        second_col_val = second_column.get(first_col_val, "")  # Retrieve second column value based on first column value
        cluster = cluster_assignments.get(first_col_val, -1)  # Retrieve cluster based on the first column value
        
        # If second column value doesn't exist, set it to "NA"
        if not second_col_val:
            second_col_val = "NA"

        file.write(f"{cluster} : {first_col_val} - {second_col_val:<30} {point}\n")
    file.write(f"\nNumber of iterations: {num_iterations}\n")

# FOR SCATTERPLOT
clustered_points = {i: [] for i in range(k)}
for index, point in enumerate(data):
    cluster = cluster_assignments.get(first_column[index], -1)
    if cluster != -1:
        clustered_points[cluster].append(point)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors for each cluster
colors = ['r', 'g', 'b', 'y', 'c', 'm']

# Plot the points for each cluster in 3D space
for i, points in clustered_points.items():
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]
    ax.scatter(x, y, z, c=colors[i], label=f'Cluster {i}')

# Set labels and title
ax.set_xlabel('Points Per Game')
ax.set_ylabel('Assists Per Game')
ax.set_zlabel('Rebounds Per Game')
ax.set_title('Clustering of NBA Players')

# Add a legend
ax.legend()

# Show the plot
plt.show()
