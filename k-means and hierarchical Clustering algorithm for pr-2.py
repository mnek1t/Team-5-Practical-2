# k-means algorithm !!!!!!!!!!!!!!!!
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ADD DATASET HERE For example: my_working_dataset = pd.read_csv('path_to_our_dataset.csv')

# Selecting the attributes for clustering
features = ['MajorAxisLength', 'MinorAxisLength', 'AspectRatio']
clustering_dataset = my_working_dataset[features]

# Normalize the features to give them equal importance
scaler = StandardScaler()
clustering_dataset_scaled = scaler.fit_transform(clustering_dataset)

# Creating a list of silhouette coefficient values
silhouette_coefficients = []

# A loop with 5 iterations is implemented (for 2 to 6 clusters)
for i in range(2, 7):
    # Creation of a K-means clustering model where the number of clusters (n_clusters) is equal to the loop iteration number
    kmeans_model = KMeans(n_clusters=i, init='k-means++', random_state=42)
    
    # Training of K-means model
    kmeans_model.fit(clustering_dataset_scaled)
    
    # Calculating the Silhouette coefficient
    silhouette_score = metrics.silhouette_score(clustering_dataset_scaled, kmeans_model.labels_)
    silhouette_coefficients.append(silhouette_score)
    print(f"Silhouette Score for {i} clusters: {silhouette_score}")

# Representing the results visually
plt.figure(figsize=(6, 4))
plt.plot(range(2, 7), silhouette_coefficients, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficients for Varying Number of Clusters')
plt.grid(True)
plt.show()



# Hierarchical Clustering algorithm !!!!!!!!!!
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc


# ADD DATASET HERE Example: my_working_dataset = pd.read_csv('path_to_your_dataset.csv')

# Selecting a subset of features for the clustering task
features = ['MajorAxisLength', 'MinorAxisLength', 'AspectRatio']
clustering_dataset = my_working_dataset[features]

# Normalize the features to give them equal importance
scaler = StandardScaler()
clustering_dataset_scaled = scaler.fit_transform(clustering_dataset)

# Creating an Agglomerative Clustering model
clustering = AgglomerativeClustering().fit(clustering_dataset_scaled)

# Plotting the dendrogram to visualize the clustering
plt.figure(figsize=(14, 8))
plt.title('Dendrogram')
Dendrogram = shc.dendrogram(shc.linkage(clustering_dataset_scaled, method='ward'))
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Experiment with different cut-off lines to form clusters
thresholds = [5, 10, 15]  # Define different thresholds for experiments
colors = ['viridis', 'plasma', 'inferno']  # Color maps for visualization
for i, t in enumerate(thresholds):
    plt.figure(figsize=(6, 4))
    clusters = shc.fcluster(shc.linkage(clustering_dataset_scaled, method='ward'), t, criterion='distance')
    plt.scatter(clustering_dataset['MajorAxisLength'], clustering_dataset['MinorAxisLength'], c=clusters, cmap=colors[i])
    plt.title(f'Clusters at threshold {t}')
    plt.xlabel('Major Axis Length')
    plt.ylabel('Minor Axis Length')
    plt.colorbar(label='Cluster Label')
    plt.show()
