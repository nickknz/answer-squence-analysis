import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def perform_clustering(student_embeddings, n_clusters=5, random_state=42):
    """
    Perform K-means clustering on student embeddings
    
    Args:
        student_embeddings (dict): Dictionary mapping student IDs to embeddings
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (kmeans model, student cluster assignments)
    """
    # Convert student embeddings dictionary to array
    student_ids = list(student_embeddings.keys())
    embeddings_array = np.array([student_embeddings[student_id] for student_id in student_ids])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # Create dictionary mapping student IDs to cluster assignments
    student_clusters = {student_id: label for student_id, label in zip(student_ids, cluster_labels)}
    
    return kmeans, student_clusters

def find_optimal_clusters(student_embeddings, max_clusters=10, random_state=42):
    """
    Find optimal number of clusters using silhouette score
    
    Args:
        student_embeddings (dict): Dictionary mapping student IDs to embeddings
        max_clusters (int): Maximum number of clusters to try
        random_state (int): Random seed for reproducibility
        
    Returns:
        int: Optimal number of clusters
    """
    # Convert student embeddings dictionary to array
    embeddings_array = np.array(list(student_embeddings.values()))
    
    # Try different numbers of clusters
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(embeddings_array)
        
        # Calculate silhouette score
        score = silhouette_score(embeddings_array, labels)
        silhouette_scores.append(score)
        
    # Find optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for different numbers of clusters')
    plt.axvline(x=optimal_clusters, color='red', linestyle='--')
    plt.show()
    
    return optimal_clusters

def visualize_clusters(student_embeddings, cluster_assignments):
    """
    Visualize clusters using PCA
    
    Args:
        student_embeddings (dict): Dictionary mapping student IDs to embeddings
        cluster_assignments (dict): Dictionary mapping student IDs to cluster assignments
    """
    # Convert embeddings to array
    student_ids = list(student_embeddings.keys())
    embeddings_array = np.array([student_embeddings[student_id] for student_id in student_ids])
    labels = np.array([cluster_assignments[student_id] for student_id in student_ids])
    
    # Use PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_array)
    
    # Plot clusters
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        cluster_points = reduced_embeddings[labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
    
    plt.title('Student clusters based on submission patterns')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()