import sys
import os
import numpy as np
import networkx as nx
import torch

# Add project root to path to import from submissions_graph_transform.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from submissions_graph_transform import submissions_to_graph_for_all_students
from models.embedding import embed_graph
from src.models.sequence_model import train_sequence_model
from models.clustering import perform_clustering, find_optimal_clusters, visualize_clusters

class SubmissionAnalysisPipeline:
    """
    Pipeline for analyzing student submission patterns through graph embedding and clustering
    """
    
    def __init__(self, embedding_dim=128, lstm_hidden_dim=64, lstm_layers=1, 
                 n_clusters=None, auto_find_clusters=True, max_clusters=10):
        """
        Initialize the pipeline
        
        Args:
            embedding_dim (int): Dimension for node2vec embeddings
            lstm_hidden_dim (int): Hidden dimension for LSTM model
            lstm_layers (int): Number of LSTM layers
            n_clusters (int): Number of clusters (if None, auto-find optimal clusters)
            auto_find_clusters (bool): Whether to automatically find optimal number of clusters
            max_clusters (int): Maximum number of clusters to consider if auto-finding
        """
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.n_clusters = n_clusters
        self.auto_find_clusters = auto_find_clusters
        self.max_clusters = max_clusters
        
        # Models to be trained
        self.node_embeddings = None
        self.lstm_model = None
        self.sequence_embeddings = None
        self.kmeans_model = None
        self.cluster_assignments = None
        
    def run(self, submissions_data, student_submission_sequences=None):
        """
        Run the full analysis pipeline
        
        Args:
            submissions_data: Data for constructing the submission graph
            student_submission_sequences (dict): If provided, a dictionary mapping student IDs to their submission sequences
            
        Returns:
            dict: Cluster assignments for each student
        """
        # Step 1: Create submission graph for all students
        print("Creating submission graph...")
        graph = submissions_to_graph_for_all_students(submissions_data)
        
        # Step 2: Generate node embeddings using node2vec
        print("Generating node embeddings...")
        self.node_embeddings = embed_graph(graph, dimensions=self.embedding_dim)
        
        # Step 3: If student_submission_sequences not provided, extract from graph
        if student_submission_sequences is None:
            # This assumes graph structure has information about which student submitted which answer
            # and in what order. You might need to adapt this based on your graph structure.
            print("Extracting submission sequences from graph...")
            student_submission_sequences = self._extract_submission_sequences(graph)
        
        # Step 4: Create sequence embeddings using LSTM
        print("Training sequence model...")
        self.lstm_model, self.sequence_embeddings = train_sequence_model(
            student_submission_sequences,
            self.node_embeddings,
            hidden_dim=self.lstm_hidden_dim,
            num_layers=self.lstm_layers
        )
        
        # Step 5: Perform clustering
        if self.auto_find_clusters and self.n_clusters is None:
            print("Finding optimal number of clusters...")
            self.n_clusters = find_optimal_clusters(
                self.sequence_embeddings, 
                max_clusters=self.max_clusters
            )
            print(f"Optimal number of clusters: {self.n_clusters}")
        
        print(f"Performing clustering with {self.n_clusters} clusters...")
        self.kmeans_model, self.cluster_assignments = perform_clustering(
            self.sequence_embeddings,
            n_clusters=self.n_clusters
        )
        
        # Step 6: Visualize clusters
        print("Visualizing clusters...")
        visualize_clusters(self.sequence_embeddings, self.cluster_assignments)
        
        return self.cluster_assignments
    
    def _extract_submission_sequences(self, graph):
        """
        Extract submission sequences for each student from the graph
        
        Args:
            graph (nx.Graph): Submission graph
            
        Returns:
            dict: Dictionary mapping student IDs to submission sequences
        """
        # This is a placeholder. You'll need to implement this based on your graph structure.
        # The function should return a dictionary like:
        # {
        #     'student1': [node_id1, node_id2, node_id3, ...],
        #     'student2': [node_id5, node_id1, node_id8, ...],
        #     ...
        # }
        
        # Example implementation (assumes nodes have 'student_id' and 'timestamp' attributes):
        student_sequences = {}
        
        # Group nodes by student
        for node, data in graph.nodes(data=True):
            if 'student_id' in data and 'timestamp' in data:
                student_id = data['student_id']
                
                if student_id not in student_sequences:
                    student_sequences[student_id] = []
                
                student_sequences[student_id].append((node, data['timestamp']))
        
        # Sort each student's submissions by timestamp
        for student_id in student_sequences:
            student_sequences[student_id].sort(key=lambda x: x[1])
            # Keep only node IDs, not timestamps
            student_sequences[student_id] = [node for node, _ in student_sequences[student_id]]
        
        return student_sequences