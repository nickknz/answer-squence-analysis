import numpy as np
from node2vec import Node2Vec
import networkx as nx

def embed_graph(graph, dimensions=128, walk_length=30, num_walks=200, workers=4):
    """
    Convert a graph to node embeddings using node2vec
    
    Args:
        graph (nx.Graph): NetworkX graph of student submissions
        dimensions (int): Embedding dimensions
        walk_length (int): Length of each random walk
        num_walks (int): Number of random walks per node
        workers (int): Number of parallel workers
        
    Returns:
        dict: Dictionary mapping node IDs to their embeddings
    """
    # Initialize node2vec model
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers
    )
    
    # Train node2vec model
    model = node2vec.fit(window=10, min_count=1)
    
    # Get embeddings for each node in the graph
    embeddings = {}
    for node in graph.nodes():
        embeddings[node] = model.wv.get_vector(str(node))
    
    return embeddings