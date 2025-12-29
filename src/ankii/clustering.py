"""Clustering and dimensionality reduction for card embeddings."""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Try to import HDBSCAN, fall back to KMeans if not available
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        method: "tsne" (only option for now, UMAP needs older Python)
        n_components: Target dimensions (usually 2 for visualization)
        perplexity: t-SNE perplexity (lower for small datasets)
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_components)
    """
    n_samples = len(embeddings)
    
    # Adjust perplexity for small datasets
    adjusted_perplexity = min(perplexity, max(5, n_samples // 4))
    
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=adjusted_perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int = 5,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """Cluster embeddings into groups.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        method: "hdbscan" or "kmeans"
        n_clusters: Number of clusters for KMeans
        min_cluster_size: Minimum cluster size for HDBSCAN
        
    Returns:
        Array of cluster labels, shape (n_samples,)
    """
    # Normalize embeddings
    scaler = StandardScaler()
    normalized = scaler.fit_transform(embeddings)
    
    if method == "hdbscan" and HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(normalized)
    elif method == "kmeans" or not HAS_HDBSCAN:
        clusterer = KMeans(
            n_clusters=min(n_clusters, len(embeddings)),
            random_state=42,
            n_init=10,
        )
        labels = clusterer.fit_predict(normalized)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels


def get_cluster_info(labels: np.ndarray) -> dict:
    """Get information about clusters.
    
    Returns:
        Dict with cluster stats
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
    n_noise = np.sum(labels == -1)
    
    cluster_sizes = {}
    for label in unique_labels:
        if label >= 0:
            cluster_sizes[int(label)] = int(np.sum(labels == label))
    
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes,
    }


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        
    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    
    # Cosine similarity is just dot product of normalized vectors
    return np.dot(normalized, normalized.T)


def find_similar_pairs(
    embeddings: np.ndarray,
    threshold: float = 0.9,
) -> list[tuple[int, int, float]]:
    """Find pairs of cards with high similarity.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        threshold: Minimum cosine similarity to consider (0-1)
        
    Returns:
        List of (idx1, idx2, similarity) tuples, sorted by similarity descending
    """
    sim_matrix = cosine_similarity_matrix(embeddings)
    n = len(embeddings)
    
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                pairs.append((i, j, float(sim)))
    
    # Sort by similarity, highest first
    pairs.sort(key=lambda x: -x[2])
    return pairs


def group_similar_cards(
    embeddings: np.ndarray,
    threshold: float = 0.9,
) -> list[list[int]]:
    """Group cards by similarity using connected components.
    
    Cards with similarity >= threshold are grouped together.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        threshold: Minimum cosine similarity to group together
        
    Returns:
        List of groups, each group is a list of card indices
    """
    n = len(embeddings)
    pairs = find_similar_pairs(embeddings, threshold)
    
    # Build adjacency using union-find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for i, j, _ in pairs:
        union(i, j)
    
    # Group by root
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)
    
    # Only return groups with more than 1 card (actual duplicates)
    groups = [g for g in groups_dict.values() if len(g) > 1]
    
    # Sort groups by size, largest first
    groups.sort(key=lambda g: -len(g))
    
    return groups
