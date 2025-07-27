import matplotlib.pyplot as plt
from src.utils.submissions_graph_transform import generate_students_paths
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

def path_to_transition_vector(path):
    mat = np.zeros((n_tasks, n_tasks))
    for i in range(len(path)-1):
        mat[task_idx[path[i]], task_idx[path[i+1]]] += 1
    return mat.flatten()


if __name__ == "__main__":
    df = pd.read_parquet("data/all_submissions.parquet")
    student_paths = generate_students_paths(df)
    print(f"Number of students: {len(student_paths)}")

    all_tasks = sorted({task for path in student_paths.values() for task in path})
    task_idx = {task: i for i, task in enumerate(all_tasks)}
    n_tasks = len(all_tasks)

    X = np.array([path_to_transition_vector(path) for path in student_paths.values()])

    inertias = []
    k_range = range(30, len(student_paths), 20)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    save_dir = os.path.join(project_root, "visualization/vector_clustering")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "elbow_method.png")

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Elbow plot saved to {save_path}")