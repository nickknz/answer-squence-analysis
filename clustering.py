from submissions_graph_transform import generate_students_paths
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def path_to_transition_vector(path):
    mat = np.zeros((n_tasks, n_tasks))
    for i in range(len(path)-1):
        mat[task_idx[path[i]], task_idx[path[i+1]]] += 1
    return mat.flatten()


if __name__ == "__main__":
    df = pd.read_parquet("data/all_submissions.parquet")
    student_paths = generate_students_paths(df)

    all_tasks = sorted({task for path in student_paths.values() for task in path})
    task_idx = {task: i for i, task in enumerate(all_tasks)}
    n_tasks = len(all_tasks)

    X = np.array([path_to_transition_vector(path) for path in student_paths.values()])

    # 聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    print(labels)  