import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def plot_one_student_graph(
    student_id: str,
    student_paths: dict,
    graph: nx.DiGraph,):
    """
    Plot the submission path of a single student as a directed graph.
    Parameters:
    - student_id: The ID of the student whose path is to be plotted.
    - student_paths: Dictionary mapping student IDs to their ordered list of task IDs.
    - graph: The directed graph containing all students' submission paths.
    """
    if student_id not in student_paths:
        print(f"Student ID {student_id} not found in paths.")
        return
    path = student_paths[student_id]
    subG = nx.DiGraph()
    for i in range(len(path) - 1):
        subG.add_edge(path[i], path[i + 1])
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subG, seed=42)  # Use spring layout for better visualization
    nx.draw(subG, pos, with_labels=True, node_size=500, font_size=8, arrows=True)
    plt.title(f"Submission Path for Student {student_id}")
    plt.show()

def submissions_to_graph(submissions_df: pd.DataFrame, student_id_col: str = 'student_id', task_id_col: str = 'task_id', timestamp_col: str = 'timestamp'):
    """
    Transform student submissions into a directed graph where each question/task is a node,
    and each student's ordered submissions form a path (edges) between these nodes.

    Parameters:
    - submissions_df: DataFrame containing student submissions.
    - student_id_col: Column name for student IDs.
    - task_id_col: Column name for task/question IDs.
    - timestamp_col: Column name for submission timestamps.

    Returns:
    - G: networkx.DiGraph representing all students' submission paths.
    - student_paths: Dict mapping student_id to their ordered list of task_ids (path).
    """
    G = nx.DiGraph()
    student_paths = {}

    # Ensure correct ordering
    submissions_df = submissions_df.sort_values([student_id_col, timestamp_col])

    for student_id, group in submissions_df.groupby(student_id_col):
        path = group[task_id_col].tolist()
        student_paths[student_id] = path

        # Add nodes and edges for this student's path
        for i in range(len(path) - 1):
            G.add_node(path[i])
            G.add_node(path[i + 1])
            G.add_edge(path[i], path[i + 1], student=student_id)

    return G, student_paths

if __name__ == "__main__":
    df = pd.read_parquet("data/all_submissions.parquet")
    G, student_paths = submissions_to_graph(df)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print("Sample student path:", next(iter(student_paths.values())))


    student_id = '4a7c189a4f8f93ceccc87c51e48a7a327bae04899ace4a161b6103c5035c1dff'
    # Plot one student's submission path
    plot_one_student_graph(student_id, student_paths, G)
