import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def submissions_to_graph_for_all_students(submissions_df: pd.DataFrame, student_id_col: str = 'student_id', task_id_col: str = 'task_id', timestamp_col: str = 'timestamp'):
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
    G = nx.MultiDiGraph()
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

def plot_one_student_graph(
    student_id: str,
    graph: nx.MultiDiGraph,):
    """
    Plot the submission path of a single student as a directed graph.
    Parameters:
    - student_id: The ID of the student whose path is to be plotted.
    - student_paths: Dictionary mapping student IDs to their ordered list of task IDs.
    - graph: The directed graph containing the students' submission paths.
    """

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)  # Use spring layout for better visualization
    nx.draw(graph, pos, with_labels=True, node_size=500, font_size=8, arrows=True)
    plt.title(f"Submission Path for Student {student_id}")
    plt.show()

def generate_students_paths(submissions_df: pd.DataFrame, student_id_col: str = 'student_id', task_id_col: str = 'task_id', timestamp_col: str = 'timestamp'):
    """
    Generate a student's submission path from the submissions DataFrame.
    
    Parameters:
    - submissions_df: DataFrame containing student submissions.
    - student_id_col: Column name for student IDs.
    - task_id_col: Column name for task/question IDs.
    - timestamp_col: Column name for submission timestamps.

    Returns:
    - students_paths: Dict mapping student_id to their ordered list of task_ids (path).
    """

    students_paths = {}
    # Ensure correct ordering
    submissions_df = submissions_df.sort_values([student_id_col, timestamp_col])
    for student_id, group in submissions_df.groupby(student_id_col):
        path = group[task_id_col].tolist()
        students_paths[student_id] = path

    return students_paths

def generate_one_student_graph(student_id, student_path):
    """
    Generate a directed graph for a single student's submission path.
    Parameters:
    - student_id: The ID of the student whose path is to be generated.
    - student_path: A list of task IDs representing the student's ordered submissions.
    Returns:
    - subG: A directed graph representing the student's submission path.
    """
    subG = nx.MultiDiGraph()
    for i in range(len(student_path) - 1):
        subG.add_edge(student_path[i], student_path[i + 1])
    return subG

def print_edges_for_student(G, student_id):
    """
    Print all edges in the graph G that belong to a specific student.
    Parameters:
    - G: The directed graph containing one students' submission paths.
    - student_id: The ID of the student whose edges are to be printed.
    """
    edges = [(u, v) for u, v, d in G.edges(data=True)]
    for edge in edges:
        print(edge)


if __name__ == "__main__":
    df = pd.read_parquet("data/all_submissions.parquet")
    # G, student_paths = submissions_to_graph_for_all_students(df)
    students_paths = generate_students_paths(df)

    # create a directed graph for all students' submission paths
    student_graphs = {}
    for student_id, path in students_paths.items():
        student_graphs[student_id] = generate_one_student_graph(student_id, path)

    student_id = 'aba64fa34e052d9f2c5473f26136afa4c053f049191f40ea7e0d56708274d9a8'
    one_student_path = students_paths[student_id]
    one_student_graph = student_graphs[student_id]
    print_edges_for_student(one_student_graph, student_id)

    # Plot one student's submission path
    plot_one_student_graph(student_id, one_student_graph)
