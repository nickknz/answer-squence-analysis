import src.utils.answer_sequence_vis as vis_tools
from src.utils.plotly_sequence_vis import plot_multiple_students_submission_sequence_plotly


def visualize_clusters_paths(df, student_clusters, student_paths, output_dir="../../visualization/"):
    """
    Create visualizations for student paths in each cluster
    
    Args:
        df (pd.DataFrame): Original submission data
        student_clusters (dict): Mapping from student ID to cluster ID
        student_paths (dict): Mapping from student ID to path sequence
        output_dir (str): Output directory for visualizations
    """
    import os
    from collections import defaultdict
    
    # Organize students by cluster
    clusters = defaultdict(list)
    for student_id, cluster_id in student_clusters.items():
        clusters[cluster_id].append(student_id)
    
    # Get sorted task IDs
    sorted_task_ids = vis_tools.get_sorted_task_ids(df)
    
    # Create visualization for each cluster
    for cluster_id, student_ids in clusters.items():
        print(f"Processing Cluster {cluster_id} with {len(student_ids)} students")
        
        # Create cluster directory
        cluster_dir = os.path.join(output_dir, f"cluster{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Filter data for students in this cluster
        cluster_students_df = df[df['student_id'].isin(student_ids)].copy()
        
        if len(cluster_students_df) == 0:
            print(f"Warning: No data found for Cluster {cluster_id}")
            continue
        
        # Generate visualization for this cluster
        try:
            cluster_html = plot_multiple_students_submission_sequence_plotly(
                cluster_students_df, 
                sorted_task_ids,
                title=f"Cluster {cluster_id} - {len(student_ids)} Students Learning Paths"
            )
            
            # Save HTML file
            output_file = os.path.join(cluster_dir, f"cluster{cluster_id}_visualization.html")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cluster_html)
            
            print(f"✓ Cluster {cluster_id} visualization saved to: {output_file}")
            
            # Generate cluster summary information
            # summary_info = generate_cluster_summary(cluster_id, student_ids, student_paths)
            # summary_file = os.path.join(cluster_dir, f"cluster{cluster_id}_summary.txt")
            # with open(summary_file, "w", encoding="utf-8") as f:
            #     f.write(summary_info)
            
        except Exception as e:
            print(f"❌ Error processing Cluster {cluster_id}: {str(e)}")
    
    print(f"\nAll cluster visualizations completed! Files saved in: {output_dir}")

def generate_cluster_summary(cluster_id, student_ids, student_paths):
    """Generate summary information for a cluster"""
    import numpy as np
    from collections import Counter
    
    summary = f"Cluster {cluster_id} Summary\n"
    summary += "=" * 50 + "\n\n"
    
    # Basic statistics
    summary += f"Number of students: {len(student_ids)}\n"
    
    # Path length statistics
    valid_students = [sid for sid in student_ids if sid in student_paths]
    if valid_students:
        path_lengths = [len(student_paths[sid]) for sid in valid_students]
        summary += f"Average path length: {np.mean(path_lengths):.2f}\n"
        summary += f"Path length range: {min(path_lengths)} - {max(path_lengths)}\n\n"
        
        # Most common starting and ending questions
        first_questions = [student_paths[sid][0] for sid in valid_students if len(student_paths[sid]) > 0]
        last_questions = [student_paths[sid][-1] for sid in valid_students if len(student_paths[sid]) > 0]
        
        if first_questions:
            most_common_first = Counter(first_questions).most_common(3)
            summary += "Most common starting questions:\n"
            for i, (question, count) in enumerate(most_common_first, 1):
                summary += f"  {i}. {question}: {count} times ({count/len(first_questions)*100:.1f}%)\n"
        
        if last_questions:
            most_common_last = Counter(last_questions).most_common(3)
            summary += "\nMost common ending questions:\n"
            for i, (question, count) in enumerate(most_common_last, 1):
                summary += f"  {i}. {question}: {count} times ({count/len(last_questions)*100:.1f}%)\n"
    
    # Student ID list
    summary += f"\nStudent ID list:\n"
    for i, sid in enumerate(student_ids, 1):
        path_len = len(student_paths.get(sid, [])) if sid in student_paths else 0
        summary += f"  {i:2d}. {sid[:16]}... (path length: {path_len})\n"
    
    return summary

def visualize_specific_clusters(df, student_clusters, student_paths, cluster_ids, output_dir="../../visualization"):
    """Visualize only specified clusters"""
    filtered_clusters = {k: v for k, v in student_clusters.items() if v in cluster_ids}
    visualize_clusters_paths(df, filtered_clusters, student_paths, output_dir)

