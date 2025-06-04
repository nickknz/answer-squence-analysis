import analysis
import pandas as pd
import matplotlib.pyplot as plt

def get_sorted_task_ids(all_submissions_df: pd.DataFrame) -> list:
    """
    Get a list of sorted unique task IDs from the DataFrame.
    The DataFrame should have a 'task_id' column.
    """

    sorted_task_ids = sorted(
        all_submissions_df["task_id"].unique(),
        key=lambda x: tuple(int(part[1:]) for part in x.split("_"))  # "q1_p2_s1_t3" → (1,2,1,3)
    )
    return sorted_task_ids


def get_one_student_submissions(all_submissions_df: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """
    Get all submissions for a single student.
    The DataFrame should have 'student_id', 'task_id', and 'timestamp' columns.
    
    The output DataFrame will have 'student_id', 'task_id', and 'timestamp' columns.
    """
    student_df = (
        all_submissions_df[all_submissions_df['student_id'] == student_id]
        .sort_values(by="timestamp")
        .reset_index(drop=True)
    )
    student_df = student_df.drop(columns=["question", "part", "section", "task"])
    return student_df

def plot_student_submission_counts(student_df: pd.DataFrame, sorted_task_ids: list = None):
    """
    Plot the number of submissions for each task by a single student.
    The DataFrame should have 'task_id' and 'timestamp' columns.
    """
    submission_counts = student_df['task_id'].value_counts().sort_index()
    stu_id = student_df['student_id'].iloc[0] if not student_df.empty else "Unknown"

    if sorted_task_ids is not None:
        submission_counts = submission_counts.reindex(sorted_task_ids, fill_value=0)
    
    plt.figure(figsize=(10, 6))
    submission_counts.plot(kind='bar')
    plt.title(f"Number of Submissions per Task by Student ({stu_id[:6]}...)")
    plt.xlabel('Task ID')
    plt.ylabel('Number of Submissions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualization/student_submission_counts.png")


def plot_student_submissions_timestamp(student_df: pd.DataFrame, sorted_task_ids: list = None):
    """
    Plot the timestamps of submissions for a single student.
    The DataFrame should have 'task_id' and 'timestamp' columns and be sorted by 'timestamp'.
    """
    stu_id = student_df['student_id'].iloc[0] if not student_df.empty else "Unknown"

    student_df["task_order"] = pd.Categorical(
        student_df["task_id"],
        categories=sorted_task_ids,
        ordered=True
    ).codes

    plt.figure(figsize=(12, 6))
    plt.plot(student_df["timestamp"], student_df["task_order"], marker="o", linestyle="-", color="royalblue")
    plt.yticks(range(len(sorted_task_ids)), sorted_task_ids)
    plt.title(f"Submissions Over Time for Student ({stu_id[:6]}...)")
    plt.xlabel("Timestamp")
    plt.ylabel("Task ID")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualization/student_ansewr_sequence.png")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    all_submissions_df = analysis.create_all_submissions_df_from_JSON("example-student-sequence-v2.json")
    one_student_df = get_one_student_submissions(all_submissions_df, "4a7c189a4f8f93ceccc87c51e48a7a327bae04899ace4a161b6103c5035c1dff")

    task_ids = get_sorted_task_ids(all_submissions_df)

    print("Submissions for one student:")
    print(one_student_df)

    plot_student_submission_counts(one_student_df, task_ids)
    plot_student_submissions_timestamp(one_student_df, task_ids)

    print("Analysis complete.")