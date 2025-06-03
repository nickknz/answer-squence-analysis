import analysis
import json
import pandas as pd



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



if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    all_submissions_df = analysis.create_all_submissions_df_from_JSON("example-student-sequence-v2.json")
    one_student_df = get_one_student_submissions(all_submissions_df, "4a7c189a4f8f93ceccc87c51e48a7a327bae04899ace4a161b6103c5035c1dff")

    print("Submissions for one student:")
    print(one_student_df)


    print("Analysis complete.")