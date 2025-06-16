import pandas as pd
import json

def create_all_submissions_df_from_JSON(filename: str) -> pd.DataFrame:
    """
    Create a DataFrame from the student sequence JSON file.
    Each submission is expanded to include student_id and task_id.
    And ordered by student_id and timestamp.
    """
    
    with open(filename) as f:
        student_view_data = json.load(f)

    all_submissions = []
    for student_id, submissions in student_view_data.items():
        for s in submissions:
            s['student_id'] = student_id
            s['timestamp'] = pd.to_datetime(s['timestamp'])
            s['task_id'] = f"q{s['question']}_p{s['part']}_s{s['section']}_t{s['task']}"
            all_submissions.append(s)

    all_submissions_df = pd.DataFrame(all_submissions)
    return all_submissions_df

def store_datafram_to_parquet(df: pd.DataFrame, filename: str):
    """
    Store the DataFrame to a Parquet file.
    """
    df.to_parquet(filename)

def store_datafram_to_csv(df: pd.DataFrame, filename: str):
    """
    Store the DataFrame to a CSV file.
    """
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    # Example usage
    all_submissions_df = create_all_submissions_df_from_JSON("example-student-sequence-v2.json")
    # store_datafram_to_parquet(all_submissions_df, "data/all_submissions.parquet")
    store_datafram_to_csv(all_submissions_df, "data/all_submissions.csv")
    print("DataFrame created and stored successfully.")