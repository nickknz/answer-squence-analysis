import json
import pandas as pd

def create_all_submissions_df_from_JSON(filename: str) -> pd.DataFrame:
    """
    Create a DataFrame from the example student sequence JSON file.
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

def count_submissions_each_student_each_task(all_submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of submissions per student for each task.
    The DataFrame should have 'student_id', 'task_id', and 'timestamp' columns.
    
    The output DataFrame will have 'student_id', 'task_id', and 'submission_count' columns.
    """
    
    per_task_submission_counts = all_submissions_df.groupby(['student_id', 'task_id']).size().reset_index(name='submission_count')
    return per_task_submission_counts

def calculate_average_submission_count_per_task(all_submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average number of submissions per task across all students.
    The DataFrame should have 'student_id', 'task_id', and 'timestamp' columns.
    
    The output DataFrame will have 'task_id' and 'average_submission_count' columns.
    """
    
    task_submission_counts = count_submissions_each_student_each_task(all_submissions_df)
    average_submission_counts = task_submission_counts.groupby('task_id')['submission_count'].mean().reset_index(name='average_submission_count')
    return average_submission_counts

def count_total_submissions_per_student(all_submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of submissions per student.
    The DataFrame should have 'student_id' and 'timestamp' columns.
    
    The output DataFrame will have 'student_id' and 'total_submission_count' columns.
    """
    
    total_submission_counts = all_submissions_df.groupby('student_id').size().reset_index(name='total_submission_count')
    return total_submission_counts

def count_total_submissions_per_task(all_submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of submissions per task.
    The DataFrame should have 'task_id' and 'timestamp' columns.
    
    The output DataFrame will have 'task_id' and 'total_submission_count' columns.
    """
    
    total_submission_counts = all_submissions_df.groupby('task_id').size().reset_index(name='total_submission_count')
    return total_submission_counts

def calculate_time_spent_each_task_each_student(all_submissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the time spent on each task by each student.
    The DataFrame should have 'student_id', 'task_id', and 'timestamp' columns 
    and they are ordered by 'student_id' and 'timestamp'.

    The output DataFrame will have 'student', 'task_id', and 'duration' columns.
    """
    
    df = all_submissions_df.sort_values(by=['student_id', 'timestamp']).reset_index(drop=True)

    df['prev_submission_time'] = df.groupby('student_id')['timestamp'].shift(1)

    # Just an example for filling NaN values in 'prev_submission_time'
    exam_start_time = pd.to_datetime("2025-05-08 20:00:00+00:00")
    df['prev_submission_time'] = df['prev_submission_time'].fillna(exam_start_time)

    # Calculate duration in seconds
    df['time_spent_seconds'] = (df['timestamp'] - df['prev_submission_time']).dt.total_seconds()
    
    return df[['student_id', 'timestamp', 'prev_submission_time', 'task_id', 'time_spent_seconds']]


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    print("Data analysis complete. Processed data:")
    all_submissions_df = create_all_submissions_df_from_JSON("example-student-sequence.json")
    student_task_duration_df = calculate_time_spent_each_task_each_student(all_submissions_df)
    print(student_task_duration_df)
    # print(f"Total unique students: {df_clean['student'].nunique()}")
    # print(f"Total unique tasks: {df_clean['task_id'].nunique()}")
    # print(f"Total submissions processed: {len(df_clean)}")
    # print("Duration statistics:")
    # print(df_clean['duration'].describe())
    print("Analysis complete.")
