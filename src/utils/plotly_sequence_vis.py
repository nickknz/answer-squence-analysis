
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import src.utils.answer_sequence_vis as vis_tools

def plot_multiple_students_submission_sequence_plotly(
    multiple_students_df: pd.DataFrame,
    sorted_task_ids: list,
    title: str = "Multiple Students' Submission Sequence",
    include_plotlyjs: str = 'cdn'
) -> str:
    """
    use Plotly plot multiple students' answer sequence interactive chart, 
    and return HTML string (can be directly embedded in Flask template).
    
    parameters:
    - multiple_students_df
    - sorted_task_ids
    - include_plotlyjs: 'cdn' or 'directory' or 'None'
    
    returns:
    - HTML string for the Plotly chart. 
    """
    # task_id -> task_order
    multiple_students_df["task_order"] = pd.Categorical(
        multiple_students_df["task_id"],
        categories=sorted_task_ids,
        ordered=True
    ).codes

    readable_task_labels = [convert_task_id_to_answerbook_format(tid) for tid in sorted_task_ids]
    multiple_students_df["readable_task_id"] = multiple_students_df["task_id"].apply(convert_task_id_to_answerbook_format)

    fig = go.Figure()
    
    # add traces for each student
    for student_id, group in multiple_students_df.groupby('student_id'):
        fig.add_trace(go.Scatter(
            x=group['timestamp'],
            y=group['task_order'],
            mode='lines+markers',
            name=student_id[:6] + '...',
            marker=dict(size=8),
            line=dict(width=2),
            opacity=0.7,
            hovertemplate=(
                'Student: %{text}<br>'
                'Task: %{customdata[0]}<br>'
                'Time: %{x|%H:%M:%S}'
            ),
            text=[student_id[:6]]*len(group),
            customdata=group[['readable_task_id']].values
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='Time',
            tickformat='%H:%M',
            dtick=5*60*1000,  # 5 mins
            range=[
                multiple_students_df['timestamp'].min().replace(minute=0, second=0, microsecond=0),
                multiple_students_df['timestamp'].max().ceil('15min')
            ]
        ),
        yaxis=dict(
            title='Task',
            tickmode='array',
            tickvals=list(range(len(readable_task_labels))),
            ticktext=readable_task_labels
        ),
        legend=dict(title='Student ID'),
        margin=dict(l=80, r=40, t=80, b=60),
        hovermode='closest'
    )
    
    # covnert to HTML string
    plot_html = pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs)
    return plot_html

def convert_task_id_to_answerbook_format(task_id: str) -> str:
    q, p, s, t = task_id.split("_")
    q_num = q[1:]
    p_letter = chr(ord('a') + int(p[1:]) - 1)  # 1 -> a, 2 -> b ...
    
    # convert section number to roman
    roman_map = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    s2roman = roman_map[int(s[1:]) - 1] if 0 < int(s[1:]) <= len(roman_map) else f"s{s[1:]}"
    
    t_num = t[1:]
    return f"{q_num}{p_letter}{s2roman}{t_num}"

def filter_submissions_less_than_seconds(
    multiple_students_df: pd.DataFrame, 
    seconds: int = 15
) -> pd.DataFrame:
    """
    Filter out submissions that are less than `seconds` apart.
    
    parameters:
    - multiple_students_df: DataFrame with 'timestamp' column.
    - seconds: Minimum time difference in seconds.
    
    returns:
    - Filtered DataFrame.
    """
    multiple_students_df = multiple_students_df.sort_values(by=['student_id', 'timestamp'])
    
    # Calculate time difference between consecutive submissions
    multiple_students_df['interval_sec'] = multiple_students_df.groupby('student_id')['timestamp'].diff().dt.total_seconds()
    
    # Filter out rows where time difference is less than `seconds`
    filtered_df = multiple_students_df[multiple_students_df['interval_sec'].fillna(seconds + 1) >= seconds]
    
    return filtered_df.drop(columns=['interval_sec'])

def filter_submissions_by_answer_length(
    multiple_students_df: pd.DataFrame, 
    min_length: int = 1
) -> pd.DataFrame:
    """
    Filter out submissions with answer length less than 'min_length'.
    
    parameters:
    - multiple_students_df: DataFrame with 'answer' column.
    - min_length: Minimum length of the answer.
    
    returns:
    - Filtered DataFrame.
    """
    return multiple_students_df[multiple_students_df['answer'].str.len() >= min_length].copy()

def plotly_one_students_submission_interval_counts(
    one_students_df: pd.DataFrame,
    bin_size: int = 30,
) -> str:
    """
    Plot one student the intervals of sumbissions count

    parameters:
    - one_students_df: DataFrame with 'task_id' and 'timestamp' columns.
    - bin_size: Size of the bins for the histogram (default is 30 seconds).
    
    returns:
    - HTML string for the Plotly chart.
    """

    one_students_df = one_students_df.sort_values("timestamp").copy()
    one_students_df["interval_sec"] = one_students_df["timestamp"].diff().dt.total_seconds()

    valid_intervals = one_students_df["interval_sec"].dropna()

    if valid_intervals.empty:
        return "<p>No intervals above threshold to display.</p>"

    fig = px.histogram(
        x=valid_intervals,
        title="Submission Interval Distribution",
        labels={"x": "Time Interval (seconds)", "y": "Count"}
    )

    fig.update_layout(
        xaxis_title=f"Time Between Submissions (bin size: {bin_size} seconds)",
        yaxis_title="Number of Occurrences",
        bargap=0.1,
        template="plotly_white"
    )

    fig.update_traces(
        xbins=dict(
            start=valid_intervals.min(),
            end=valid_intervals.max(),
            size=bin_size
        )
    )

    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def count_submissions_in_window(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    selected_task_ids: list
) -> int:
    """
    Count the number of submissions in a given time window for selected tasks.
    
    parameters:
    - df: DataFrame with 'timestamp' and 'task_id' columns.
    - start_time: Start of the time window.
    - end_time: End of the time window.
    - selected_task_ids: List of task IDs to filter by.
    
    returns:
    - Count of submissions in the specified time window for the selected tasks.
    """
    mask = (
        (df['timestamp'] >= start_time) &
        (df['timestamp'] <= end_time) &
        (df['task_id'].isin(selected_task_ids))
    )
    return df[mask].shape[0]


if __name__ == "__main__":

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    all_submissions_df = pd.read_parquet("data/all_submissions.parquet")

    task_ids = vis_tools.get_sorted_task_ids(all_submissions_df)

    stu1_df = vis_tools.get_one_student_submissions(all_submissions_df, "4a7c189a4f8f93ceccc87c51e48a7a327bae04899ace4a161b6103c5035c1dff")
    stu2_df = vis_tools.get_one_student_submissions(all_submissions_df, "aba64fa34e052d9f2c5473f26136afa4c053f049191f40ea7e0d56708274d9a8")
    stu3_df = vis_tools.get_one_student_submissions(all_submissions_df, "6444fad4ad42f277da7e8c45468d271d12426bbbbf8ff9fa9c28abf16d2cef19")

    print(stu1_df.head())
    # print(count_submissions_in_window(
    #     stu1_df,
    #     pd.Timestamp("2022-05-03 09:03:19+00:00"),
    #     pd.Timestamp("2022-05-03 10:52:47+00:00"),
    #     ['q1_p1_s1_t5']
    # ))

    # filtered_stus = filter_submissions_less_than_seconds(pd.concat([stu1_df, stu2_df, stu3_df]), 15)


    # plot_html = plot_multiple_students_submission_sequence_plotly(filtered_stus, task_ids)

    # plot_html = plotly_one_students_submission_interval_counts(stu3_df)

    # with open("test/time_windows_between_one_student_plot.html", "w", encoding="utf-8") as f:
    #     f.write(plot_html)

    print("Analysis complete.")