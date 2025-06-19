
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import answer_sequence_vis as vis_tools

def plot_multiple_students_submission_sequence_plotly(
    multiple_students_df: pd.DataFrame,
    sorted_task_ids: list,
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
        title='Submissions Over Time for Multiple Students',
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



if __name__ == "__main__":

    all_submissions_df = pd.read_parquet("data/all_submissions.parquet")

    task_ids = vis_tools.get_sorted_task_ids(all_submissions_df)

    stu1_df = vis_tools.get_one_student_submissions(all_submissions_df, "4a7c189a4f8f93ceccc87c51e48a7a327bae04899ace4a161b6103c5035c1dff")
    stu2_df = vis_tools.get_one_student_submissions(all_submissions_df, "aba64fa34e052d9f2c5473f26136afa4c053f049191f40ea7e0d56708274d9a8")
    stu3_df = vis_tools.get_one_student_submissions(all_submissions_df, "6444fad4ad42f277da7e8c45468d271d12426bbbbf8ff9fa9c28abf16d2cef19")

    plot_html = plot_multiple_students_submission_sequence_plotly(pd.concat([stu1_df, stu2_df, stu3_df]), task_ids)

    with open("test/test_plot.html", "w", encoding="utf-8") as f:
        f.write(plot_html)

    print("Analysis complete.")