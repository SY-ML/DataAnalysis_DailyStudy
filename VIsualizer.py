import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Define the data
df = pd.DataFrame([
    dict(UserID='User 1', Activity='Activity 1', Start='2023-01-01', Finish='2023-01-02'),
    dict(UserID='User 1', Activity='Activity 2', Start='2023-01-02', Finish='2023-01-03'),
    dict(UserID='User 2', Activity='Activity 1', Start='2023-01-03', Finish='2023-01-04'),
    dict(UserID='User 2', Activity='Activity 2', Start='2023-01-04', Finish='2023-01-05'),
])

def visualize_in_gantt_chart(df, col_xstart, col_xend, col_y, col_val):
    # Create a Gantt chart
    fig = px.timeline(df, x_start=col_xstart, x_end=col_xend, y=col_y, color= col_val)
    fig.show()


visualize_in_gantt_chart(df, 'Start', 'Finish', 'UserID', 'Activity')

# Create the subplots
fig = make_subplots(rows=2, cols=1)

# Add the overall Gantt chart
fig.add_trace(
    go.Bar(x=df['Start'], y=df['Activity'], name='Overall'),
    row=1, col=1
)

# Add the individual user timelines
for user in df['UserID'].unique():
    df_user = df[df['UserID'] == user]
    fig.add_trace(
        go.Bar(x=df_user['Start'], y=df_user['Activity'], name=user),
        row=2, col=1
    )

fig.update_layout(height=600, width=800, title_text="Overall and User Timelines")
fig.show()


def visualize_in_boxplot(df, col):
    return None