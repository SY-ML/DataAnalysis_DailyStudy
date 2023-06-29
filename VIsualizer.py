import plotly.express as px
import pandas as pd

# Define the data
df = pd.DataFrame([
    dict(UserID='User 1', Activity='Activity 1', Start='2023-01-01', Finish='2023-01-02'),
    dict(UserID='User 1', Activity='Activity 2', Start='2023-01-02', Finish='2023-01-03'),
    dict(UserID='User 2', Activity='Activity 1', Start='2023-01-03', Finish='2023-01-04'),
    dict(UserID='User 2', Activity='Activity 2', Start='2023-01-04', Finish='2023-01-05'),
])

# Create a Gantt chart
fig = px.timeline(df, x_start='Start', x_end='Finish', y='UserID', color='Activity')
fig.show()
