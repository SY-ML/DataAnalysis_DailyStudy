import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Define the data

def load_data_sample():
    df = pd.DataFrame([
        dict(UserID='User 1', Activity='Activity 1', Start='2023-01-01', Finish='2023-01-02'),
        dict(UserID='User 1', Activity='Activity 2', Start='2023-01-02', Finish='2023-01-03'),
        dict(UserID='User 2', Activity='Activity 1', Start='2023-01-03', Finish='2023-01-04'),
        dict(UserID='User 2', Activity='Activity 2', Start='2023-01-04', Finish='2023-01-05'),
    ])
    return df


def visualize_in_gantt_chart(df, col_xstart, col_xend, col_y, col_val):
    # Create a Gantt chart
    fig = px.timeline(df, x_start=col_xstart, x_end=col_xend, y=col_y, color= col_val)
    fig.show()

df = pd.read_csv('./dataset/Customer Shopping Dataset - Retail Sales Data/customer_shopping_data.csv')
df['year'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y').dt.year
def visualize_in_kdeplot(df, col_x, title, col_hue=None):

    plt.figure(figsize = (10, 6))

    if col_hue == None:
        sns.kdeplot(df[col_x])
    else:
        for by_val in df[col_hue].unique():
            # sns.histplot(df[df[col_hue] == by_val][col_x], label = by_val)
            sns.kdeplot(df[df[col_hue] == by_val][col_x], label = by_val)
            plt.legend(title = col_x)

    plt.title(title)
    plt.show()

visualize_in_kdeplot(df = df, col_x= 'quantity', title='Quantity Distribution by year', col_hue='year')


def late_clock_out_report(df):
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    # Assume the dataframe df is already in scope
    # and 'date', 'user_id', 'last_clockout_time' are the relevant columns

    df['hour'] = df['last_clockout_time'].dt.hour + df['last_clockout_time'].dt.minute / 60
    df['wasted_time'] = 18 - df['hour']

    # Create cumulative wasted time graph
    df_cumulative = df.groupby('date')['wasted_time'].sum().cumsum().reset_index()
    cumulative_wasted_time = go.Scatter(x=df_cumulative['date'], y=df_cumulative['wasted_time'], mode='lines',
                                        name='Cumulative Wasted Time')

    # Create average wasted time per operator graph
    df_operator = df.groupby('user_id')['wasted_time'].mean().reset_index()
    average_wasted_time = go.Bar(x=df_operator['user_id'], y=df_operator['wasted_time'], name='Average Wasted Time')

    # Create histogram of clockout times
    histogram_end_times = go.Histogram(x=df['hour'], nbinsx=24, name='Histogram of Clock Out Times')

    # Add all plots to a subplot
    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(cumulative_wasted_time, row=1, col=1)
    fig.add_trace(average_wasted_time, row=2, col=1)
    fig.add_trace(histogram_end_times, row=3, col=1)

    # Update layout for better viewing
    fig.update_layout(height=900, width=1200, title_text="Time Wasted Analysis", showlegend=True)

    # Export to html
    pio.write_html(fig, 'time_wasted_analysis.html')
