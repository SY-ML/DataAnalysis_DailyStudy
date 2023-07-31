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