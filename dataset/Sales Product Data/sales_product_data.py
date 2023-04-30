import os
import pandas as pd

# Get all the CSV files in the current directory
csv_files = [file for file in os.listdir() if file.endswith('.csv') and file.startswith('Sales_')]

# Read each CSV file, add a YYYY-mm column, and concatenate them
all_dataframes = []
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Convert the 'Order Date' column to a datetime object
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M', errors='coerce')

    # Drop rows with missing 'Order Date' values
    df = df.dropna(subset=['Order Date'])

    # Extract the month and year from the 'Order Date' column and add a YYYY-mm column
    df['YYYY-mm'] = df['Order Date'].dt.strftime('%Y-%m')

    # Check pandas version
    df['week'] = df['Order Date'].dt.isocalendar().week

    # Append the DataFrame to the list
    all_dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_df = combined_df.sort_values(by='Order Date')

# Save the concatenated DataFrame as a CSV file
combined_df.to_csv('Sales_total_2019.csv', index=False)
