import os
import pandas as pd

# Get all the CSV files in the current directory
csv_files = [file for file in os.listdir() if file.endswith('.csv') and file.startswith('Sales_')]

# Read each CSV file, add a YYYY-mm column, and concatenate them
all_dataframes = []
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Extract the month name from the file name and add a YYYY-mm column
    month_name = file.split('_')[1]
    df['YYYY-mm'] = f'2019-{pd.to_datetime(month_name, format="%B").month:02d}'

    # Append the DataFrame to the list
    all_dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(all_dataframes, ignore_index=True)
combined_df = combined_df.sort_values(by='YYYY-mm')

# Save the concatenated DataFrame as a CSV file
combined_df.to_csv('Sales_total_2019.csv', index=False)
