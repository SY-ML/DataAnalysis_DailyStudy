import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Read CSV files and combine all the files into one data frame
path = './dataset/Sales Product Data'
csv_files = [file for file in os.listdir(path) if file.endswith('.csv') and file.startswith('Sales_')]

all_dataframes = []
for file in csv_files:
    df = pd.read_csv(os.path.join(path, file))
    df.columns = df.columns.str.strip()  # Strip spaces from column names
    all_dataframes.append(df)

combined_df = pd.concat(all_dataframes, ignore_index=True)

# Filter out the header rows
combined_df = combined_df[combined_df['Order Date'] != 'Order Date']

# Convert 'Order Date' from object to datetime
combined_df['Order Date'] = pd.to_datetime(combined_df['Order Date'], infer_datetime_format=True)

# 2. Drop empty rows
combined_df.dropna(inplace=True)

# 3. Add week of the year based on values in Order Date
combined_df['Week of the Year'] = combined_df['Order Date'].dt.isocalendar().week

# 4. Create 'Apriori Result' folder
apriori_results_path = './dataset/Apriori Result'
os.makedirs(apriori_results_path, exist_ok=True)

# 5. Perform apriori algorithm on a weekly basis and save results
weeks = combined_df['Week of the Year'].unique()
all_apriori_results = []

for week in weeks:
    weekly_data = combined_df[combined_df['Week of the Year'] == week]
    transactions = weekly_data.groupby('Order ID')['Product'].apply(list)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    rules['Week of the Year'] = week
    all_apriori_results.append(rules)
    rules.to_csv(f'{apriori_results_path}/AprioriResult_week_{week}.csv', index=False)

# 6. Read all the apriori algorithm results and combine into one data frame
apriori_files = [file for file in os.listdir(apriori_results_path) if file.endswith('.csv') and file.startswith('AprioriResult_week_')]
all_apriori_results = []

for file in apriori_files:
    df = pd.read_csv(os.path.join(apriori_results_path, file))
    all_apriori_results.append(df)

final_df = pd.concat(all_apriori_results, ignore_index=True)
