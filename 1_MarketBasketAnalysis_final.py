import os
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from matplotlib import pyplot as plt
import seaborn as sns

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

    # Read the apriori results file for the current week
    week_number = file.split("_")[-1].split(".")[0]
    df = pd.read_csv(os.path.join(apriori_results_path, file))

    # Convert antecedents and consequents to string format
    df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(list(x.strip('{}').split(', '))))
    df['consequents'] = df['consequents'].apply(lambda x: ', '.join(list(x.strip('{}').split(', '))))

    # Create a mapping of products to indices for the x and y axis
    unique_products = np.unique(np.concatenate([df['antecedents'].unique(), df['consequents'].unique()]))
    product_mapping = {product: index for index, product in enumerate(unique_products)}

    # Map the antecedents and consequents to their corresponding indices
    df['antecedents_index'] = df['antecedents'].map(product_mapping)
    df['consequents_index'] = df['consequents'].map(product_mapping)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the support values
    ax.scatter(df['antecedents_index'], df['consequents_index'], df['support'], c=df['support'], cmap='viridis', s=100)

    # Set the axis labels and title
    ax.set_xlabel('Antecedents')
    ax.set_ylabel('Consequents')
    ax.set_zlabel('Support')
    ax.set_title(f'Apriori Results for Week {week_number}')

    # Set the x and y axis ticks and labels
    ax.set_xticks(range(len(unique_products)))
    ax.set_xticklabels(unique_products, rotation=90, ha='right')
    ax.set_yticks(range(len(unique_products)))
    ax.set_yticklabels(unique_products)

    # Show the plot
    plt.show()

final_df = pd.concat(all_apriori_results, ignore_index=True)

## TODO - 결과에서 집중할 제품 추려내기


# Distribution of association rules over weeks
plt.figure(figsize=(12, 6))
sns.countplot(x='Week of the Year', data=final_df)
plt.title('Number of Association Rules by Week of the Year')
plt.show()

# Distribution of support, confidence, and lift values
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

sns.histplot(final_df['support'], kde=True, ax=axs[0])
axs[0].set_title('Support Distribution')

sns.histplot(final_df['confidence'], kde=True, ax=axs[1])
axs[1].set_title('Confidence Distribution')

sns.histplot(final_df['lift'], kde=True, ax=axs[2])
axs[2].set_title('Lift Distribution')

plt.show()

# Distribution of support, confidence, and lift values
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

sns.histplot(final_df['support'], kde=True, ax=axs[0])
axs[0].set_title('Support Distribution')

sns.histplot(final_df['confidence'], kde=True, ax=axs[1])
axs[1].set_title('Confidence Distribution')

sns.histplot(final_df['lift'], kde=True, ax=axs[2])
axs[2].set_title('Lift Distribution')

plt.show()

import networkx as nx

# Create a network graph
G = nx.DiGraph()

for index, row in final_df.iterrows():
    antecedents = tuple(row['antecedents'])
    consequents = tuple(row['consequents'])
    support = row['support']
    confidence = row['confidence']
    lift = row['lift']

    G.add_edges_from([(antecedents, consequents)], weight=lift)

# Plot the network graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', alpha=0.8)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)
plt.title('Network Graph of Association Rules')
plt.show()

