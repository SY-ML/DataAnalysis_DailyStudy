import os
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx

class SalesDataset():
    def __init__(self):
        self.df = self.get_combined_data()

    def get_combined_data(self):
        # Read CSV files and combine all the files into one data frame
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

        return combined_df

class AprioriAlgorithm():
    def __init__(self):
        # Create 'Apriori Result' folder
        self.apriori_results_path = './dataset/Apriori Result'
        os.makedirs(self.apriori_results_path, exist_ok=True)

    def perform_apriori_algorithm(self, df):
        transactions = df.groupby('Order ID')['Product'].apply(list)
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

        return rules


class Plotting:
    @staticmethod
    def plot_3d_scatter(df, week_number):
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
        ax.scatter(df['antecedents_index'], df['consequents_index'], df['support'], c=df['support'], cmap='viridis',
                   s=100)

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


def main():
    ds = SalesDataset()
    apr = AprioriAlgorithm()

    combined_df = ds.df
    apriori_results_path = apr.apriori_results_path

    weeks = combined_df['Week of the Year'].unique()
    all_apriori_results = []

    for week in weeks:
        weekly_data = combined_df[combined_df['Week of the Year'] == week]
        rules = apr.perform_apriori_algorithm(weekly_data)
        rules['Week of the Year'] = week
        all_apriori_results.append(rules)
        rules.to_csv(f'{apriori_results_path}/AprioriResult_week_{week}.csv', index=False)

        # Define apriori_files
    apriori_files = [file for file in os.listdir(apriori_results_path) if
                     file.endswith('.csv') and file.startswith('AprioriResult_')]

    all_apriori_results = []

    for file in apriori_files:
        df = pd.read_csv(os.path.join(apriori_results_path, file))
        all_apriori_results.append(df)

        week_number = file.split("_")[-1].split(".")[0]
        Plotting.plot_3D_scatter(df, week_number)

    final_df = pd.concat(all_apriori_results, ignore_index=True)

    Plotting.plot_distribution(final_df)
    Plotting.plot_network_graph(final_df)

if __name__ == '__main__':
    main()