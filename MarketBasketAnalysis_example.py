import pandas as pd
from MarketBasketAnaylsis import MarketBasketAnalysis

if __name__ == '__main__':
    # Read dataset
    df = pd.read_csv("./Dataset/Sales Product Data/Sales_total_2019.csv")

    # Initiate the market basket analysis class
    mba = MarketBasketAnalysis(df=df,
                               col_order='Order ID',
                               col_sku='Product',
                               col_date='Order Date',
                               min_order=120,
                               cumulative_pct=80,
                               quantity_wise=False)

    # Result of analysis
    result = mba.perform_apriori_algorithm(df)
    print(result)


#
#
# # Define the SalesDataset class
# class SalesDataset:
#     def __init__(self):
#         self.path = './dataset/Sales Product Data'
#         self.df = self.get_combined_data()
#         self.ls_date = self.df['Order Date'].unique().tolist()
#         # self.ls_df_basis = [self.add_basis_column(basis) for basis in ['weekly', 'bi-weekly', 'monthly', 'quarterly', 'semi-yearly']]
#         # self.df_weekly, self.df_biweekly, self.df_monthly, self.df_quarterly, self.df_semiyearly = self.ls_df_basis
#
#     def get_combined_data(self):
#         # Define the path for the sales product data
#         path = self.path
#         csv_files = [file for file in os.listdir(path) if file.endswith('.csv') and file.startswith('Sales_')]
#
#         all_dataframes = []
#
#         # Read all CSV files and store them in a list
#         for file in csv_files:
#             df = pd.read_csv(os.path.join(path, file))
#             df.columns = df.columns.str.strip()  # Strip spaces from column names
#             all_dataframes.append(df)
#
#         # Combine all dataframes into one
#         combined_df = pd.concat(all_dataframes, ignore_index=True)
#         combined_df = combined_df[combined_df['Order Date'] != 'Order Date']
#         combined_df['Order Date'] = pd.to_datetime(combined_df['Order Date'], infer_datetime_format=True).dt.date
#         combined_df.dropna(inplace=True)
#         filtered_df = combined_df[['Order ID', 'Order Date', 'Product']]
#
#         # Add week of the year based on values in Order Date
#         return filtered_df
#
#     def time_columns_by_date(self):
#         df = pd.DataFrame(self.df['Order Date'].unique(), columns = ['Date'])
#         df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#         df['Day of the week'] = df['Date'].dt.dayofweek
#         df['Week of the year'] = df['Date'].dt.isocalendar().week
#         df['Biweekly'] = (df['Date'].dt.isocalendar().week+1) // 2
#         df['Month'] = df['Date'].dt.month
#         df['Quarter'] = df['Date'].dt.quarter
#         df['Half'] = (df['Date'].dt.quarter + 1) // 2
#         df['Year'] = df['Date'].dt.year
#
#         # Sort data by date
#         df = df.sort_values(by='Date')
#
#         path = f'{self.path}/timedata'
#         os.makedirs(path, exist_ok=True)
#         df.to_csv(f'{path}/time data by date.csv', index=False)
#
#         return df
#
#
#
#
#     def add_basis_column(self, basis):
#         df = self.df.copy()
#         conditions = [
#             basis == 'weekly',
#             basis == 'bi-weekly',
#             basis == 'monthly',
#             basis == 'quarterly',
#             basis == 'semi-yearly'
#         ]
#
#         choices = [
#             df['Order Date'].dt.strftime('%YW%U'),
#             df['Order Date'].dt.year.astype(str) + 'BW' + ((df['Order Date'].dt.weekofyear + 1) // 2).astype(str),
#             df['Order Date'].dt.strftime('%Y%m'),
#             df['Order Date'].dt.to_period('Q').astype(str),
#             df['Order Date'].dt.year.astype(str) + 'H' + ((df['Order Date'].dt.quarter + 1) // 2).astype(str)
#         ]
#
#         df['Basis'] = np.select(conditions, choices, default=None)
#
#         if df['Basis'].isnull().any():
#             raise ValueError(
#                 "Invalid basis. Accepted values: 'weekly', 'bi-weekly', 'monthly', 'quarterly', 'semi-yearly'")
#         return df
#
# d = SalesDataset()
# print(d.df['Order Date'])
# print(d.ls_date)
# print(d.time_columns_by_date())
# print(d.time_columns_by_date().dtypes)
# exit()
# for df in d.ls_df_basis:
#     print(df[['Order Date', 'Basis']].head())
#     print(df[['Order Date', 'Basis']].tail())


#
#
# # Define the AprioriAlgorithm class
# class AprioriAlgorithm:
#     def __init__(self):
#         self.apriori_results_path = './dataset/Apriori Result'
#         os.makedirs(self.apriori_results_path, exist_ok=True)
#
#     def perform_apriori_algorithm(self):
#         transactions = df.groupby('Order ID')['Product'].apply(list)
#         te = TransactionEncoder()
#         te_ary = te.fit(transactions).transform(transactions)
#         transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
#
#         # Calculate frequent itemsets and association rules
#         frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
#         rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
#         return rules
#
#
# # Define the Plotting class
# class Plotting:
#     @staticmethod
#     def plot_3d_scatter(df, week_number):
#         df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(list(x.strip('{}').split(', '))))
#         df['consequents'] = df['consequents'].apply(lambda x: ', '.join(list(x.strip('{}').split(', '))))
#
#         unique_products = np.unique(np.concatenate([df['antecedents'].unique(), df['consequents'].unique()]))
#         product_mapping = {product: index for index, product in enumerate(unique_products)}
#
#         df['antecedents_index'] = df['antecedents'].map(product_mapping)
#         df['consequents_index'] = df['consequents'].map(product_mapping)
#
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(111, projection='3d')
#
#         ax.scatter(df['antecedents_index'], df['consequents_index'], df['support'], c=df['support'], cmap='viridis',
#                    s=100)
#
#         ax.set_xlabel('Antecedents')
#         ax.set_ylabel('Consequents')
#
#         ax.set_zlabel('Support')
#         ax.set_title(f'Apriori Results for Week {week_number}')
#
#         # Set the x and y axis ticks and labels
#         ax.set_xticks(range(len(unique_products)))
#         ax.set_xticklabels(unique_products, rotation=90, ha='right')
#         ax.set_yticks(range(len(unique_products)))
#         ax.set_yticklabels(unique_products)
#
#         # Show the plot
#         plt.show()
#
# def main():
#     ds = SalesDataset()
#     print(ds.add_basis_column(basis='weekly'))
#     exit()
#     apr = AprioriAlgorithm()
#
#     combined_df = ds.df
#     apriori_results_path = apr.apriori_results_path
#
#     frequent_skus_path = './dataset/Weekly top 80 frequent Skus data'
#     os.makedirs(frequent_skus_path, exist_ok=True)
#
#     weeks = combined_df['Week of the Year'].unique()
#     all_apriori_results = []
#
#     for week in weeks:
#         weekly_data = combined_df[combined_df['Week of the Year'] == week]
#
#         product_count = weekly_data.groupby('Product')['Order ID'].nunique().reset_index()
#         product_count.columns = ['Product', 'Count']
#         product_count['Proportion'] = product_count['Count'] / weekly_data['Order ID'].nunique()
#
#         product_count_sorted = product_count.sort_values(by='Count', ascending=False)
#         product_count_sorted['Cumulative Proportion'] = product_count_sorted['Proportion'].cumsum()
#         filtered_products = product_count_sorted[product_count_sorted['Cumulative Proportion'] <= 0.8]
#
#         filtered_weekly_data = weekly_data[weekly_data['Product'].isin(filtered_products['Product'])]
#
#         # Save the filtered_weekly_data to the new folder
#         filtered_weekly_data.to_csv(f'{frequent_skus_path}/High 80 pct Frequent Sku Data_{week}.csv', index=False)
#
#         rules = apr.perform_apriori_algorithm(filtered_weekly_data)
#         rules['Week of the Year'] = week
#         all_apriori_results.append(rules)
#         rules.to_csv(f'{apriori_results_path}/AprioriResult_week_{week}.csv', index=False)
#
#     apriori_files = [file for file in os.listdir(apriori_results_path) if
#                      file.endswith('.csv') and file.startswith('AprioriResult_')]
#
#     all_apriori_results = []
#
#     for file in apriori_files:
#         df = pd.read_csv(os.path.join(apriori_results_path, file))
#         all_apriori_results.append(df)
#
#         week_number = file.split("_")[-1].split(".")[0]
#         Plotting.plot_3d_scatter(df, week_number)
#
# def plot_sku_order_count(df):
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='Product', y='Count', data=df)
#     plt.xticks(rotation=90)
#     plt.xlabel('SKU')
#     plt.ylabel('Order Count')
#     plt.title('Top 80% Frequent SKUs by Order Count')
#     plt.show()
#
# def combine_and_plot_frequent_skus(frequent_skus_path):
#     frequent_sku_files = [file for file in os.listdir(frequent_skus_path) if
#                           file.endswith('.csv') and file.startswith('High 80 pct Frequent Sku Data_')]
#
#     all_frequent_sku_data = []
#
#     for file in frequent_sku_files:
#         df = pd.read_csv(os.path.join(frequent_skus_path, file))
#         all_frequent_sku_data.append(df)
#
#     combined_frequent_sku_data = pd.concat(all_frequent_sku_data, ignore_index=True)
#
#     # Calculate product count of order ID
#     product_count = combined_frequent_sku_data.groupby('Product')['Order ID'].nunique().reset_index()
#     product_count.columns = ['Product', 'Count']
#     combined_frequent_sku_data.groupby('Product')['Order ID'].nunique().reset_index()
#     product_count.columns = ['Product', 'Count']
#
#     # Sort by order count in descending order
#     product_count_sorted = product_count.sort_values(by='Count', ascending=False)
#
#     # Plot the SKUs by their order count
#     plot_sku_order_count(product_count_sorted)
#
# if __name__ == '__main__':
#     main()
#     frequent_skus_path = './dataset/Weekly top 80 frequent Skus data'
#     combine_and_plot_frequent_skus(frequent_skus_path)
