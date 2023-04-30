import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class SY_Apiriori():
    def __init__(self):
        self.df = self.load_dataset()
        # self.generate_apiriori_result_by_mlxtend(df = self.df)
        self.result_mlxtend = pd.read_csv("1_MarketBasketAnalysis_Apiriori_by_mlxtend.csv")
    def load_dataset(self):
        df = pd.read_csv("./dataset/Sales Product Data/Sales_April_2019.csv")
        df = df[['Order ID', 'Product', 'Quantity Ordered']]
        df = df.dropna(axis=0)
        return df
    def generate_apiriori_result_by_mlxtend(self, df):
        # df = self.df.copy()
        # Group by 'Order ID' and aggregate 'Product' as lists
        transaction_groups = df.groupby('Order ID')['Product'].apply(list)

        # Transform transactions into a one-hot encoded DataFrame
        te = TransactionEncoder()
        te_array = te.fit_transform(transaction_groups)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        # Find frequent itemsets using the Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=0.0001, use_colnames=True)

        # Generate association rules
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.0001)
        rules_sorted = rules.sort_values(by=['antecedent support'], ascending=False)

        # Print rules sorted by lift
        rules_sorted.to_csv("./1_MarketBasketAnalysis_Apiriori_by_mlxtend.csv", index=False)

    def filter_orders_containing_any_of_given_items(self, ls_items_to_find):
        """
        
        :param ls_items_to_find: list with up to two elements 
        :return: 
        """
        df = self.df.copy()
        df = df[df['Product'].isin(ls_items_to_find)]

        return df

    def filter_orders_containing_all_of_given_items(self, ls_items_to_find):
        # df_filtered = self.filter_orders_containing_any_of_given_items(ls_items_to_find)

        df = self.df

        def contains_all_items(group):
            return all(item in group['Product'].values for item in ls_items_to_find)

        # Filter rows containing all the items in column 'Product'
        filtered_groups = df[df['Product'].isin(ls_items_to_find)].groupby('Order ID').apply(
            lambda x: x if contains_all_items(x) else None)
        filtered_df = pd.concat([group for group in filtered_groups if isinstance(group, pd.DataFrame)],
                                ignore_index=True)

        print(filtered_df)

        return filtered_groups



    def get_support_of_selected_item(self, ls_items_to_find):
        count_allOrds = self.df['Order ID'].nunique()

        df_filteredOrds = self.filter_orders_containing_any_of_given_items(ls_items_to_find)
        count_filteredOrders = df_filteredOrds['Order ID'].nunique()

        return count_filteredOrders/count_allOrds



# Get all the CSV files in the current directory
relative_path = os.path.join('dataset', 'Sales Product Data')
csv_files = [file for file in os.listdir(relative_path) if file.endswith('.csv') and file.startswith('Sales_')]
print(csv_files)
exit()
# Read each CSV file, add a YYYY-mm column, and concatenate them
all_dataframes = []
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file)

    # Extract the month name from the file name and add a YYYY-mm column
    month_name = file.split('_')[1]
    df['YYYY-mm'] = f'2019-{pd.to_datetime(month_name, format="%B").month:02d}'

    print(df)

sy = SY_Apiriori()
df = sy.df

a = sy.filter_orders_containing_all_of_given_items(['Lightning Charging Cable', 'Wired Headphones'])
print(a)


# TODOs- Get support by month, visualization, Validation
