import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class SY_Apiriori():
    def __init__(self):
        self.df = self.load_dataset()
        # self.generate_apiriori_result_by_mlxtend()
        self.result_mlxtend = pd.read_csv("1_MarketBasketAnalysis_Apiriori_by_mlxtend.csv")
    def load_dataset(self):
        df = pd.read_csv("./dataset/Sales Product Data/Sales_April_2019.csv")
        df = df[['Order ID', 'Product', 'Quantity Ordered']]
        df = df.dropna(axis=0)
        return df
    def generate_apiriori_result_by_mlxtend(self):
        df = self.df.copy()
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

    # TODO
    def filter_orders_containing_all_of_given_items(self, ls_items_to_find):
        df_filtered = self.filter_orders_containing_any_of_given_items(ls_items_to_find)

        grp = df_filtered.groupby(['Product'])
        # Filter rows containing all the values in column 'B'
        grouped = df[df['B'].isin(values_to_filter)].groupby('A')
        filtered_df = pd.concat(
            [group for _, group in grouped if all(value in group['B'].values for value in values_to_filter)])

        return df_filtered



    def get_support_of_selected_item(self, ls_items_to_find):
        count_allOrds = self.df['Order ID'].nunique()

        df_filteredOrds = self.filter_orders_containing_any_of_given_items(ls_items_to_find)
        count_filteredOrders = df_filteredOrds['Order ID'].nunique()

        return count_filteredOrders/count_allOrds


sy = SY_Apiriori()
df = sy.df

a = sy.filter_orders_containing_all_of_given_items(['Lightning Charging Cable', 'Wired Headphones'])
print(a)

# a = sy.get_support_of_selected_item(['Lightning Charging Cable'])
# b = sy.get_support_of_selected_item(['Wired Headphones'])
# c = sy.get_support_of_selected_item(['Lightning Charging Cable', 'Wired Headphones'])
# d = a*b
#
# print(a, b, c, d)
#
#
#
#

