import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("./dataset/Sales Product Data/Sales_April_2019.csv")
df = df[['Order ID', 'Product', 'Quantity Ordered']]
df = df.dropna(axis=0)

# Group by 'Order ID' and aggregate 'Product' as lists
transaction_groups = df.groupby('Order ID')['Product'].apply(list)
print(transaction_groups)


# Transform transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit_transform(transaction_groups)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.01)

# Print rules sorted by lift
print(rules.sort_values(by='lift', ascending=False))