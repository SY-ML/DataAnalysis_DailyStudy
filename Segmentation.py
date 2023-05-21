import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans

df = pd.read_csv('./dataset/Sales Product Data/Sales_August_2019.csv')

# Assuming df is your DataFrame and "Order Date" is in MM/DD/YY H:M format
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M')

print(df)
print(df.dtypes)
exit()
# Calculate Recency, Frequency and Monetary value for each customer
snapshot_date = df['Order Date'].max() + datetime.timedelta(days=1) # latest date in the data set plus one day
df_RFM = df.groupby(['Order ID']).agg({
    'Order Date': lambda x: (snapshot_date - x.max()).days,
    'Product': 'count',
    'Price Each': 'sum'
})

# Rename columns
df_RFM.rename(columns = {'Order Date': 'Recency',
                         'Product': 'Frequency',
                         'Price Each': 'MonetaryValue'}, inplace=True)

# Scaling is very important in KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_RFM_normalized = scaler.fit_transform(df_RFM)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_RFM_normalized)

# Append the KMeans clustering result back to the RFM data frame
df_RFM['Cluster'] = kmeans.labels_
