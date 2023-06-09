import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import statistics

df = pd.read_csv('./dataset/Sales Product Data/Sales_April_2019.csv')
df = df[df['Order Date'] != 'Order Date']
df.dropna(inplace=True)

def fix_double_period(val):
    if isinstance(val, str):  # only apply the operation to string values
        parts = val.split('.')
        if len(parts) > 2:
            return parts[0] + '.' + ''.join(parts[1:])
    return val

df['Price Each'] = df['Price Each'].apply(fix_double_period).astype(float)

from collections import Counter

# Create a list of all words in the 'Product' column
print(df['Product'])
all_words = df['Product'].str.split()
# all_words = df['Product'].str.split(expand=True).stack()
print(all_words)
print(type(all_words))
# Count the frequency of each word
word_freq = Counter(all_words)

# Print the most common words and their frequencies
print(word_freq.most_common())


# Assuming df is your DataFrame and "Order Date" is in MM/DD/YY H:M format
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M')

# Calculate Recency, Frequency and Monetary value for each customer
snapshot_date = df['Order Date'].max() + timedelta(days=1) # latest date in the data set plus one day
df_RFM = df.groupby(['Order ID']).agg({
    'Order Date': lambda x: (snapshot_date - x.max()).days,
    'Product': 'count',
    'Price Each': 'sum'
})


# Rename the columns to Recency, Frequency and Monetary
df_RFM.rename(columns = {'Order Date': 'Recency',
                         'Product': 'Frequency',
                         'Price Each': 'Monetary'}, inplace=True)


# Scaling is very important in KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_RFM_normalized = scaler.fit_transform(df_RFM)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_RFM_normalized)

# Append the KMeans clustering result back to the RFM data frame
df_RFM['Cluster'] = kmeans.labels_

print(df_RFM['Cluster'].value_counts())
print(df_RFM)
# Prepare your figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare a color map
cmap = plt.cm.get_cmap("viridis")

# Create a scatter plot
sc = ax.scatter(df_RFM['Recency'], df_RFM['Frequency'], df_RFM['Monetary'], s=50, c=df_RFM['Cluster'], cmap=cmap)

# Create a colorbar
cb = plt.colorbar(sc, ax=ax)

# Set colorbar title
cb.set_label('Cluster')

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title('3D scatter plot of RFM data')
plt.show()
