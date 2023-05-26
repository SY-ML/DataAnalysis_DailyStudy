import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import statistics
# poff
#off today
# https://datarian.io/blog/what-is-rfm
class RFMAnalysis():
    def __init__(self, df, col_date, col_customer, col_order, col_qty, col_price):
        self._df = df
        self._snapshot_date = df['Order Date'].max() + timedelta(days=1) # latest date in the data set plus one day

        self._col_date = col_date
        self._col_customer = col_customer
        self._col_order = col_order
        self._col_qty = col_qty
        self._col_price = col_price

        self._ls_cols_in_use = [self._col_date, self._col_customer, self._col_order, self._col_qty, self._col_price]

        self.df_prcd = self.preprocess_dataset()

    def preprocess_dataset(self):
        df = self._df[self._ls_cols_in_use].copy()

        # Convert the date column to datetime format
        df[self._col_date] = pd.to_datetime(df[self._col_date], infer_datetime_format=True, dayfirst=True)

        # Drop NA values. Consider specifying a subset of columns if not all columns are equally important.
        df.dropna(inplace=True)

        # Drop duplicates. Consider specifying a subset of columns if not all columns are equally important.
        df.drop_duplicates(inplace=True)

        df['sales'] = df[self._col_price] * df[self._col_qty]
        df.drop(columns = [self._col_price, self._col_qty])

        #TODO- start here
        df_RFM = df.groupby([self._col_customer]).agg({
            self._col_date: lambda x: (self._snapshot_date - x.max()).days,
            self._: 'count',
            'Price Each': 'sum'

        return df


df = pd.read_csv('./dataset/Customer Shopping Dataset - Retail Sales Data/customer_shopping_data.csv')
print(df.columns)
rfm = RFMAnalysis(df= df,
                  col_date= 'invoice_date',
                  col_order='invoice_no',
                  col_customer='customer_id',
                  col_price= 'price',
                  col_qty='quantity')

print(rfm.df_prcd)


# # Assuming df is your DataFrame and "Order Date" is in MM/DD/YY H:M format
# df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y %H:%M')
#
# # Calculate Recency, Frequency and Monetary value for each customer
# snapshot_date = df['Order Date'].max() + timedelta(days=1) # latest date in the data set plus one day
# df_RFM = df.groupby(['Order ID']).agg({
#     'Order Date': lambda x: (snapshot_date - x.max()).days,
#     'Product': 'count',
#     'Price Each': 'sum'
# })
#
#
# # Rename the columns to Recency, Frequency and Monetary
# df_RFM.rename(columns = {'Order Date': 'Recency',
#                          'Product': 'Frequency',
#                          'Price Each': 'Monetary'}, inplace=True)
#
#
# # Scaling is very important in KMeans
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# df_RFM_normalized = scaler.fit_transform(df_RFM)
#
# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=3, random_state=0).fit(df_RFM_normalized)
#
# # Append the KMeans clustering result back to the RFM data frame
# df_RFM['Cluster'] = kmeans.labels_
#
# print(df_RFM['Cluster'].value_counts())
# print(df_RFM)
# # Prepare your figure and axes
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Prepare a color map
# cmap = plt.cm.get_cmap("viridis")
#
# # Create a scatter plot
# sc = ax.scatter(df_RFM['Recency'], df_RFM['Frequency'], df_RFM['Monetary'], s=50, c=df_RFM['Cluster'], cmap=cmap)
#
# # Create a colorbar
# cb = plt.colorbar(sc, ax=ax)
#
# # Set colorbar title
# cb.set_label('Cluster')
#
# ax.set_xlabel('Recency')
# ax.set_ylabel('Frequency')
# ax.set_zlabel('Monetary')
# plt.title('3D scatter plot of RFM data')
# plt.show()
