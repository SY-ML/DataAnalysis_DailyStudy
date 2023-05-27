import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import statistics


# https://datarian.io/blog/what-is-rfm
class RFMAnalysis():
    def __init__(self, df, col_date, col_customer, col_product, col_order, col_qty, col_price):
        self._df = df
        self._snapshot_date = df[col_date].max() + timedelta(days=1)

        self._col_date = col_date
        self._col_customer = col_customer
        self._col_order = col_order
        self._col_product = col_product
        self._col_qty = col_qty
        self._col_price = col_price

        self._ls_cols_in_use = [self._col_date, self._col_customer, self._col_order, self._col_qty, self._col_price]

        self.df_prcd = self.preprocess_dataset()

    def preprocess_dataset(self):
        df = self._df[self._ls_cols_in_use].copy()

        df[self._col_date] = pd.to_datetime(df[self._col_date], infer_datetime_format=True, dayfirst=True)

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        df['sales'] = df[self._col_price] * df[self._col_qty]
        df.drop(columns=[self._col_price, self._col_qty])

        df_RFM = df.groupby([self._col_customer]).agg({
            self._col_date: lambda x: (self._snapshot_date - x.max()).days,
            self._col_product: 'nunique',  # Number of unique products per customer
            'sales': 'sum'})

        df_RFM.rename(columns={self._col_date: 'Recency', self._col_product: 'Frequency', 'sales': 'Monetary'},
                      inplace=True)

        return df_RFM

    def find_optimal_clusters(self, data, max_k):
        iters = range(2, max_k + 1)

        sse = []
        for k in iters:
            sse.append(KMeans(n_clusters=k, random_state=0).fit(data).inertia_)

        p1 = np.array([2, sse[0]])
        p2 = np.array([max_k, sse[-1]])

        distances = []
        for i, sse_i in enumerate(sse):
            p = np.array([i + 2, sse_i])
            distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            distances.append(distance)

        optimal_k = distances.index(max(distances)) + 2

        return optimal_k

    def run_kmeans(self, max_k=10):
        df_RFM = self.df_prcd

        scaler = StandardScaler()
        df_RFM_normalized = scaler.fit_transform(df_RFM)

        optimal_k = self.find_optimal_clusters(df_RFM_normalized, max_k)

        kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(df_RFM_normalized)

        df_RFM['Cluster'] = kmeans.labels_

        return df_RFM

    def visualize_clusters(self):
        df_RFM = self.df_prcd
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = plt.cm.get_cmap("viridis")

        sc = ax.scatter(df_RFM['Recency'], df_RFM['Frequency'], df_RFM['Monetary'], s=50, c=df_RFM['Cluster'],
                        cmap=cmap)

        cb = plt.colorbar(sc, ax=ax)

        cb.set_label('Cluster')

        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary')
        plt.title('3D scatter plot of RFM data')
        plt.show()

