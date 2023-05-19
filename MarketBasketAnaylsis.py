import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class MarketBasketAnalysis():
    def __init__(self, df, col_order, col_sku, col_date, col_qty,min_order, cumulative_pct, effective_by_qty=False):
        """

        :param df: dataset
        :param col_order: colum name of order information
        :param col_sku: colum name of product information
        :param col_date: column name of date information
        :param min_order: minimum number of orders in which a pair of products should appear
        :param cumulative_pct: percentage used to determine the range of effective products by appearance in descending order
        """
        # Preprocess the data
        self._df = self._preprocess_dataset(df, col_date)
        # Set class variables
        self._col_order = col_order
        self._col_sku = col_sku
        self._col_date = col_date
        self._col_qty = col_qty
        self._cumulative_pct = cumulative_pct
        self._quantity_wise = effective_by_qty
        # Count the unique number of orders
        self._nunique_ord = self._df[self._col_order].nunique()
        self._sum_qty = self._df[self._col_qty].sum()

        # Calculate the support threshold for apriori
        self._min_support = min_order / self._nunique_ord
        # Calculate the minimum support threshold for effective SKUs
        self._min_thold = self._get_min_probability_of_effective_sku()

    def _preprocess_dataset(self, df, col_date):
        """
        Preprocess the dataset.
        :param df: The dataset
        :param col_date: The column name for date information
        :return: Preprocessed DataFrame
        """
        # Load necessary columns
        df = df[[self._col_date, self._col_order, self._col_sku, self._col_qty]]

        # Convert date column to datetime format
        df[col_date] = pd.to_datetime(df[col_date]).dt.date
        # Remove rows with missing data
        df.dropna(inplace=True)
        return df

    def _get_min_probability_of_effective_sku(self):
        """
        Calculate the minimum probability for effective SKUs.
        :return: Minimum probability
        """

        ### Define effective skus

        total = self._sum_qty if self._quantity_wise else self._nunique_ord

        # quantity wise
        if self._quantity_wise is True:
            # Group by SKU and count the unique number of orders
            grp = self._df.groupby(self._col_sku)[self._col_qty].sum()
            # Sort the group in descending order
            grp_sorted = grp.sort_values(ascending=False)
            # Calculate the cumulative sum and convert to percentage
            grp_sorted_cumsum = grp_sorted.cumsum() / total * 100
        # order apperance wise
        else:
            # Group by SKU and count the unique number of orders
            grp = self._df.groupby(self._col_sku)[self._col_order].nunique()
            # Sort the group in descending order
            grp_sorted = grp.sort_values(ascending=False)
            # Calculate the cumulative sum and convert to percentage
            grp_sorted_cumsum = grp_sorted.cumsum() / total * 100

        # Get the SKUs that account for a certain percentage of the total number of orders
        top_skus = grp_sorted[grp_sorted_cumsum <= self._cumulative_pct]
        # Calculate the probability of these SKUs
        top_skus_probability = top_skus / total
        # Get the minimum probability
        min_probability = top_skus_probability.min()

        return min_probability

    def perform_apriori_algorithm(self, df):
        """
        Perform the Apriori algorithm to find frequent itemsets and association rules.
        :param df: The dataset
        :return: Sorted rules
        """

        # Group by order and apply list to SKU, creating a list of products for each order
        transactions = df.groupby(self._col_order)[self._col_sku].apply(list)

        # Initialize TransactionEncoder, which transforms transaction data into a boolean matrix
        te = TransactionEncoder()

        # Fit and transform the data to a boolean matrix, where each column corresponds to a unique item in the transactions
        te_ary = te.fit(transactions).transform(transactions)

        # Convert the array into a DataFrame
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

        # Perform the Apriori algorithm to find frequent itemsets, given the minimum support threshold
        frequent_itemsets = apriori(transaction_df, min_support=self._min_support, use_colnames=True)


        # Generate association rules from the frequent itemsets, specifying 'antecedent support' as the metric and using the defined minimum threshold
        rules = association_rules(frequent_itemsets, metric='antecedent support', min_threshold=self._min_thold)

        # Convert the lists in the antecedents and consequents columns to strings
        for col in ['antecedents', 'consequents']:
            rules[col] = rules[col].apply(lambda x: ', '.join(x))

        # Sort the generated rules by 'antecedent support' in descending order
        rules_sorted = rules.sort_values(by=['antecedent support'], ascending=False)

        return rules_sorted
