import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysis1:
    def __init__(self, df):
        self.df = df
        self.column_types = self.list_column_types()

    def list_column_types(self):
        column_types = {}
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                unique_values = self.df[column].nunique()
                column_types[column] = 'numeric ordinal' if unique_values < 20 else 'interval'
            else:
                column_types[column] = 'non-numeric ordinal' if self.df[column].nunique() < 20 else 'nominal'
        return column_types

    def select_variable(self, data_type, allow_skip=False):
        available_vars = [col for col, col_type in self.column_types.items() if col_type == data_type]
        if allow_skip and not available_vars:
            return None
        print(f"Available {data_type} variables: {available_vars}")
        while True:
            selected_var = input(f"Please select a {data_type} variable: ")
            if selected_var in available_vars:
                return selected_var
            print("Invalid choice. Please try again.")

    def plot_qq_histogram(self, data, title):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sm.qqplot(data, line='s', ax=axes[0])
        axes[0].set_title(f"Q-Q Plot of {title}")
        sns.histplot(data, kde=True, ax=axes[1])
        axes[1].set_title(f"Histogram of {title}")
        plt.tight_layout()
        plt.show()

    def check_normality(self, data, size_limit=2000):
        data = data.dropna()
        if len(data) <= size_limit:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk Test"
        else:
            stat, p_value = stats.anderson(data, dist='norm')[:2]
            test_name = "Anderson-Darling Test"
        print(f"{test_name}: Statistic={stat}, p-value={p_value}")
        return stat, p_value

    def check_skewness(self, data):
        skewness = stats.skew(data.dropna())
        print(f"Skewness: {skewness}")
        return abs(skewness) > 1

    def hypothesis_test(self, continuous_var, categorical_var, skewed, null_hyp):
        data = self.df[[continuous_var, categorical_var]].dropna()
        groups = [group[continuous_var].values for name, group in data.groupby(categorical_var)]

        if skewed:
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis Test"
        else:
            stat, p_value = stats.f_oneway(*groups)
            test_name = "ANOVA"

        print(f"{test_name}: Statistic={stat}, p-value={p_value}")
        return stat, p_value
