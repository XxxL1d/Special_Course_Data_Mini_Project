import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class DataAnalysis2:
    def __init__(self, df):
        self.df = df
        self.column_types = self.list_column_types()

    def list_column_types(self):
        column_types = {}
        for column in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                if len(self.df[column].unique()) <= 10:
                    column_types[column] = 'numeric ordinal'
                else:
                    column_types[column] = 'interval'
            else:
                if len(self.df[column].unique()) <= 10:
                    column_types[column] = 'nominal'
                else:
                    column_types[column] = 'non-numeric ordinal'
        print("Column Classifications:")
        for col, dtype in column_types.items():
            print(f"{col}: {dtype}")
        return column_types

    def select_variable(self, data_type, max_categories=None, allow_skip=False):
        available_columns = [col for col, dtype in self.column_types.items() if dtype == data_type]
        if max_categories is not None:
            available_columns = [col for col in available_columns if self.df[col].nunique() <= max_categories]

        if not available_columns:
            if allow_skip:
                return None
            else:
                raise ValueError("No columns available with the specified data type and max categories.")

        print(f"Available {data_type} columns: {available_columns}")
        selected_column = input("Please select a column: ")
        while selected_column not in available_columns:
            selected_column = input("Invalid selection. Please select a column: ")

        return selected_column

    def check_normality(self, data, size_limit=2000):
        data = data.dropna()
        if len(data) > size_limit:
            stat, p_value = stats.anderson(data, dist='norm')[:2]
            test_name = "Anderson-Darling Test"
        else:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk Test"
        print(f"{test_name}: Statistic={stat}, p-value={p_value}")
        return stat, p_value

    def perform_regression(self, x_var, y_var):
        X = self.df[x_var].dropna()
        Y = self.df[y_var].dropna()

        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]

        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_value ** 2:.4f}")
        print(f"P-value: {p_value:.15f}")
        print(f"Standard error: {std_err:.4f}")

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        groups = [group[continuous_var].dropna() for name, group in self.df.groupby(categorical_var)]
        normality_test = self.check_normality(self.df[continuous_var])

        if normality_test[1] > 0.05:
            stat, p_value = stats.ttest_ind(*groups)
            test_name = "t-test"
        else:
            stat, p_value = stats.mannwhitneyu(*groups)
            test_name = "Mann-Whitney U Test"

        print(f"{test_name}: Statistic = {stat:.4f}, p-value = {p_value:.15f}")

    def chi_square_test(self, categorical_var_1, categorical_var_2):
        contingency_table = pd.crosstab(self.df[categorical_var_1], self.df[categorical_var_2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        print(f"Chi-square Test: chi2 = {chi2:.4f}, p-value = {p:.15f}")
