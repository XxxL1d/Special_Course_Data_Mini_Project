import pandas as pd
import matplotlib.pyplot as plt

class DataInspection:
    def __init__(self):
        self.df = None

    def load_csv(self, file_path):
        self.df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    def plot_histogram(self, col):
        plt.figure()
        self.df[col].hist(bins=10)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self, x_col, y_col):
        plt.figure()
        self.df.boxplot(column=y_col, by=x_col)
        plt.title(f'Box Plot of {y_col} by {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.suptitle('')
        plt.show()

    def plot_bar_chart(self, col):
        plt.figure()
        self.df[col].value_counts().plot(kind='bar')
        plt.title(f'Bar Chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    def plot_scatter(self, x_col, y_col):
        plt.figure()
        plt.scatter(self.df[x_col], self.df[y_col])
        plt.title(f'Scatter Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def handle_missing_values(self, col):
        total_missing = self.df[col].isnull().sum()
        total = len(self.df[col])
        missing_percentage = (total_missing / total) * 100

        if missing_percentage > 50:
            self.df.drop(columns=[col], inplace=True)
            print(f"Column '{col}' dropped due to more than 50% missing values.")
            return False
        else:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in numeric column '{col}' with median value {median_value}.")
            else:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in non-numeric column '{col}' with mode value '{mode_value}'.")
            return True

    def check_data_types(self, col):
        if self.df[col].dtype == object:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
                print(f"Converted column '{col}' to numeric data type.")
            except ValueError:
                print(f"Column '{col}' remains as object type.")

    def classify_and_calculate(self, col):
        if not self.handle_missing_values(col):
            return None

        self.check_data_types(col)
        unique_values = self.df[col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(self.df[col])

        central_tendency = None

        if is_numeric:
            if unique_values > 10:
                central_tendency = self.df[col].mean()
                print(f"Mean of '{col}': {central_tendency}")
                self.plot_histogram(col)
            else:
                central_tendency = self.df[col].median()
                print(f"Median of ordinal numeric column '{col}': {central_tendency}")
                self.plot_boxplot(col, col)
        else:
            central_tendency = self.df[col].mode()[0]
            print(f"Mode of nominal column '{col}': {central_tendency}")
            self.plot_bar_chart(col)

        return central_tendency

    def classify_columns(self):
        for col in self.df.columns:
            print(f"\nProcessing column: {col}")
            self.classify_and_calculate(col)

    def numeric_columns(self):
        return [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]

    def ordinal_columns(self):
        return [col for col in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[col])]
