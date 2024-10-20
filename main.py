import sys
import pandas as pd
from data_inspection import DataInspection
from data_analysis1 import DataAnalysis1
from data_analysis2 import DataAnalysis2
from sentiment_analysis import SentimentAnalysis

def main():
    file_path = input("Please provide the file path to the CSV dataset: ")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    while True:
        print("\nMain Menu:")
        print("1. Data Inspection")
        print("2. Statistical Analysis 1")
        print("3. Statistical Analysis 2")
        print("4. Sentiment Analysis")
        print("5. Exit")
        choice = input("Please select an option (1-5): ")

        if choice == '1':
            data_inspection_menu(df)
        elif choice == '2':
            data_analysis1_menu(df)
        elif choice == '3':
            data_analysis2_menu(df)
        elif choice == '4':
            sentiment_analysis_menu(df)
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

def data_inspection_menu(df):
    di = DataInspection()
    di.df = df.copy()
    di.classify_columns()

    while True:
        print("\nData Inspection Menu:")
        print("1. Scatter Plot")
        print("2. Box Plot")
        print("3. Correlation")
        print("4. Standard Deviation")
        print("5. Kurtosis")
        print("6. Skewness")
        print("7. Back to Main Menu")
        choice = input("Please select an option (1-7): ")

        if choice == '1':
            numeric_cols = di.numeric_columns()
            print(f"Numeric Columns: {numeric_cols}")
            x_col = input("Enter the name of the X-axis column: ")
            y_col = input("Enter the name of the Y-axis column: ")
            if x_col in numeric_cols and y_col in numeric_cols:
                di.plot_scatter(x_col, y_col)
            else:
                print("Invalid columns selected.")
        elif choice == '2':
            numeric_cols = di.numeric_columns()
            ordinal_cols = di.ordinal_columns()
            print(f"Numeric Columns: {numeric_cols}")
            print(f"Ordinal Columns: {ordinal_cols}")
            num_col = input("Enter the name of the numeric column: ")
            ord_col = input("Enter the name of the ordinal column: ")
            if num_col in numeric_cols and ord_col in ordinal_cols:
                di.plot_boxplot(ord_col, num_col)
            else:
                print("Invalid columns selected.")
        elif choice == '3':
            numeric_cols = di.numeric_columns()
            print(f"Numeric Columns: {numeric_cols}")
            col1 = input("Enter the first column: ")
            col2 = input("Enter the second column: ")
            if col1 in numeric_cols and col2 in numeric_cols:
                corr_value = di.df[col1].corr(di.df[col2])
                print(f"Correlation between '{col1}' and '{col2}': {corr_value}")
            else:
                print("Invalid columns selected.")
        elif choice == '4':
            numeric_cols = di.numeric_columns()
            print(f"Numeric Columns: {numeric_cols}")
            col = input("Enter the column name: ")
            if col in numeric_cols:
                std_value = di.df[col].std()
                print(f"Standard Deviation of '{col}': {std_value}")
            else:
                print("Invalid column selected.")
        elif choice == '5':
            numeric_cols = di.numeric_columns()
            print(f"Numeric Columns: {numeric_cols}")
            col = input("Enter the column name: ")
            if col in numeric_cols:
                kurtosis_value = di.df[col].kurt()
                print(f"Kurtosis of '{col}': {kurtosis_value}")
            else:
                print("Invalid column selected.")
        elif choice == '6':
            numeric_cols = di.numeric_columns()
            print(f"Numeric Columns: {numeric_cols}")
            col = input("Enter the column name: ")
            if col in numeric_cols:
                skewness_value = di.df[col].skew()
                print(f"Skewness of '{col}': {skewness_value}")
            else:
                print("Invalid column selected.")
        elif choice == '7':
            break
        else:
            print("Invalid choice. Please try again.")

def data_analysis1_menu(df):
    da1 = DataAnalysis1(df)

    while True:
        print("\nStatistical Analysis 1 Menu:")
        print("1. Normality Test")
        print("2. Hypothesis Test")
        print("3. Back to Main Menu")
        choice = input("Please select an option (1-3): ")

        if choice == '1':
            continuous_var = da1.select_variable('interval')
            data = da1.df[continuous_var]
            da1.plot_qq_histogram(data, continuous_var)
            da1.check_normality(data)
        elif choice == '2':
            continuous_var = da1.select_variable('interval')
            categorical_var = da1.select_variable('nominal', allow_skip=True)
            skewed = da1.check_skewness(da1.df[continuous_var])
            null_hyp = input("Please enter the null hypothesis: ")
            if categorical_var:
                da1.hypothesis_test(continuous_var, categorical_var, skewed, null_hyp)
            else:
                print("No categorical variable selected.")
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

def data_analysis2_menu(df):
    da2 = DataAnalysis2(df)

    while True:
        print("\nStatistical Analysis 2 Menu:")
        print("1. Linear Regression")
        print("2. t-test or Mann-Whitney U Test")
        print("3. Chi-square Test")
        print("4. Back to Main Menu")
        choice = input("Please select an option (1-4): ")

        if choice == '1':
            x_var = da2.select_variable('interval')
            y_var = da2.select_variable('interval')
            da2.perform_regression(x_var, y_var)
        elif choice == '2':
            continuous_var = da2.select_variable('interval')
            categorical_var = da2.select_variable('nominal', max_categories=2)
            da2.t_test_or_mannwhitney(continuous_var, categorical_var)
        elif choice == '3':
            categorical_var_1 = da2.select_variable('nominal')
            categorical_var_2 = da2.select_variable('nominal')
            da2.chi_square_test(categorical_var_1, categorical_var_2)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

def sentiment_analysis_menu(df):
    sa = SentimentAnalysis(df)
    text_columns_df = sa.get_text_columns()
    print("\nText Columns in the Dataset:")
    print(text_columns_df)

    column_name = input("Please enter the column name to analyze: ")
    if column_name not in df.columns:
        print("Invalid column name.")
        return
    if df[column_name].dtype != 'object':
        print("Selected column is not a text column.")
        return

    print("\nChoose the type of sentiment analysis:")
    print("1. VADER")
    print("2. TextBlob")
    print("3. DistilBERT")
    analysis_type = input("Enter your choice (1-3): ")

    if analysis_type == '1':
        scores, sentiments = sa.vader_sentiment_analysis(df[column_name].dropna())
        result_df = pd.DataFrame({
            'Text': df[column_name].dropna(),
            'Score': scores,
            'Sentiment': sentiments
        })
        print(result_df)
    elif analysis_type == '2':
        scores, sentiments, subjectivity = sa.textblob_sentiment_analysis(df[column_name].dropna())
        result_df = pd.DataFrame({
            'Text': df[column_name].dropna(),
            'Polarity Score': scores,
            'Sentiment': sentiments,
            'Subjectivity': subjectivity
        })
        print(result_df)
    elif analysis_type == '3':
        try:
            scores, sentiments = sa.distilbert_sentiment_analysis(df[column_name].dropna())
            result_df = pd.DataFrame({
                'Text': df[column_name].dropna(),
                'Score': scores,
                'Sentiment': sentiments
            })
            print(result_df)
        except ImportError as e:
            print("Transformers library is not installed. Please install it to use this feature.")
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()
