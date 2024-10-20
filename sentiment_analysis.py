import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class SentimentAnalysis:
    def __init__(self, df):
        self.df = df

    def get_text_columns(self):
        text_columns = []
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                text_columns.append(column)

        result_data = {
            'Column Name': [],
            'Average Entry Length': [],
            'Unique Entries': []
        }

        for col in text_columns:
            avg_length = self.df[col].dropna().apply(len).mean()
            unique_entries = self.df[col].nunique()
            result_data['Column Name'].append(col)
            result_data['Average Entry Length'].append(avg_length)
            result_data['Unique Entries'].append(unique_entries)

        return pd.DataFrame(result_data)

    def vader_sentiment_analysis(self, data):
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []

        for entry in data:
            score = analyzer.polarity_scores(entry)['compound']
            scores.append(score)
            if score >= 0.05:
                sentiments.append('positive')
            elif score <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')

        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        scores = []
        sentiments = []
        subjectivity = []

        for entry in data:
            blob = TextBlob(entry)
            score = blob.sentiment.polarity
            scores.append(score)
            subjectivity.append(blob.sentiment.subjectivity)
            if score > 0:
                sentiments.append('positive')
            elif score == 0:
                sentiments.append('neutral')
            else:
                sentiments.append('negative')

        return scores, sentiments, subjectivity

    def distilbert_sentiment_analysis(self, data):
        if pipeline is None:
            raise ImportError("Transformers library is not installed.")

        sentiment_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        scores = []
        sentiments = []

        for entry in data:
            result = sentiment_pipeline(entry)[0]
            label = result['label']
            score = result['score']
            scores.append(score)
            if label in ['4 stars', '5 stars']:
                sentiments.append('positive')
            elif label == '3 stars':
                sentiments.append('neutral')
            else:
                sentiments.append('negative')

        return scores, sentiments
