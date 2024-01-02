import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

class SentimentAnalyzer:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def fetch_tweets(self, query, count=100):
        tweets = self.api.search(q=query, count=count)
        return tweets

    def categorize_sentiment(self, polarity):
        if polarity > 0.5:
            return 'Positive'
        elif polarity < -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    def analyze_and_visualize(self, query, count=100):
        tweets = self.fetch_tweets(query, count)
        sentiments = [self.analyze_sentiment(tweet.text) for tweet in tweets]

        categorized_sentiments = [self.categorize_sentiment(polarity) for polarity in sentiments]

        self.visualize_sentiment(sentiments, categorized_sentiments)

        # Store sentiment data in a CSV file
        self.store_sentiment_data(query, tweets, sentiments, categorized_sentiments)

    def visualize_sentiment(self, sentiments, categories):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(sentiments, marker='o', linestyle='', markersize=5, label='Sentiment Polarity')
        plt.axhline(y=0, color='gray', linestyle='--', label='Neutral Line')
        plt.xlabel('Tweet Index')
        plt.ylabel('Sentiment Polarity')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.hist(categories, bins=['Negative', 'Neutral', 'Positive'], color=['red', 'gray', 'green'], edgecolor='black')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Number of Tweets')

        plt.suptitle('Sentiment Analysis of Tweets')
        plt.show()

    def store_sentiment_data(self, query, tweets, sentiments, categories):
        # Create a DataFrame to store sentiment data
        data = {
            'Tweet Index': range(1, len(tweets) + 1),
            'Tweet ID': [tweet.id_str for tweet in tweets],
            'Tweet Text': [tweet.text for tweet in tweets],
            'Sentiment Polarity': sentiments,
            'Sentiment Category': categories,
            'Created At': [tweet.created_at for tweet in tweets]
        }

        df = pd.DataFrame(data)

        # Save sentiment data to a CSV file
        filename = f'{query}_sentiment_data_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        print(f'Sentiment data saved to {filename}')

def main():
    consumer_key = 'your_consumer_key'
    consumer_secret = 'your_consumer_secret'
    access_token = 'your_access_token'
    access_token_secret = 'your_access_token_secret'

    analyzer = SentimentAnalyzer(consumer_key, consumer_secret, access_token, access_token_secret)

    query = input("Enter the topic you want to analyze: ")
    count = int(input("Enter the number of tweets to analyze: "))

    analyzer.analyze_and_visualize(query, count)

if __name__ == "__main__":
    main()
