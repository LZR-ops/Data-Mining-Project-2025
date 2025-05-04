import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"Error downloading VADER lexicon: {e}")
    raise Exception("Please install vader_lexicon manually using: nltk.download('vader_lexicon')")

# Initialize VADER
sid = SentimentIntensityAnalyzer()

# Load labeled tweets
df = pd.read_csv('labeled_tweets.csv')
print(f"Original labeled tweet count: {len(df)}")
print("Original sentiment distribution:")
print(df['sentiment'].value_counts())

# Compute VADER scores
df['vader_score'] = df['cleaned_tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Fix specific misclassifications
df.loc[df['cleaned_tweet'] == "She won't if she found Tesla spacex PayPal boring", 'sentiment'] = 'negative'
df.loc[df['cleaned_tweet'] == "How do you reconcile having many successful businesses but seemingly poor customer service?", 'sentiment'] = 'negative'

# Relabel neutral tweets with strong VADER scores
neutral_tweets = df[df['sentiment'] == 'neutral']
print("\nNeutral tweets with VADER > 0.5 (potential positive):")
print(neutral_tweets[neutral_tweets['vader_score'] > 0.5][['cleaned_tweet', 'vader_score']])
print("\nNeutral tweets with VADER < -0.5 (potential negative):")
print(neutral_tweets[neutral_tweets['vader_score'] < -0.5][['cleaned_tweet', 'vader_score']])

# Automatically relabel based on VADER thresholds
df.loc[(df['sentiment'] == 'neutral') & (df['vader_score'] > 0.5), 'sentiment'] = 'positive'
df.loc[(df['sentiment'] == 'neutral') & (df['vader_score'] < -0.5), 'sentiment'] = 'negative'

# Validate negative tweets (ensure VADER < 0)
negative_tweets = df[df['sentiment'] == 'negative']
print("\nNegative tweets with VADER > 0.2 (potential neutral/positive):")
print(negative_tweets[negative_tweets['vader_score'] > 0.2][['cleaned_tweet', 'vader_score']])

# Manually fix known negative tweet
df.loc[df['cleaned_tweet'] == "Looks like Tesla investors are losing faith. Can you comment?", 'sentiment'] = 'negative'

# Save updated labels
df.to_csv('labeled_tweets.csv', index=False)
print("\nUpdated sentiment distribution:")
print(df['sentiment'].value_counts())
print("Updated labeled tweets saved to labeled_tweets.csv")

# Print neutral VADER score statistics
neutral_vader = df[df['sentiment'] == 'neutral']['vader_score']
print("\nUpdated neutral VADER score statistics:")
print(neutral_vader.describe())