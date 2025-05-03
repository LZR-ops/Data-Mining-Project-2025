import pandas as pd
import re

# Load the dataset
df = pd.read_csv("Tesla.csv")
print("Original tweet count:", len(df))

# Keep only English tweets
df = df[df['language'] == 'en'].copy()
print("English tweet count:", len(df))

# Function to clean tweets
def clean_tweet(text):
    text = re.sub(r'@[\w]+', '', text)  # Remove @mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    text = re.sub(r'http\S+', '', text) # Remove URLs
    return text.strip()

# Apply cleaning
df['cleaned_tweet'] = df['tweet'].astype(str).apply(clean_tweet)

# Save cleaned tweets to a CSV
df[['cleaned_tweet']].to_csv("cleaned_tweets.csv", index=False)
print("Cleaned tweets saved to cleaned_tweets.csv")



import pandas as pd

# Load the cleaned tweets
df = pd.read_csv("cleaned_tweets.csv")

# Drop NaN and sample 200 tweets randomly
sample_df = df[['cleaned_tweet']].dropna().sample(n=200, random_state=42).copy()

# Add an empty 'sentiment' column
sample_df['sentiment'] = ''

# Save as a new CSV
sample_df.to_csv("labeled_tweets_template.csv", index=False)

print("Template saved to labeled_tweets_template.csv — ready for manual labeling!")



# STEP 6: Train a Naive Bayes Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load your manually labeled tweets (make sure you've filled in sentiments)
df = pd.read_csv("labeled_tweets.csv")

# Extract features and labels
X_text = df['cleaned_tweet']
y = df['sentiment']

# Convert text data into bag-of-words vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy on test set: {accuracy:.2%}")



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the labeled dataset from the CSV file
labeled_df = pd.read_csv('labeled_tweets.csv')  # Make sure the path is correct

# Example of your main dataframe containing tweets (replace with your actual data)
df = pd.DataFrame({
    'cleaned_tweet': ['Tesla is great', 'I don’t like Tesla', 'Tesla is the future']
})

# Initialize the vectorizer and model (assuming you have trained them before)
vectorizer = TfidfVectorizer()
X_labeled = vectorizer.fit_transform(labeled_df['cleaned_tweet'])

# Train a model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_labeled, labeled_df['sentiment'])

# Get unlabeled tweets (those not in labeled_df)
unlabeled_df = df[~df['cleaned_tweet'].isin(labeled_df['cleaned_tweet'])].copy()

# Vectorize the unlabeled tweets
X_unlabeled = vectorizer.transform(unlabeled_df['cleaned_tweet'])

# Predict sentiment on the unlabeled tweets
unlabeled_df['predicted_sentiment'] = model.predict(X_unlabeled)

# Show a summary of sentiment distribution in percentage
summary = unlabeled_df['predicted_sentiment'].value_counts(normalize=True) * 100
print("Sentiment distribution (%):")
print(summary)

# Check the distribution of sentiments in your labeled data
print(labeled_df['sentiment'].value_counts())


# === Step 7: Predict Sentiment on Unlabeled Tweets and Visualize ===

import matplotlib.pyplot as plt

# Load labeled data with only the required columns
df = pd.read_csv("labeled_tweets.csv", usecols=['cleaned_tweet', 'sentiment'])
df['predicted_sentiment'] = df['sentiment']

# Load full cleaned data
df_full = pd.read_csv("cleaned_tweets.csv")

# Remove already labeled tweets
unlabeled_df = df_full[~df_full['cleaned_tweet'].isin(df['cleaned_tweet'])].copy()

# Transform using the same vectorizer
X_unlabeled = vectorizer.transform(unlabeled_df['cleaned_tweet'])

# Predict sentiment for the rest of the tweets
unlabeled_df['predicted_sentiment'] = model.predict(X_unlabeled)

# Combine both labeled and predicted
all_df = pd.concat([
    df[['cleaned_tweet', 'predicted_sentiment']],
    unlabeled_df[['cleaned_tweet', 'predicted_sentiment']]
])

import matplotlib.pyplot as plt

# Count sentiment distribution
sentiment_counts = all_df['predicted_sentiment'].value_counts()

# Plot and save
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution of Tesla Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)
plt.tight_layout()
plt.tight_layout()
plt.savefig("sentiment_distribution.png") 


