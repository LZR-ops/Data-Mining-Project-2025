import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import preprocessor as p
from imblearn.over_sampling import SMOTE
import os
import emoji
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK resources with fallback."""
    resources = ['punkt', 'punkt_tab', 'wordnet', 'stopwords', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK resource {resource}: {e}")
            if resource == 'punkt_tab':
                print("Falling back to 'punkt' tokenizer.")
            else:
                raise Exception(f"Please install {resource} manually using: nltk.download('{resource}')")

# Initialize NLTK resources
download_nltk_data()

# Initialize lemmatizer, VADER, and custom stopwords
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
custom_stopwords = [word for word in stop_words if word not in ['great', 'bad', 'good', 'not', 'no', 'never', 'awesome', 'terrible', 'love', 'hate']]

# Expanded sentiment-rich emoji dictionary
emoji_sentiment = {
    ':smile:': 'positive', ':happy:': 'positive', ':thumbs_up:': 'positive', ':grinning_face:': 'positive',
    ':heart:': 'positive', ':clap:': 'positive', ':smiling_face_with_heart_eyes:': 'positive',
    ':sad:': 'negative', ':angry:': 'negative', ':thumbs_down:': 'negative', ':disappointed_face:': 'negative',
    ':warning:': 'negative', ':broken_heart:': 'negative', ':crying_face:': 'negative'
}

def clean_tweet(text):
    """Clean tweet text, preserving sentiment-rich tokens and normalizing emojis."""
    if not isinstance(text, str) or not text.strip():
        return "empty"
    # Normalize sentiment-rich emojis to text and repeat thrice
    text = emoji.demojize(text, delimiters=(":", ":"))
    for em, sent in emoji_sentiment.items():
        text = re.sub(em, f"{sent} {sent} {sent}", text)
    # Use tweet-preprocessor to handle URLs, mentions
    p.set_options(p.OPT.URL, p.OPT.MENTION)
    text = p.clean(text)
    # Convert to lowercase
    text = text.lower()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and lemmatize
    try:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in custom_stopwords]
        cleaned_text = ' '.join(tokens)
        return cleaned_text if cleaned_text.strip() else "empty"
    except LookupError:
        raise Exception("Tokenizer not found. Ensure 'punkt' or 'punkt_tab' is downloaded.")

def load_and_preprocess_data(tesla_file='Tesla.csv', cleaned_file='cleaned_tweets.csv'):
    """Load and preprocess Tesla tweets, saving cleaned data."""
    if not os.path.exists(tesla_file):
        raise FileNotFoundError(f"{tesla_file} not found")
    
    # Load dataset
    df = pd.read_csv(tesla_file)
    print(f"Original tweet count: {len(df)}")
    
    # Keep only English tweets
    df = df[df['language'] == 'en'].copy()
    print(f"English tweet count: {len(df)}")
    
    # Apply cleaning
    df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)
    
    # Drop NaN or empty tweets
    df = df.dropna(subset=['cleaned_tweet'])
    df = df[df['cleaned_tweet'] != "empty"]
    print(f"Non-empty cleaned tweet count: {len(df)}")
    
    # Save cleaned tweets
    df[['cleaned_tweet']].to_csv(cleaned_file, index=False)
    print(f"Cleaned tweets saved to {cleaned_file}")
    return df

def train_and_evaluate_model(labeled_file='labeled_tweets.csv'):
    """Train and evaluate a sentiment classifier."""
    if not os.path.exists(labeled_file):
        raise FileNotFoundError(f"{labeled_file} not found")
    
    # Load labeled data
    df = pd.read_csv(labeled_file)
    print(f"Raw labeled tweet count: {len(df)}")
    
    # Handle missing values and invalid labels
    df = df.dropna(subset=['cleaned_tweet', 'sentiment'])
    df = df[df['cleaned_tweet'] != "empty"]
    df = df[df['cleaned_tweet'].apply(lambda x: isinstance(x, str))]
    df = df[df['sentiment'].isin(['positive', 'negative', 'neutral'])]
    print(f"Filtered labeled tweet count: {len(df)}")
    print("Labeled data sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    if len(df) < 300:
        raise ValueError(f"Expected ~300 labeled tweets, found {len(df)} after filtering")
    
    # Add weighted VADER sentiment scores
    df['vader_score'] = df['cleaned_tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_score'] = df.apply(lambda row: row['vader_score'] * 2.0 if any(word in row['cleaned_tweet'] for word in ['bad', 'terrible', 'awful']) else row['vader_score'], axis=1)
    
    # Validate neutral VADER scores
    neutral_vader = df[df['sentiment'] == 'neutral']['vader_score']
    print("\nNeutral VADER score statistics:")
    print(neutral_vader.describe())
    
    # Extract features and labels
    X_text = df['cleaned_tweet']
    y = df['sentiment']
    
    # Vectorize text with n-grams
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2)
    X_text_features = vectorizer.fit_transform(X_text)
    
    # Combine TF-IDF with VADER scores
    X = np.hstack((X_text_features.toarray(), df[['vader_score']].values))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("SMOTE applied. New training data distribution:")
        print(pd.Series(y_train).value_counts())
    except ValueError as e:
        print(f"SMOTE failed: {e}. Proceeding without oversampling.")
    
    # Encode labels for XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    print("Label encoder classes:", list(label_encoder.classes_))
    
    # Compute sample weights
    sample_weights = compute_sample_weight(class_weight={'positive': 4, 'negative': 2, 'neutral': 1}, y=y_train)
    
    # Train XGBoost with GridSearchCV
    model = XGBClassifier(random_state=42, objective='multi:softprob', device='cpu')
    param_grid = {
        'max_depth': [3],
        'n_estimators': [100, 200],
        'learning_rate': [0.05],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0.5],
        'reg_lambda': [2.0]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', n_jobs=1, error_score='raise')
    grid_search.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    
    # Best model
    model = grid_search.best_estimator_
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    
    # Evaluate model
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'], zero_division=0))
    
    # Cross-validation with F1-score
    cv_scores = cross_val_score(model, X, label_encoder.transform(y), cv=5, scoring='f1_macro')
    print(f"5-fold cross-validation macro F1-score: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['negative', 'neutral', 'positive'])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['negative', 'neutral', 'positive'])
    plt.yticks(tick_marks, ['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance (TF-IDF features only)
    feature_names = vectorizer.get_feature_names_out()
    importance = model.feature_importances_[:-1]  # Exclude VADER score
    top_indices = np.argsort(importance)[-10:]
    print("\nTop 10 TF-IDF features by importance:")
    for idx in top_indices:
        print(f"{feature_names[idx]}: {importance[idx]:.4f}")
    
    return model, vectorizer, label_encoder

def predict_unlabeled_tweets(model, vectorizer, label_encoder, cleaned_file='cleaned_tweets.csv', labeled_file='labeled_tweets.csv'):
    """Predict sentiment for unlabeled tweets and visualize distribution."""
    if not os.path.exists(cleaned_file) or not os.path.exists(labeled_file):
        raise FileNotFoundError("Required files not found")
    
    # Load data
    df_full = pd.read_csv(cleaned_file)
    df_labeled = pd.read_csv(labeled_file)
    
    # Handle NaN and empty tweets
    df_full = df_full.dropna(subset=['cleaned_tweet'])
    df_full = df_full[df_full['cleaned_tweet'] != "empty"]
    print(f"Valid unlabeled tweet count: {len(df_full)}")
    
    # Remove labeled tweets
    unlabeled_df = df_full[~df_full['cleaned_tweet'].isin(df_labeled['cleaned_tweet'])].copy()
    print(f"Unlabeled tweet count after filtering: {len(unlabeled_df)}")
    
    # Add weighted VADER scores
    unlabeled_df['vader_score'] = unlabeled_df['cleaned_tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
    unlabeled_df['vader_score'] = unlabeled_df.apply(lambda row: row['vader_score'] * 2.0 if any(word in row['cleaned_tweet'] for word in ['bad', 'terrible', 'awful']) else row['vader_score'], axis=1)
    
    # Vectorize and predict
    X_text_unlabeled = vectorizer.transform(unlabeled_df['cleaned_tweet'])
    X_unlabeled = np.hstack((X_text_unlabeled.toarray(), unlabeled_df[['vader_score']].values))
    unlabeled_pred_encoded = model.predict(X_unlabeled)
    unlabeled_df['predicted_sentiment'] = label_encoder.inverse_transform(unlabeled_pred_encoded)
    print("Sample unlabeled predictions (first 5):")
    print(unlabeled_df[['cleaned_tweet', 'predicted_sentiment']].head())
    
    # Validate predictions are categorical
    unique_predictions = unlabeled_df['predicted_sentiment'].unique()
    if not all(pred in ['positive', 'negative', 'neutral'] for pred in unique_predictions):
        raise ValueError(f"Non-categorical predictions detected: {unique_predictions}")
    
    # Combine labeled and predicted data
    all_df = pd.concat([
        df_labeled[['cleaned_tweet', 'sentiment']].rename(columns={'sentiment': 'predicted_sentiment'}),
        unlabeled_df[['cleaned_tweet', 'predicted_sentiment']]
    ])
    print("Combined sentiment distribution (before plotting):")
    print(all_df['predicted_sentiment'].value_counts())
    
    # Plot sentiment distribution
    sentiment_counts = all_df['predicted_sentiment'].value_counts()
    percentages = 100 * sentiment_counts / len(all_df)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(sentiment_counts)), sentiment_counts, color=['red', 'gray', 'green'])
    plt.title('Sentiment Distribution of Tesla Tweets')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.xticks(range(len(sentiment_counts)), sentiment_counts.index, rotation=0)
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{percentage:.1f}%', 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()
    
    print("\nSentiment distribution (%):")
    print(percentages)
    
    # Save predictions
    all_df.to_csv('all_predictions.csv', index=False)
    print("Predictions saved to all_predictions.csv")
    
    return all_df

def main():
    """Main function to run the sentiment analysis pipeline."""
    try:
        # Check for dependencies
        try:
            import preprocessor
            import emoji
            import xgboost
        except ImportError as e:
            raise ImportError(f"Missing dependency {e.name}. Install it with 'pip install {e.name}'")
        
        # Step 1: Preprocess data
        df = load_and_preprocess_data()
        
        # Step 2: Train and evaluate model
        model, vectorizer, label_encoder = train_and_evaluate_model()
        
        # Step 3: Predict and visualize
        all_df = predict_unlabeled_tweets(model, vectorizer, label_encoder)
        
        # Step 4: Error analysis
        labeled_df = pd.read_csv('labeled_tweets.csv')
        print(f"Raw error analysis tweet count: {len(labeled_df)}")
        labeled_df = labeled_df.dropna(subset=['cleaned_tweet'])
        labeled_df = labeled_df[labeled_df['cleaned_tweet'] != "empty"]
        labeled_df = labeled_df[labeled_df['cleaned_tweet'].apply(lambda x: isinstance(x, str))]
        print(f"Filtered error analysis tweet count: {len(labeled_df)}")
        labeled_df['vader_score'] = labeled_df['cleaned_tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
        labeled_df['vader_score'] = labeled_df.apply(lambda row: row['vader_score'] * 2.0 if any(word in row['cleaned_tweet'] for word in ['bad', 'terrible', 'awful']) else row['vader_score'], axis=1)
        X_text_labeled = vectorizer.transform(labeled_df['cleaned_tweet'])
        X_labeled = np.hstack((X_text_labeled.toarray(), labeled_df[['vader_score']].values))
        labeled_pred_encoded = model.predict(X_labeled)
        labeled_df['predicted_sentiment'] = label_encoder.inverse_transform(labeled_pred_encoded)
        errors = labeled_df[labeled_df['sentiment'] != labeled_df['predicted_sentiment']][['cleaned_tweet', 'sentiment', 'predicted_sentiment']]
        errors['vader_score'] = errors['cleaned_tweet'].apply(lambda x: sid.polarity_scores(x)['compound'])
        errors.to_csv('misclassifications.csv', index=False)
        print("\nSample misclassifications (first 5):")
        print(errors[['cleaned_tweet', 'sentiment', 'predicted_sentiment', 'vader_score']].head())
        print("All misclassifications saved to misclassifications.csv")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main()