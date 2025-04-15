import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Get absolute path to the dataset
dataset_path = os.path.abspath("Dataset/spam.csv")

# Load dataset
df = pd.read_csv(dataset_path, encoding="latin-1")
df = df.rename(columns={'v1': 'class', 'v2': 'message'})
df = df[['class', 'message']]  # Keep only relevant columns

# Encode labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

# Features and labels
X = df['message']
y = df['label']

# Vectorize text
cv = CountVectorizer()
X_vectorized = cv.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectorized, y)

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
os.makedirs(model_dir, exist_ok=True)

# Saving the trained model using pickle
filename = os.path.join(model_dir, 'spam_classifier_model.pkl')
pickle.dump(model, open(filename, 'wb'))

print("SUCCESS")