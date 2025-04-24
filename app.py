from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.abspath("Model/spam_classifier_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load dataset and fit CountVectorizer little changes no changes reuquired hello sir
dataset_path = os.path.abspath("Dataset/spam.csv")
df = pd.read_csv(dataset_path, encoding="latin-1")
df = df.rename(columns={'v1': 'class', 'v2': 'message'})  # if needed
df = df[['class', 'message']]  # drop unnecessary columns
df['label'] = df['class'].map({'ham': 0, 'spam': 1})

X = df['message']
vectorizer = CountVectorizer()
vectorizer.fit(X)  # refit the vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)

        if prediction[0] == 1:
            return render_template('index.html', prediction="Spam Detected - Be Careful!")
        else:
            return render_template('index.html', prediction="Not Spam")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

