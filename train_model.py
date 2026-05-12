import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("Loading dataset...")

# Fake.csv aur True.csv dono alag load karo
fake_df = pd.read_csv(os.path.join("dataset", "Fake.csv"))
true_df = pd.read_csv(os.path.join("dataset", "True.csv"))

# Label add karo manually
# 1 = Fake, 0 = Real
fake_df["label"] = 1
true_df["label"] = 0

# Dono ko ek mein combine karo
df = pd.concat([fake_df, true_df], ignore_index=True)
print(f"Total rows: {len(df)}")
print(f"Fake: {len(fake_df)} | Real: {len(true_df)}")

# Missing values fill karo
df['title'] = df['title'].fillna('')
df['text']  = df['text'].fillna('')

# Title + text combine karo
df['content'] = df['title'] + ' ' + df['text']

X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining: {len(X_train)} | Testing: {len(X_test)}")

print("\nApplying TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred   = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

os.makedirs("model", exist_ok=True)
with open(os.path.join("model", "model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join("model", "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved! Ab python app.py chalaao.")