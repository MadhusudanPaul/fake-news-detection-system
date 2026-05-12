# ============================================================
# app.py
# ============================================================
# This is the Flask web server that connects the trained ML
# model to the website.
#
# What it does:
#   1. Loads the saved model and vectorizer from /model/
#   2. Serves the HTML page at http://localhost:5000
#   3. Accepts a POST request with article text
#   4. Runs the text through the model
#   5. Returns a JSON response with prediction + analysis
# ============================================================

import os
import re
import pickle
import time
from flask import Flask, request, jsonify, render_template

# ── Create the Flask app ─────────────────────────────────────
# Flask needs to know where to find the HTML templates.
# By default it looks inside a folder called "templates/".
app = Flask(__name__)

# ── Load the saved model and vectorizer ──────────────────────
# We load these once when the server starts, not on every request.
# This makes the app faster.

model_path      = os.path.join("model", "model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print("Model and vectorizer loaded successfully.")

except FileNotFoundError:
    print("\n ERROR: Model files not found!")
    print(" Please run train_model.py first:")
    print("   python train_model.py\n")
    model = None
    vectorizer = None


# ── Helper: Analyze text for extra signals ───────────────────
# These are simple rule-based checks that add context
# to the result beyond just the ML model's prediction.

def analyze_text(text):
    """
    Runs basic checks on the article text and returns
    a dictionary with verified facts and reliability scores.
    """

    text_lower = text.lower()
    word_count = len(text.split())

    # --- Verified Facts ---
    # Check for presence of dates (e.g. "January 2024", "12/03/2024")
    has_date = bool(re.search(
        r'\b(january|february|march|april|may|june|july|august|september'
        r'|october|november|december|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
        text_lower
    ))

    # Check if there is a byline / author name (e.g. "By John Smith")
    has_author = bool(re.search(r'\bby\s+[A-Z][a-z]+\s+[A-Z][a-z]+', text))

    # Check if any sources are cited (e.g. "according to", "said", "reported")
    has_sources = bool(re.search(
        r'\b(according to|reported by|said|stated|confirmed|sources say)\b',
        text_lower
    ))

    # Detect emotional / sensational language that is common in fake news
    emotional_words = [
        'shocking', 'unbelievable', 'secret', 'exposed', 'breaking',
        'urgent', 'you won\'t believe', 'mainstream media', 'they don\'t want',
        'wake up', 'hoax', 'crisis actor', 'coverup', 'bombshell'
    ]
    emotional_count = sum(1 for w in emotional_words if w in text_lower)
    has_emotional_language = emotional_count >= 2

    # Check if headline (first sentence) is consistent with body
    # Simple proxy: does the text have more than 3 sentences?
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    headline_matches_body = len(sentences) >= 3

    # --- Source Reliability Scores (0–100) ---
    # These are heuristic scores based on text features.

    # Factual language: presence of numbers, statistics, quotes
    factual_patterns = len(re.findall(
        r'\b\d+[\.,]?\d*\s*(%|percent|million|billion|thousand)\b', text_lower
    ))
    factual_score = min(100, factual_patterns * 15 + (30 if has_sources else 0))

    # Neutral tone: penalise emotional language
    neutral_score = max(0, 80 - (emotional_count * 20))

    # Citation quality: rewards explicit source references
    citation_patterns = len(re.findall(
        r'\b(according to|cited by|study|report|research|published)\b',
        text_lower
    ))
    citation_score = min(100, citation_patterns * 20)

    # Writing consistency: rewards longer, structured text
    writing_score = min(100, int((word_count / 500) * 60) + (20 if has_author else 0))

    # Overall reliability: weighted average of the above
    overall_score = int(
        (factual_score * 0.30) +
        (neutral_score * 0.25) +
        (citation_score * 0.25) +
        (writing_score * 0.20)
    )

    return {
        "verified_facts": {
            "has_date":             has_date,
            "has_author":           has_author,
            "has_sources":          has_sources,
            "emotional_language":   has_emotional_language,
            "headline_matches_body": headline_matches_body
        },
        "reliability_scores": {
            "factual_language":     factual_score,
            "neutral_tone":         neutral_score,
            "citation_quality":     citation_score,
            "writing_consistency":  writing_score,
            "overall_reliability":  overall_score
        },
        "word_count": word_count
    }


# ── Route: Home page ─────────────────────────────────────────
# When the user visits http://localhost:5000 in their browser,
# Flask returns the index.html file from the templates/ folder.

@app.route("/")
def home():
    return render_template("index.html")


# ── Route: Analyze article ───────────────────────────────────
# The JavaScript in index.html sends a POST request here
# with the article text. We return a JSON response.

@app.route("/analyze", methods=["POST"])
def analyze():

    # Check that the model was loaded successfully
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Please run train_model.py first."
        }), 500

    # Get the article text sent from the browser
    data = data = request.get_json()
    article_text = data.get("text", "").strip()

    # Basic input validation
    if not article_text:
        return jsonify({"error": "No text provided."}), 400

    if len(article_text.split()) < 10:
        return jsonify({"error": "Article is too short. Please paste more text."}), 400

    # ── Run the ML model ─────────────────────────────────────
    start_time = time.time()

    # Step 1: Transform the text into TF-IDF numbers
    text_tfidf = vectorizer.transform([article_text])

    # Step 2: Predict the label (0 = Real, 1 = Fake)
    prediction = model.predict(text_tfidf)[0]

    # Step 3: Get the probability scores for both classes
    # proba[0] = probability of being REAL
    # proba[1] = probability of being FAKE
    probabilities = model.predict_proba(text_tfidf)[0]

    elapsed_time = round(time.time() - start_time, 2)

    # ── Run extra text analysis ───────────────────────────────
    analysis = analyze_text(article_text)

    # ── Build the response ────────────────────────────────────
    response = {
        "prediction":    "Fake" if prediction == 1 else "Real",
        "fake_percent":  round(probabilities[1] * 100, 1),
        "real_percent":  round(probabilities[0] * 100, 1),
        "elapsed_time":  elapsed_time,
        "word_count":    analysis["word_count"],
        "verified_facts":    analysis["verified_facts"],
        "reliability_scores": analysis["reliability_scores"]
    }

    return jsonify(response)


# ── Start the server ─────────────────────────────────────────
# debug=True means the server restarts automatically when you
# edit the code, and shows detailed error messages.
# Only use debug=True during development, not in production.

if __name__ == "__main__":
    print("\nTrueWire is running!")
    print("Open your browser at: http://localhost:5000\n")
    app.run(debug=True)
