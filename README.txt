================================================================
 TrueWire — Fake News Detector
 Informed Trust through Objective Clarity
================================================================

FOLDER STRUCTURE
----------------
fake-news-detector/
├── app.py              ← Flask web server (run this to start the site)
├── train_model.py      ← Train the ML model (run this once first)
├── dataset/
│   └── fake_news.csv   ← Place your downloaded dataset here
├── model/
│   ├── model.pkl       ← Saved after training (auto-created)
│   └── vectorizer.pkl  ← Saved after training (auto-created)
├── templates/
│   └── index.html      ← The webpage (Flask serves this)
└── static/
    └── style.css       ← All the CSS styling


STEP 1 — Install Python libraries
-----------------------------------
Open your terminal and run:

    pip install flask scikit-learn pandas numpy

(pickle is built into Python — no install needed)


STEP 2 — Download the dataset
-------------------------------
1. Go to: https://www.kaggle.com/c/fake-news/data
2. Download "train.csv"
3. Rename it to "fake_news.csv"
4. Place it inside the dataset/ folder:
       fake-news-detector/dataset/fake_news.csv

The CSV must have these columns:
    id | title | author | text | label
    (label: 0 = Real, 1 = Fake)


STEP 3 — Train the model
--------------------------
In your terminal, navigate to the project folder:

    cd fake-news-detector

Then run:

    python train_model.py

This will:
  - Load and process the dataset
  - Train the TF-IDF + Logistic Regression model
  - Print the accuracy score
  - Save model.pkl and vectorizer.pkl inside model/

You only need to run this ONCE.


STEP 4 — Start the Flask server
---------------------------------
    python app.py

You will see:
    TrueWire is running!
    Open your browser at: http://localhost:5000


STEP 5 — Use the website
--------------------------
1. Open your browser
2. Go to: http://localhost:5000
3. Paste any news article into the text area
4. Click "Analyze article"
5. See the result: Fake or Real, with percentage confidence,
   verified facts, and source reliability scores


HOW EACH FILE WORKS
--------------------
train_model.py
  Loads the CSV, combines title + author + text into one string,
  applies TF-IDF vectorization, trains Logistic Regression,
  evaluates accuracy, and saves the model + vectorizer with pickle.

app.py
  Starts the Flask server. Loads the saved model on startup.
  Has two routes:
    GET  /          → serves index.html
    POST /analyze   → receives article text, runs the model,
                      returns a JSON prediction response.

templates/index.html
  The webpage. Contains the textarea, analyze button, result
  cards, and all the JavaScript that talks to Flask via fetch().

static/style.css
  All the styling. Uses the TrueWire color palette:
  #1A365D (navy), #E2E8F0 (silver), #4F2E00 (brown), #718096 (slate).


TROUBLESHOOTING
----------------
"Model not loaded" error
  → Run train_model.py first before app.py

"Dataset not found" error
  → Make sure fake_news.csv is inside the dataset/ folder

"Could not connect to the server"
  → Make sure app.py is running in your terminal

Port already in use
  → Change the port in app.py: app.run(debug=True, port=5001)
  → Then visit: http://localhost:5001

================================================================
