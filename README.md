# 🧠 Hybrid AI Fake News Intelligence System

A futuristic AI-powered fake news detector built with **Machine Learning + Gemini AI**.

## 🔥 Features
- **TF-IDF + Logistic Regression** — trained on 40,000+ real/fake news articles
- **Confidence Scoring** — probability-based trust metric with low/high confidence warnings
- **Gemini AI Integration** — article summarization + credibility analysis via Gemini 2.0 Flash
- **Futuristic dark UI** — glassmorphism, neon gradients, glow effects

## 📁 Project Structure
```
fakenewsdetection/
├── app.py              # Streamlit application
├── train.py            # Model training script
├── requirements.txt    # Dependencies
├── model/              # Saved model (generated after training)
│   ├── model.pkl
│   └── vectorizer.pkl
└── data/               # Dataset (not included — see below)
    ├── Fake.csv
    └── True.csv
```

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/fakenewsdetection.git
cd fakenewsdetection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download the [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle and place the files in the `data/` folder:
```
data/Fake.csv
data/True.csv
```

### 4. Train the model
```bash
python train.py
```
This saves `model/model.pkl` and `model/vectorizer.pkl`.

### 5. Set your Gemini API key *(optional)*
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
```

### 6. Run the app
```bash
python -m streamlit run app.py
```
Open **http://localhost:8501** in your browser.

> ⚠️ Set the Gemini API key **in the same terminal session** before running the app.

## 🧪 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | **98.99%** |
| Weighted F1 | **0.99** |

**Hyperparameters:**
- `TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")`
- `LogisticRegression(max_iter=2000, class_weight="balanced")`

## 🔑 Environment Variables
| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com) |

## 🛠 Tech Stack
- [Streamlit](https://streamlit.io) — UI framework
- [scikit-learn](https://scikit-learn.org) — ML model
- [Google Gemini](https://ai.google.dev) — AI insights
- [joblib](https://joblib.readthedocs.io) — model serialization

## 📄 License
MIT License — © 2026 Birahadeeshwaran S.
