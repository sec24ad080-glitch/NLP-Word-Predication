# Next-Word Prediction Model Using a Given Text Corpus
### *College NLP Mini Project Prototype*

An end-to-end interactive statistical Natural Language Processing (NLP) prototype built to predict the next word in a sequence using **Unigram, Bigram, and Trigram N-gram Language Models** with **Laplace (Add-k) Smoothing**.

---

## 🌟 Key Features

1. **Dynamic Model Architecture**:
   - **Unigram ($n=1$)**: Frequency-based baseline model.
   - **Bigram ($n=2$)**: 1st-order Markov chain predicting the next word from the previous word.
   - **Trigram ($n=3$)**: 2nd-order Markov chain predicting the next word from the prior two words.
2. **Laplace Smoothing ($Add\text{-}k$)**:
   - Eliminates the zero-frequency problem for unseen context words and n-grams.
3. **Corpus Management**:
   - Upload any custom `.txt` corpus file.
   - Choose from 3 built-in domain corpora (*AI & NLP*, *Science & Tech*, *Classic Literature*).
   - Live editable text area for on-the-fly model retraining.
4. **Interactive Prediction Playground**:
   - Real-time Top-$K$ candidate word ranking with conditional probabilities and counts.
   - Interactive probability bar charts.
   - Autonomous multi-word sentence continuation / autocomplete.
5. **Empirical Model Evaluation**:
   - **Perplexity ($PP$)**: Measures language model uncertainty (lower is better).
   - **Top-1 / Top-3 / Top-5 Accuracy**: Real test-set predictive accuracy.
   - Side-by-side comparative benchmarking across all 3 models.

---

## 🛠️ Technology Stack

- **Language:** Python 3.11+
- **NLP & Preprocessing:** NLTK
- **Data Structures & Processing:** Pandas, NumPy
- **Model Evaluation & Splitting:** Scikit-learn
- **Frontend Dashboard:** Streamlit

---

## 🚀 How to Run the Prototype

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Application
```bash
streamlit run app.py
```

### 3. Run Backend Unit Tests
```bash
python test_backend.py
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit Web Interface & Dashboard
├── model.py                # N-gram Language Models (Unigram, Bigram, Trigram) & Smoothing
├── preprocessor.py         # NLTK Tokenization, Sentence Boundary & Preprocessing
├── evaluator.py            # Perplexity and Top-K Accuracy Computation
├── sample_data.py          # Curated Demonstration Corpora
├── test_backend.py         # Automated Test Suite
├── requirements.txt        # Python Dependencies
├── sample_corpora/         # Pre-built Text Corpus Files
│   └── ai_and_nlp.txt
└── README.md               # Project Documentation
```

---

## 📐 Mathematical Formulation

### 1. Unigram Probability
$$P(w_i) = \frac{C(w_i) + k}{N + k \cdot |V|}$$

### 2. Bigram Conditional Probability
$$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k \cdot |V|}$$

### 3. Trigram Conditional Probability
$$P(w_i | w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i) + k}{C(w_{i-2}, w_{i-1}) + k \cdot |V|}$$

### 4. Perplexity ($PP$)
$$PP(W) = \exp\left(-\frac{1}{N}\sum_{i=1}^N \ln P(w_i | \text{context})\right)$$
