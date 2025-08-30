# Cyberbullying Classification Using Machine Learning

This repository contains a **machine-learning** for detecting cyberbullying in social‑media text.  
It is centered around a single Jupyter notebook, `cyberbullying-ml-project.ipynb`, which walks through data loading, preprocessing, model training, evaluation, and inference.

> Works for **binary** (bullying vs. not) or **multi‑class** setups (e.g., `age`, `religion`, `gender`, `ethnicity`, `other`, `not_cyberbullying`). Adjust label names in the notebook as needed.

---

## 🚀 What’s Included
- `cyberbullying-ml-project.ipynb` – end‑to‑end notebook (preprocess → train → evaluate → predict).
- (optional) `data/` – place your CSV(s) here.
- (optional) `models/` – saved artifacts (vectorizer / tokenizer, trained model).
- (optional) `reports/` – confusion matrices, metrics, and plots exported from the notebook.

---

## 📂 Data Format
Place your dataset in `data/` (you can name it freely). A common format is a CSV with at least:
- `text` – the post/message content
- `label` – ground‑truth class (string or int)

Example (CSV):
```csv
text,label
"You're such a loser",bullying
"Have a great day!",not_cyberbullying
```

> Large/locked datasets should **not** be committed. Keep small samples only and document where to download the full data.

---

## 🔧 Environment Setup
Create a virtual environment (any manager is fine). Example with `venv`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -U pip
pip install jupyter pandas numpy scikit-learn matplotlib seaborn nltk spacy emoji imbalanced-learn
# OPTIONAL (for transformer models like DistilBERT/BERT)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
```

> The notebook contains switches to use either a **classic ML pipeline** (TF‑IDF + Logistic Regression / SVM / RandomForest) or a **transformer** (e.g., DistilBERT). If you only need classic ML, you can skip the `torch/transformers` installs.

---

## ▶️ Run the Notebook
```bash
jupyter notebook cyberbullying-ml-project.ipynb
```
In the notebook, you can:
- Clean text (URLs/mentions/hashtags/emojis), lowercase, remove stopwords
- Tokenize & vectorize (e.g., TF‑IDF) or use transformer tokenizers
- Handle class imbalance (class weights or `imblearn` resampling)
- Train baseline + tuned models
- Evaluate with **Accuracy, Precision, Recall, F1 (macro), ROC‑AUC** and confusion matrices
- Save artifacts (vectorizer/model) to `models/`

---
