# SVM-Based Email Spam Detection

A machine learning pipeline that classifies SMS/email messages as spam or ham (legitimate), comparing four classifiers — Logistic Regression, Naive Bayes, SVM, and Random Forest — on TF-IDF text features. The Linear SVM is the best-performing model (spam F1 = 0.9514, ROC-AUC = 0.9944).

## Project Description

This project builds an end-to-end text classification pipeline: load and clean a labeled message dataset, extract TF-IDF features, train and cross-validate four supervised classifiers, and evaluate them on accuracy, precision, recall, F1, and ROC-AUC — with spam explicitly scored as the positive class throughout, since the dataset is moderately imbalanced (86.6% ham / 13.4% spam). The best model is used in a `verify_email()` function that classifies a new message end-to-end, directly from the fitted in-memory model (no files saved to disk).

Full methodology, results, and discussion are in the project report (`.docx`).

## Dataset

- **File**: `mail_data.csv`
- **Size**: 5,572 labeled messages, two columns (`Category`: spam/ham, `Message`: raw text)
- **Source**: This closely matches the well-known [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (Almeida & Hidalgo, hosted on Kaggle/UCI), a public collection of 5,574 SMS messages labeled spam/ham. If you're using a different copy of the file, the exact row count and content may vary slightly.
- Not included in this submission — place `mail_data.csv` in the same directory as the notebook before running it (see Setup below).

## Technologies Used

- Python 3
- pandas, numpy — data loading and manipulation
- scikit-learn — TF-IDF vectorization, model training, cross-validation, evaluation metrics
- matplotlib, seaborn — charts (model comparison, confusion matrices, ROC curves)
- Jupyter Notebook

## Setup / Run Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add the dataset**
   Place `mail_data.csv` in the same folder as the notebook. The notebook expects two columns: `Category` (spam/ham) and `Message` (text).

3. **Update the file path if needed**
   The notebook loads the dataset with:
   ```python
   df = pd.read_csv("mail_data.csv")
   ```
   Adjust this path if your CSV is located elsewhere.

4. **Run the notebook**
   ```bash
   jupyter notebook
   ```
   Open the `.ipynb` file and run all cells in order, top to bottom.

## Key Information

- **Models compared**: Logistic Regression, Multinomial Naive Bayes, SVM (linear kernel), Random Forest
- **Feature extraction**: TF-IDF (`TfidfVectorizer`, English stop words removed)
- **Train/test split**: 80% / 20%, stratified on the label, `random_state=3`
- **Evaluation**: 5-fold cross-validation (spam-focused F1) + held-out test set (accuracy, spam precision/recall/F1, ROC-AUC)
- **Best model**: SVM (linear kernel) — spam F1 = 0.9514, accuracy = 98.74%, ROC-AUC = 0.9944
- **Known limitation**: only 4 of the 6 algorithms used in the base paper this project is compared against (Kontsewaya et al., 2021) are included — K-Nearest Neighbors and Decision Tree are not evaluated here. See the project report for full discussion, limitations, and future scope.

## Files in This Submission

| File | Description |
|---|---|
| `*.ipynb` | Complete project code — data loading, preprocessing, model training, evaluation, ROC/AUC, deployment demo |
| `requirements.txt` | Python dependencies |
| `*_ProjectReport.docx` | Full project documentation and results |
| `README.md` | This file |
