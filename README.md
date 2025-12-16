# Credit Card Fraud Detection (ML Portfolio Project)
![Python](https://img.shields.io/badge/Python-ML-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-orange)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

Detect fraudulent credit card transactions using a lightweight, interpretable **Logistic Regression** baseline in Python.  
This project focuses on handling **class imbalance** via **under-sampling**, then training and evaluating the model.

---

## ✨ Highlights
- Built an end-to-end ML pipeline using **NumPy, Pandas, scikit-learn**
- Addressed extreme class imbalance using **under-sampling**
- Achieved **94.2% training accuracy** and **93.9% test accuracy** on the balanced dataset
- Clean, reproducible notebook workflow ready for extension (metrics, tuning, deployment)

---

## 🔎 Problem Statement
Credit card fraud detection is a classic **highly imbalanced classification** problem: fraudulent transactions are rare compared to legitimate ones.  
A naive model can achieve high accuracy by predicting “not fraud” for everything—so proper sampling + evaluation matters.

**Goal:** classify transactions as **Fraud (1)** or **Legit (0)**.

---

## 🧠 Approach
### 1) Data Ingestion + Basic EDA
- Load dataset into Pandas
- Validate schema, missing values (if any), and class distribution
- Explore label imbalance and basic feature behavior (e.g., `Amount` patterns)

### 2) Handle Class Imbalance (Under-sampling)
Because the dataset is dominated by non-fraud cases, the notebook:
- Separates fraud and non-fraud records
- Randomly samples non-fraud records to match the fraud count
- Combines them into a **balanced training dataset**

✅ Pros: simple, fast, avoids misleading accuracy  
⚠️ Cons: discards many valid non-fraud samples (may reduce generalization)

### 3) Train/Test Split
- Split into training and test sets
- Separate features (`X`) and label (`y`)

### 4) Model Training (Logistic Regression)
- Train **Logistic Regression** as an interpretable baseline
- Tune solver/iterations if needed for convergence (`max_iter`)

### 5) Evaluation
- Measure accuracy on train and test sets  
> In real fraud systems, you’d also prioritize **precision/recall/F1/ROC-AUC** and threshold tuning.

---

## 📈 Results
| Metric | Score |
|-------|------:|
| Train Accuracy | **94.2%** |
| Test Accuracy  | **93.9%** |

---

## 🧰 Tech Stack
- **Python**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Matplotlib**
- **Seaborn** (for visualizations)

---

## 📦 Dataset
Expected file name:
- `creditcard.csv`

Label column:
- `Class` → `0` (Legit), `1` (Fraud)

> Dataset not included in repo. Add it locally under `data/`.

---

## ▶️ How to Run
### Option A: Jupyter Notebook
1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
2. Place the dataset:
   ```bash
   data/creditcard.csv
3. Open and run the notebook:
   ```bash
   jupyter notebook
Then run: `CreditCardFraudDetection.ipynb`

## 🗂️ Recommended Repo Structure

```text
CreditCard-Fraud-Detection/
├── CreditCardFraudDetection.ipynb
├── README.md
├── data/
│   └── creditcard.csv 
├── images/
│   └── class_distribution.png (optional)
└── requirements.txt          (optional)
