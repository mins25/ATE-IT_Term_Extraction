# ATE-IT_Term_Extraction
Minseok KIM

# Term Extraction Pipeline (CRF-Based)

The term-extraction system is built around a supervised sequence-labeling architecture that identifies domain-specific terms within Italian text. The core of the system is a Conditional Random Field (CRF) model trained on manually annotated sentences. The pipeline follows a classical yet proven approach: feature-rich token representation, BIO sequence labeling, and structured prediction optimized with L-BFGS.

This project provides a complete pipeline including data loading, feature extraction, BIO labeling, training, cross-validation, final model training, prediction on test data, and evaluation.

---

## 1. Project Structure

```
/projet_extraction_termes/
│
├── data/
│   ├── subtask_a_train.json            Training set (with gold terms)
│   ├── subtask_a_dev.json              Development set (with gold terms)
│   ├── test.json                       Test set (without gold terms)
│   └── train_dev.json                  Combined training set for final prediction (with gold terms)
│
├── models/                             Saved CRF models
│
├── extract_terms.py                    Main pipeline script
│
└── results/
    └── test_predictions.json           Final predictions
```

---

## 2. Installation and Environment Setup

### 2.1 Create and activate a virtual environment

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```
python3 -m venv venv
source venv/bin/activate
```

---

### 2.2 Install required packages

```
pip install -r requirements.txt
```

You may also install the SpaCy model manually:

```
python -m spacy download it_core_news_lg
```

---

## 3. Running the Pipeline
From the project root:

```
python extract_terms.py
```

The script performs:

1. Cross-validation on the train + dev set  
2. Comparison with baseline performance  
3. Training of the final CRF model on the full dataset  
4. Predictions on the test file  
5. Export of predictions to:

```
results/test_predictions.json
```

---

## 4. Final Results vs Baseline

| Metric     | Model Score | Baseline |
|------------|-------------|----------|
| micro_p    | 0.799       | 0.439    |
| micro_r    | 0.698       | 0.616    |
| micro_f1   | 0.745       | 0.513    |
| type_p     | 0.761       | 0.372    |
| type_r     | 0.628       | 0.636    |
| type_f1    | 0.687       | 0.470    |

---

## 5. Notes and possible future improvements

- The current model relies heavily on SpaCy's `it_core_news_lg` embeddings and POS tags  
- CRF hyperparameters can be tuned for better performance  
- Adding character-level or transformer-based features 
