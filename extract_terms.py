import json
import spacy
import pickle
import subprocess
from tqdm import tqdm
from pathlib import Path
from sklearn_crfsuite import CRF, metrics
from sklearn.model_selection import KFold
import pandas as pd
import codecs
from collections import defaultdict

# =============================================================
# PARTIE 1: CHARGEMENT DES DONNÉES
# =============================================================

def load_data(file_path):
    """Charge les phrases et les termes gold"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)
    data = []
    for item in data_json["data"]:
        data.append({
            "document_id": item["document_id"],
            "paragraph_id": item["paragraph_id"],
            "sentence_id": item["sentence_id"],
            "sentence_text": item["sentence_text"],
            "term_list": [t.lower().strip() for t in item.get("term_list", [])]
        })
    return data

# =============================================================
# PARTIE 2: EXTRACTION DE FEATURES
# =============================================================

class TokenFeatureExtractor:
    def __init__(self, model_name="it_core_news_lg"):
        print(f"📥 Chargement du modèle SpaCy: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import os
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)
        print("✅ Modèle SpaCy chargé")

    def token2features(self, sent, i):
        token = sent[i]
        features = {
            'bias': 1.0,
            'word.lower()': token.text.lower(),
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'postag': token.pos_,
            'dep': token.dep_,
            'lemma': token.lemma_,
            'shape': token.shape_,
            'prefix2': token.text[:2].lower(),
            'suffix2': token.text[-2:].lower(),
            'prefix3': token.text[:3].lower(),
            'suffix3': token.text[-3:].lower(),
            'is_stop': token.is_stop,
        }

        if i > 0:
            token1 = sent[i - 1]
            features.update({
                '-1:word.lower()': token1.text.lower(),
                '-1:postag': token1.pos_,
                '-1:lemma': token1.lemma_,
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            token1 = sent[i + 1]
            features.update({
                '+1:word.lower()': token1.text.lower(),
                '+1:postag': token1.pos_,
                '+1:lemma': token1.lemma_,
            })
        else:
            features['EOS'] = True
        return features

    def sent2features(self, sent):
        return [self.token2features(sent, i) for i in range(len(sent))]

# =============================================================
# PARTIE 3: LABELS BIO
# =============================================================

def sentence_to_BIO(sent_text, term_list, nlp):
    doc = nlp(sent_text)
    labels = ["O"] * len(doc)
    text_lower = sent_text.lower()

    for term in term_list:
        term_lower = term.lower()
        start = text_lower.find(term_lower)
        if start == -1:
            continue
        end = start + len(term_lower)
        for i, token in enumerate(doc):
            if token.idx >= start and token.idx < end:
                if token.idx == start:
                    labels[i] = "B"
                else:
                    labels[i] = "I"
    return doc, labels

# =============================================================
# PARTIE 4: MODELE CRF
# =============================================================

class CRFTermExtractor:
    def __init__(self, model_name="it_core_news_lg"):
        self.feature_extractor = TokenFeatureExtractor(model_name)
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=200,
            all_possible_transitions=True
        )

    def prepare_data(self, file_path):
        data = load_data(file_path)
        X, y = [], []
        for item in tqdm(data, desc="Préparation des données"):
            sent_text = item["sentence_text"]
            term_list = item["term_list"]
            doc, labels = sentence_to_BIO(sent_text, term_list, self.feature_extractor.nlp)
            X.append(self.feature_extractor.sent2features(doc))
            y.append(labels)
        return X, y, data

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_sentence(self, sentence):
        doc = self.feature_extractor.nlp(sentence)
        X = [self.feature_extractor.sent2features(doc)]
        y_pred = self.model.predict(X)[0]
        return self.bio_to_terms(doc, y_pred)

    def bio_to_terms(self, doc, labels):
        terms = []
        current = []
        for token, label in zip(doc, labels):
            if label == "B":
                if current:
                    terms.append(" ".join(current))
                    current = []
                current.append(token.text)
            elif label == "I":
                current.append(token.text)
            else:
                if current:
                    terms.append(" ".join(current))
                    current = []
        if current:
            terms.append(" ".join(current))
        return [t.lower() for t in terms]

    def save(self, path="models/term_crf.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path="models/term_crf.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

# =============================================================
# PARTIE 5: EVALUATION CUSTOM
# =============================================================

def micro_f1_score(gold_standard, system_output):
    total_tp, total_fp, total_fn = 0, 0, 0
    for gold, system in zip(gold_standard, system_output):
        gold_set, system_set = set(gold), set(system)
        total_tp += len(gold_set & system_set)
        total_fp += len(system_set - gold_set)
        total_fn += len(gold_set - system_set)
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def type_f1_score(gold_standard, system_output):
    gold_terms = set(sum(gold_standard, []))
    system_terms = set(sum(system_output, []))
    tp = len(gold_terms & system_terms)
    fp = len(system_terms - gold_terms)
    fn = len(gold_terms - system_terms)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

# =============================================================
# PARTIE 6: CROSS VALIDATION
# =============================================================

def cross_validate_crf(data_file, n_splits=10):
    print("\n🔁 Cross-validation sur tout le dataset...\n")
    crf_extractor = CRFTermExtractor()
    data = load_data(data_file)
    X, y, data_full = crf_extractor.prepare_data(data_file)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        test_data = [data_full[i] for i in test_idx]

        model = CRFTermExtractor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        gold_terms, pred_terms = [], []
        for d, y_p in zip(test_data, y_pred):
            doc = model.feature_extractor.nlp(d["sentence_text"])
            pred_terms.append(model.bio_to_terms(doc, y_p))
            gold_terms.append(d["term_list"])

        micro_p, micro_r, micro_f1 = micro_f1_score(gold_terms, pred_terms)
        type_p, type_r, type_f1 = type_f1_score(gold_terms, pred_terms)
        print(f"Micro-F1: {micro_f1:.3f} | Type-F1: {type_f1:.3f}")
        results.append((micro_p, micro_r, micro_f1, type_p, type_r, type_f1))

    df = pd.DataFrame(results, columns=["micro_p", "micro_r", "micro_f1", "type_p", "type_r", "type_f1"])
    print("\n📊 Résultats cross-validation (moyennes):")
    print(df.mean())
    return df.mean().to_dict()

# =============================================================
# PARTIE 7: PREDICTION SUR TEST
# =============================================================

def predict_on_test(train_dev_file, test_file, model_path, output_file):
    crf = CRFTermExtractor()
    X_train, y_train, _ = crf.prepare_data(train_dev_file)
    print("\n🚀 Entraînement final sur tout le dataset...")
    crf.fit(X_train, y_train)
    crf.save(model_path)

    print("\n📄 Prédiction sur le fichier test...")
    with open(test_file, "r", encoding="utf-8") as f:
        test_json = json.load(f)

    results = []
    for item in tqdm(test_json["data"], desc="Prédiction test"):
        sentence = item["sentence_text"]
        terms = crf.predict_sentence(sentence)
        results.append({
            "document_id": item["document_id"],
            "paragraph_id": item["paragraph_id"],
            "sentence_id": item["sentence_id"],
            "term_list": terms
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"data": results}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Résultats sauvegardés dans: {output_file}")

# =============================================================
# PARTIE 8: MAIN
# =============================================================

def main():
    train_dev_file = "data/train_dev.json"  # fusion de train et dev
    test_file = "data/test.json"
    model_path = "models/final_crf.pkl"
    output_file = "results/test_predictions.json"

    # 1. Cross-validation
    scores = cross_validate_crf(train_dev_file, n_splits=10)

    # 2. Comparaison baseline
    baseline = {
        'micro_p': 0.439, 'micro_r': 0.616, 'micro_f1': 0.513,
        'type_p': 0.372, 'type_r': 0.636, 'type_f1': 0.470
    }
    print("\n📈 Comparaison avec baseline:")
    for k, v in scores.items():
        print(f"{k}: {v:.3f} (baseline {baseline.get(k, 0):.3f})")

    # 3. Entraînement final + prédiction test
    predict_on_test(train_dev_file, test_file, model_path, output_file)

    print("\n🎯 Pipeline complet terminé avec succès!")

# =============================================================
if __name__ == "__main__":
    main()
