import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from utils import load_config


def train_eval_models(texts, labels):
    # Vectorize
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)
    X = tfidf.fit_transform(texts)
    y = np.array([1 if l == 'post' else 0 for l in labels])

    # Split (handle tiny datasets robustly)
    n_samples = len(labels)
    if n_samples < 4 or len(set(labels)) < 2:
        return {
            "error": "Insufficient data for supervised baseline (need >=4 samples and at least two classes)"
        }, tfidf, X, y

    test_size = 0.5 if n_samples <= 10 else 0.2
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    results = {}

    # Logistic Regression (liblinear for small, else saga)
    lr = LogisticRegression(max_iter=1000, solver='liblinear')
    lr.fit(Xtr, ytr)
    lr_pred = lr.predict(Xte)
    try:
        lr_proba = lr.decision_function(Xte)
        lr_auc = roc_auc_score(yte, lr_proba)
    except Exception:
        lr_auc = None
    results['logreg'] = {
        'accuracy': float(accuracy_score(yte, lr_pred)),
        'f1_macro': float(f1_score(yte, lr_pred, average='macro')),
        'roc_auc': float(lr_auc) if lr_auc is not None else None,
        'confusion_matrix': confusion_matrix(yte, lr_pred).tolist()
    }

    # Top features by coefficient
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = lr.coef_[0]
    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]
    top_features = pd.DataFrame({
        'feature': np.concatenate([feature_names[top_pos_idx], feature_names[top_neg_idx]]),
        'coef': np.concatenate([coefs[top_pos_idx], coefs[top_neg_idx]])
    })

    # Linear SVM
    svm = LinearSVC()
    svm.fit(Xtr, ytr)
    svm_pred = svm.predict(Xte)
    results['linear_svm'] = {
        'accuracy': float(accuracy_score(yte, svm_pred)),
        'f1_macro': float(f1_score(yte, svm_pred, average='macro')),
        'roc_auc': None,  # decision_function scale not calibrated for binary AUC directly here
        'confusion_matrix': confusion_matrix(yte, svm_pred).tolist()
    }

    return {"results": results, "top_features": top_features.to_dict(orient='records')}, tfidf, X, y


def plot_lsa(X, labels, out_path):
    # 2D LSA projection
    try:
        svd = TruncatedSVD(n_components=2, random_state=42)
        X2 = svd.fit_transform(X)
        plt.figure(figsize=(6,5))
        colors = {'pre': '#1f77b4', 'post': '#ff7f0e'}
        labs = np.array(labels)
        for cls in ['pre','post']:
            mask = (labs == cls)
            if mask.sum() == 0:
                continue
            plt.scatter(X2[mask,0], X2[mask,1], s=40, alpha=0.8, label=cls, c=colors.get(cls, '#999999'))
        plt.legend()
        plt.title('TF-IDF LSA (2D)')
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        # Silently skip plotting on very tiny datasets
        pass


def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_pickle(Path(cfg["processed_dir"]) / "processed.pkl")
    if df.empty or 'text' not in df or 'period' not in df:
        print("No data available for baselines.")
        return

    texts = df['text'].astype(str).tolist()
    labels = df['period'].astype(str).tolist()

    metrics, tfidf, X, y = train_eval_models(texts, labels)
    Path('outputs/tables').mkdir(parents=True, exist_ok=True)
    Path('outputs/figures').mkdir(parents=True, exist_ok=True)

    with open('outputs/tables/bow_baselines_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    if isinstance(metrics, dict) and 'top_features' in metrics:
        tf_df = pd.DataFrame(metrics['top_features'])
        tf_df.to_csv('outputs/tables/tfidf_top_features.csv', index=False)

    plot_lsa(X, np.array(labels), 'outputs/figures/lsa_scatter.png')
    print("BoW baselines completed.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
