import argparse
import pandas as pd
from pathlib import Path
from collections import Counter
from math import log
from utils import load_config

def get_ngrams(tokens, n=2):
    return ["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def compute_shift(df, n=2, top_k=30):
    pre = df[df.period=="pre"]
    post = df[df.period=="post"]

    pre_counts = Counter()
    post_counts = Counter()
    for toks in pre.tokens:
        pre_counts.update(get_ngrams(toks, n))
    for toks in post.tokens:
        post_counts.update(get_ngrams(toks, n))

    # Log-likelihood style score (simple ratio difference)
    rows = []
    total_pre = sum(pre_counts.values()) + 1e-9
    total_post = sum(post_counts.values()) + 1e-9

    all_ngrams = set(pre_counts) | set(post_counts)
    for ng in all_ngrams:
        p_pre = pre_counts[ng] / total_pre
        p_post = post_counts[ng] / total_post
        diff = p_post - p_pre
        # Weighted difference by absolute change magnitude
        score = diff * log((p_post + 1e-9) / (p_pre + 1e-9))
        rows.append({
            "ngram": ng,
            "pre_count": pre_counts[ng],
            "post_count": post_counts[ng],
            "score": score,
            "raw_diff": diff
        })
    df_shift = pd.DataFrame(rows)
    df_shift.sort_values("score", ascending=False, inplace=True)
    return df_shift.head(top_k), df_shift.tail(top_k)

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_pickle(Path(cfg["processed_dir"]) / "processed.pkl")
    top_pos, top_neg = compute_shift(df, n=2, top_k=40)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    top_pos.to_csv("outputs/tables/ngram_shift_positive.csv", index=False)
    top_neg.to_csv("outputs/tables/ngram_shift_negative.csv", index=False)
    print("N-gram shift tables written.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)