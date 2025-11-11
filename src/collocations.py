import argparse
from pathlib import Path
from collections import Counter
import math
import pandas as pd
from utils import load_config


def bigrams(tokens):
    return [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]


def compute_pmi(tokens_list, min_count=2):
    unigram_counts = Counter()
    bigram_counts = Counter()
    total_unigrams = 0
    for toks in tokens_list:
        unigram_counts.update(toks)
        bigram_counts.update(bigrams(toks))
        total_unigrams += len(toks)

    # Filter by count
    rows = []
    for (w1,w2), c12 in bigram_counts.items():
        if c12 < min_count:
            continue
        c1 = unigram_counts[w1]
        c2 = unigram_counts[w2]
        if c1 < min_count or c2 < min_count:
            continue
        p12 = c12 / max(1, total_unigrams)
        p1 = c1 / max(1, total_unigrams)
        p2 = c2 / max(1, total_unigrams)
        pmi = math.log2(p12 / (p1 * p2 + 1e-12) + 1e-12)
        rows.append({
            'bigram': f'{w1}_{w2}',
            'count': c12,
            'pmi': pmi
        })
    if not rows:
        return pd.DataFrame(columns=['bigram','count','pmi'])
    return pd.DataFrame(rows).sort_values('pmi', ascending=False)


def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_pickle(Path(cfg['processed_dir']) / 'processed.pkl')
    if df.empty:
        print('No data for collocations.')
        return
    pre = df[df.period=='pre']
    post = df[df.period=='post']
    pre_df = compute_pmi(pre.tokens.tolist(), min_count=2)
    post_df = compute_pmi(post.tokens.tolist(), min_count=2)
    Path('outputs/tables').mkdir(parents=True, exist_ok=True)
    pre_df.head(50).to_csv('outputs/tables/collocations_pre.csv', index=False)
    post_df.head(50).to_csv('outputs/tables/collocations_post.csv', index=False)
    print('Collocations saved.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
