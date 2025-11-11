import argparse
from pathlib import Path
import pandas as pd
from gensim import corpora, models
from utils import load_config
from collections import defaultdict

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_pickle(Path(cfg["processed_dir"]) / "processed.pkl")
    # Build dictionary
    frequency = defaultdict(int)
    for tokens in df.tokens:
        for t in tokens:
            frequency[t] += 1
    filtered_docs = [[t for t in tokens if frequency[t] >= cfg["lda"]["min_token_freq"]] for tokens in df.tokens]

    dictionary = corpora.Dictionary(filtered_docs)
    corpus = [dictionary.doc2bow(text) for text in filtered_docs]
    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=cfg["lda"]["num_topics"],
        random_state=cfg["lda"]["random_state"],
        passes=cfg["lda"]["passes"]
    )

    # Topic distributions per document
    topic_rows = []
    for i, bow in enumerate(corpus):
        dist = lda.get_document_topics(bow, minimum_probability=0.0)
        row = {f"topic_{tid}": prob for tid, prob in dist}
        row["period"] = df.period.iloc[i]
        topic_rows.append(row)
    topic_df = pd.DataFrame(topic_rows)

    # Aggregate by period
    agg = topic_df.groupby("period").mean()
    # Ensure output directories exist
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    agg.to_csv("outputs/tables/topic_period_distribution.csv")

    # Save top words per topic
    top_words = []
    for t in range(cfg["lda"]["num_topics"]):
        words = lda.show_topic(t, topn=10)
        top_words.append({
            "topic": t,
            "terms": ", ".join([w for w, _ in words])
        })
    import csv
    with open("outputs/tables/lda_topics.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["topic","terms"])
        writer.writeheader()
        for r in top_words:
            writer.writerow(r)

    print("LDA topic distributions and top terms saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)