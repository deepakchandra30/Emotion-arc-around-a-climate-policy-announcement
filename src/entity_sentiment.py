import argparse
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import load_config

TARGET_ENTITY_TYPES = {"ORG","PERSON","GPE"}

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_pickle(Path(cfg["processed_dir"]) / "processed.pkl")
    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for _, row in df.iterrows():
        text = row.text
        sent_val = analyzer.polarity_scores(text)["compound"]
        # Aggregate entities in document
        ents = [e[0] for e in row.entities if e[1] in TARGET_ENTITY_TYPES]
        for ent in set(ents):
            rows.append({
                "entity": ent.lower(),
                "period": row.period,
                "sentiment": sent_val
            })

    ent_df = pd.DataFrame(rows)
    # Filter to frequent entities
    min_freq = cfg.get("entity_min_freq", 15)
    counts = ent_df["entity"].value_counts()
    freq_entities = counts[counts >= min_freq].index
    ent_df = ent_df[ent_df.entity.isin(freq_entities)]

    agg = ent_df.groupby(["entity","period"])["sentiment"].mean().unstack()
    agg["delta_post_minus_pre"] = agg.get("post", 0) - agg.get("pre", 0)
    agg.sort_values("delta_post_minus_pre", ascending=False, inplace=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    agg.to_csv("outputs/tables/entity_sentiment.csv")
    print("Entity sentiment table written to outputs/tables/entity_sentiment.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)