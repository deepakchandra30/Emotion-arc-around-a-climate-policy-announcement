import argparse
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emotion_lexicons import load_nrc, aggregate_emotions, compute_hope_proxy, HOPE_CUSTOM
from utils import load_config
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(cfg_path):
    cfg = load_config(cfg_path)
    df = pd.read_pickle(Path(cfg["processed_dir"]) / "processed.pkl")
    nrc = load_nrc(cfg["nrc_lexicon_path"])
    analyzer = SentimentIntensityAnalyzer()

    # Compute per-document emotion counts + VADER
    emo_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        counts = aggregate_emotions(row.tokens, nrc)
        hope_extra = sum(1 for t in row.tokens if t in HOPE_CUSTOM)
        hope_proxy = compute_hope_proxy(counts) + hope_extra
        vader = analyzer.polarity_scores(row.text)
        emo_rows.append({
            "date": row.date,
            "period": row.period,
            "anger": counts.get("anger", 0),
            "fear": counts.get("fear", 0),
            "trust": counts.get("trust", 0),
            "anticipation": counts.get("anticipation", 0),
            "joy": counts.get("joy", 0),
            "sadness": counts.get("sadness", 0),
            "disgust": counts.get("disgust", 0),
            "surprise": counts.get("surprise", 0),
            "hope_proxy": hope_proxy,
            "vader_compound": vader["compound"],
            "token_count": len(row.tokens)
        })

    emo_df = pd.DataFrame(emo_rows)
    emo_df.to_csv("outputs/tables/emotion_doc_level.csv", index=False)

    # Aggregate by period
    agg_cols = ["anger","fear","trust","hope_proxy"]
    period_agg = emo_df.groupby("period")[agg_cols + ["token_count"]].sum()
    for col in agg_cols:
        period_agg[col+"_per_1k_tokens"] = 1000 * period_agg[col] / period_agg["token_count"]

    period_agg.to_csv("outputs/tables/emotion_period.csv")

    # Rolling time series (overall)
    emo_df.sort_values("date", inplace=True)
    for col in ["anger","fear","trust","hope_proxy"]:
        emo_df[col+"_per_1k"] = 1000 * emo_df[col] / emo_df["token_count"].replace(0,1)

    # Rolling average
    window = cfg["plots"]["emotion_rolling_window"]
    rolling = emo_df.set_index("date")[["anger_per_1k","fear_per_1k","trust_per_1k","hope_proxy_per_1k"]].rolling(f"{window}D").mean()
    rolling.reset_index(inplace=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10,6))
    for col in ["anger_per_1k","fear_per_1k","trust_per_1k","hope_proxy_per_1k"]:
        plt.plot(rolling["date"], rolling[col], label=col.replace("_per_1k",""))
    plt.legend()
    plt.title("Emotion Arc (Rolling {}-Day Mean)".format(window))
    plt.xlabel("Date"); plt.ylabel("Occurrences per 1k tokens")
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/figures/emotion_arc.png", dpi=150)
    plt.close()

    # Bar chart pre vs post
    bar_df = period_agg[[c+"_per_1k_tokens" for c in agg_cols]].T
    bar_df.columns = bar_df.columns.str.upper()
    bar_df.plot(kind="bar", figsize=(8,5))
    plt.ylabel("Occurrences per 1k tokens")
    plt.title("Pre vs Post Emotion Intensities")
    plt.tight_layout()
    plt.savefig("outputs/figures/emotion_bar.png", dpi=150)
    plt.close()

    print("Emotion computations complete.")

if __name__ == "__main__":
    import os
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)