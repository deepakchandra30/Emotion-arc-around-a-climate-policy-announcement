import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_config
from pathlib import Path

def main(cfg_path):
    cfg = load_config(cfg_path)
    dist_path = "outputs/tables/topic_period_distribution.csv"
    if not Path(dist_path).exists():
        print("Topic distribution file not found. Run lda step first.")
        return
    df = pd.read_csv(dist_path, index_col=0)
    # Transpose for heatmap (topics as rows)
    dfT = df.T
    sns.set_theme(style="white")
    plt.figure(figsize=(10, max(6, 0.4 * dfT.shape[0])))
    sns.heatmap(dfT, annot=False, cmap="viridis")
    plt.title("Topic Proportion Shift (Pre vs Post)")
    plt.tight_layout()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("outputs/figures/topic_shift_heatmap.png", dpi=150)
    plt.close()
    print("Topic shift heatmap saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)