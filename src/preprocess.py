import argparse, re
from pathlib import Path
from collections import Counter
from datetime import datetime
import spacy
from tqdm import tqdm
import pandas as pd
from utils import load_config, read_jsonl

URL_PATTERN = re.compile(r'https?://\S+')

def clean_text(text):
    if not text:
        return ""
    text = re.sub(URL_PATTERN, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main(cfg_path):
    cfg = load_config(cfg_path)
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    raw_path = Path(cfg["raw_dir"]) / "gdelt_raw.jsonl"
    out_dir = Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in read_jsonl(raw_path):
        text = rec.get("title", "") + " " + rec.get("seendate", "")
        snippet = rec.get("extras", {}).get("articletext", "") or rec.get("title", "")
        combined = rec.get("title", "") + " " + snippet
        cleaned = clean_text(combined)
        if len(cleaned) < cfg["min_doc_chars"]:
            continue
        doc = nlp(cleaned)
        tokens = [t.lemma_.lower() for t in doc if t.is_alpha]
        ents = [(e.text, e.label_) for e in doc.ents]
        seendate = rec.get("seendate", "")
        try:
            dt = datetime.strptime(seendate, "%Y-%m-%d %H:%M:%S")
        except:
            dt = datetime.utcnow()
        rows.append({
            "id": rec.get("url", ""),
            "date": dt,
            "period": rec.get("period", ""),
            "domain": rec.get("domain", ""),
            "text": cleaned,
            "tokens": tokens,
            "entities": ents
        })

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.to_pickle(out_dir / "processed.pkl")
    print(f"Processed documents: {len(df)} -> {out_dir/'processed.pkl'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)