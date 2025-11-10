import requests, argparse
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from utils import load_config, load_keywords, daterange, write_jsonl

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def build_query_string(keywords):
    # Simple OR across keywords plus ensure climate context.
    parts = []
    for kw in keywords:
        kw_enc = kw.replace(" ", "+")
        parts.append(kw_enc)
    # Use OR logic; GDELT uses space as implicit AND; we combine with OR for breadth.
    # Simplicity: join with OR operator.
    return "+OR+".join(parts)

def fetch_chunk(query, start_dt, end_dt, max_records):
    params = {
        "query": query,
        "mode": "ArtList",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords": max_records,
        "format": "json"
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j.get("articles", [])

def main(config_path):
    cfg = load_config(config_path)
    keywords = load_keywords(cfg["keywords_file"])
    event_date = datetime.strptime(cfg["event_date"], "%Y-%m-%d")
    pre_start = event_date - timedelta(days=cfg["pre_days"])
    post_end = event_date + timedelta(days=cfg["post_days"])

    raw_dir = Path(cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    query = build_query_string(keywords)
    all_start = pre_start
    all_end = post_end

    print(f"Fetching from {all_start.date()} to {all_end.date()} for query: {query}")

    # Day chunks
    for day in tqdm(list(daterange(all_start, all_end, cfg["chunk_days"]))):
        start_dt = datetime(day.year, day.month, day.day, 0, 0, 0)
        end_dt = start_dt + timedelta(days=cfg["chunk_days"]) - timedelta(seconds=1)
        articles = fetch_chunk(query, start_dt, end_dt, cfg["max_records_per_call"])
        # Append period flag
        for a in articles:
            try:
                art_date = datetime.strptime(a["seendate"], "%Y-%m-%d %H:%M:%S")
            except:
                art_date = start_dt
            a["period"] = "pre" if art_date < event_date else "post"
        if articles:
            write_jsonl(raw_dir / "gdelt_raw.jsonl", articles)

    print("Fetch complete. Raw data stored at data/raw/gdelt_raw.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)