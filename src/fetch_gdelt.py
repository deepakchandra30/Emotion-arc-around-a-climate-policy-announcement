import requests, argparse, time, sys
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from utils import load_config, load_keywords, daterange, write_jsonl

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def build_query_string(keywords, language=None, require_term=None):
    # Original simple OR style for GDELT doc API: join keywords with +OR+
    enc = []
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        enc.append(kw.replace(" ", "+"))
    return "+OR+".join(enc)

def fetch_chunk(query, start_dt, end_dt, max_records, max_retries=5, verbose=False):
    params = {
        "query": query,
        "mode": "ArtList",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords": max_records,
        "format": "json"
    }
    backoff = 2
    for attempt in range(max_retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=40)
            if r.status_code == 429:
                time.sleep(min(30, backoff ** attempt))
                continue
            r.raise_for_status()
            data = r.json().get("articles", [])
            return data
        except requests.RequestException:
            time.sleep(min(30, backoff ** attempt))
            last_err = r.status_code if 'r' in locals() else 'request error'
            continue
    Path("data/raw/fetch_errors").mkdir(parents=True, exist_ok=True)
    with open(Path("data/raw/fetch_errors")/f"{start_dt.strftime('%Y%m%d')}.err.txt", "a", encoding="utf-8") as f:
        f.write(f"Failed window {start_dt} - {end_dt}: {last_err}\n")
    return []

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
    days = list(daterange(all_start, all_end, cfg["chunk_days"]))
    for day in tqdm(days, desc="IPCC Days"):
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
        else:
            # Log empty day for diagnostics
            Path("data/raw/fetch_errors").mkdir(parents=True, exist_ok=True)
            with open(Path("data/raw/fetch_errors")/"empty_days.log", "a", encoding="utf-8") as f:
                f.write(f"No articles for {start_dt.date()}\n")
        time.sleep(1.5)  # polite spacing

    print("\nFetch complete. Raw data stored at data/raw/gdelt_raw.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)