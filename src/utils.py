import yaml
from pathlib import Path
import json
from datetime import datetime, timedelta

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_keywords(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def daterange(start_date, end_date, step_days=1):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=step_days)

def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def period_label(date, event_date):
    return "pre" if date < event_date else "post"