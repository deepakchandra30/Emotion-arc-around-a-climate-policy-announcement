# Climate Policy Emotion Arc (CA1 Baselines)

## Project Goal
Measure how discrete emotions (anger, fear, trust, hope proxy) and related linguistic patterns in climate-policy news shift before vs after a policy announcement (default: IPCC AR6 Synthesis Report, March 20 2023).

## Structure
```
climate-emotion-arc/
  README.md
  Makefile
  requirements.txt
  configs/
    config.yaml
    keywords.txt
  data/
    raw/
    processed/
    samples/
  src/
    fetch_gdelt.py
    preprocess.py
    emotion_lexicons.py
    emotion_counts.py
    entity_sentiment.py
    ngram_shift.py
    lda_topics.py
    plot_emotions.py
    plot_topic_shift.py
    utils.py
```

## Setup
1. Python 3.10+ recommended.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
  And install the spaCy English model:
  ```
  python -m spacy download en_core_web_sm
  ```
3. Download NRC Emotion Lexicon:
   - Visit: [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
   - Place file `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` into `data/lexicons/` (create if missing).
   - Update `nrc_lexicon_path` in `configs/config.yaml` if needed.

## Configuration
Edit `configs/config.yaml`:
- `event_date`: anchor date (YYYY-MM-DD).
- `pre_days` / `post_days`: window sizes.
- `keywords_file`: list of query tokens.
- Adjust `min_doc_chars` to filter very short items.

## Workflow
Make targets (see Makefile for details):
```
make fetch        # Pull GDELT articles for pre/post windows
make preprocess   # Clean & tokenize
make emotion      # Compute NRC + VADER emotion aggregates
make entities     # Entity-centric sentiment
make ngram_shift  # Top shifting n-grams pre vs post
make lda          # LDA topics and pre/post proportions
make collocations # PMI bigram collocations (pre/post)
make bow          # TF-IDF + Logistic Regression / Linear SVM baselines
make figures      # Generate emotion arc & topic shift plots
```

Re-run everything (idempotent):
```
make all
```

## Outputs
- Emotion time series: `outputs/figures/emotion_arc.png`
- Pre/Post emotion bar chart: `outputs/figures/emotion_bar.png`
- Entity sentiment delta table: `outputs/tables/entity_sentiment.csv`
- N-gram shift tables: `outputs/tables/ngram_shift_positive.csv`, `outputs/tables/ngram_shift_negative.csv`
- Topic proportion heatmap: `outputs/figures/topic_shift_heatmap.png`
- Collocations (PMI): `outputs/tables/collocations_pre.csv`, `outputs/tables/collocations_post.csv`
- BoW baselines metrics: `outputs/tables/bow_baselines_metrics.json`
- Top TF-IDF features: `outputs/tables/tfidf_top_features.csv`
- LSA 2D scatter: `outputs/figures/lsa_scatter.png`

## Hope Proxy
"Hope" is approximated using NRC categories: Anticipation + Trust + Joy subset. See `emotion_counts.py` for mapping; you may refine with a curated lexicon in `configs/keywords.txt` (additional hope terms) or a separate file later.

## Data Ethics
- GDELT returns metadata + snippets; fetching full article text must respect robots.txt.
- Store only necessary textual content; do not redistribute proprietary full texts.
- Lexicon-based emotion detection is approximate; document limitations.

## Extension (Later Phase)
Replace lexicons with contextual emotion classifiers, add stance detection, employ BERTopic for temporal subthemes, and embed semantic shift of key terms.

## Sample Run (Small Subset)
Place a few manually saved raw JSON lines into `data/raw/sample_pre.jsonl` and `data/raw/sample_post.jsonl` (or run fetch with reduced day windows) then:
```
make preprocess emotion figures
```

## Citation
If using IPCC event:
- Cite IPCC AR6 Synthesis Report.
- Cite GDELT: Leetaru, K. (GDELT Project).
- Cite NRC lexicon paper (Mohammad & Turney).

## License
This repository (code) under MIT. External data subject to original licenses.
