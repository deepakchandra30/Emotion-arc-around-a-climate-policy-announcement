PYTHON=python

all: fetch preprocess emotion entities ngram_shift lda collocations bow figures

fetch:
	$(PYTHON) src/fetch_gdelt.py --config configs/config.yaml

preprocess:
	$(PYTHON) src/preprocess.py --config configs/config.yaml

emotion:
	$(PYTHON) src/emotion_counts.py --config configs/config.yaml

entities:
	$(PYTHON) src/entity_sentiment.py --config configs/config.yaml

ngram_shift:
	$(PYTHON) src/ngram_shift.py --config configs/config.yaml

lda:
	$(PYTHON) src/lda_topics.py --config configs/config.yaml

collocations:
	$(PYTHON) src/collocations.py --config configs/config.yaml

bow:
	$(PYTHON) src/bow_baselines.py --config configs/config.yaml


figures:
	$(PYTHON) src/plot_emotions.py --config configs/config.yaml
	$(PYTHON) src/plot_topic_shift.py --config configs/config.yaml

clean:
	rm -rf outputs
	mkdir -p outputs/figures outputs/tables

.PHONY: all fetch preprocess emotion entities ngram_shift lda collocations bow figures clean
