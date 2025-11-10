def load_nrc(path):
    """
    NRC file format: word<tab>emotion<tab>1/0
    Returns dict: word -> set(emotions)
    """
    emo_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if flag != '1':
                continue
            emo_map.setdefault(word.lower(), set()).add(emotion)
    return emo_map

HOPE_CUSTOM = {
    # Extend/adjust this curated hope lexicon
    "progress", "solution", "solutions", "opportunity", "opportunities",
    "accelerate", "advancing", "advance", "improve", "improvement",
    "transition", "commitment", "committed", "ambitious", "ambition"
}

def aggregate_emotions(tokens, nrc_map):
    from collections import Counter
    c = Counter()
    for t in tokens:
        emos = nrc_map.get(t)
        if not emos:
            continue
        for e in emos:
            c[e] += 1
    return c

def compute_hope_proxy(counts):
    # Approximate hope from anticipation + trust + curated positive lexicon occurrences
    hope = counts.get("anticipation", 0) + counts.get("trust", 0)
    return hope