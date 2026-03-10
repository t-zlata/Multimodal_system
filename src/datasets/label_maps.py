TOPICS = [
    "arts and media",
    "conflict and war",
    "military and veterans",
    "volunteering and charity",
    "economy and finance",
    "business",
    "education",
    "environment and weather",
    "health",
    "human interest",
    "humor",
    "internet culture",
    "labour and employment",
    "lifestyle and leisure",
    "news",
    "politics and government",
    "religion",
    "science and technology",
    "society (community issues)",
    "pets and animals",
]
TOPIC2ID = {t: i for i, t in enumerate(TOPICS)}

SENTIMENTS = ["negative", "neutral", "positive"]
SENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}

CONTEXTS = ["informational", "opinion", "emotional", "meme"]
CTX2ID = {c: i for i, c in enumerate(CONTEXTS)}