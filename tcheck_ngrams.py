import json

# Path to your master index file
MASTER_INDEX_FILE = "index/master_index/master_index.json"

with open(MASTER_INDEX_FILE, "r", encoding="utf-8") as f:
    master_index = json.load(f)

# Filter for keys that contain a space (indicating n-grams)
ngram_keys = [key for key in master_index if " " in key]

print("N-gram tokens in the master index:")
for key in ngram_keys:
    print(key)
