import json

# Path to your master index file
MASTER_INDEX_FILE = "index/master_index/master_index.json"

def check_index_for_term(term):
    """Check if a term exists in the index."""
    try:
        with open(MASTER_INDEX_FILE, "r", encoding="utf-8") as f:
            index = json.load(f)

        if term in index:
            print(f"✅ '{term}' found in index!")
        else:
            print(f"❌ '{term}' NOT found in index.")

    except FileNotFoundError:
        print(f"Error: {MASTER_INDEX_FILE} not found. Did you build the index?")
    except json.JSONDecodeError:
        print(f"Error: Could not parse {MASTER_INDEX_FILE}. Check if the file is valid JSON.")

# Run checks for both stemmed and unstemmed words
check_index_for_term("machine")     # Check for raw word
check_index_for_term("machin")      # Check for stemmed version
check_index_for_term("automobile")  # Check for raw word
check_index_for_term("automobil")   # Check for stemmed version


