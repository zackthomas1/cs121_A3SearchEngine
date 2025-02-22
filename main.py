import os
import json
import shelve
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from collections import Counter
from utils import clean_url, get_logger

# Download NLTK resources
nltk.download('punkt_tab')

# create logger
logger = get_logger("MAIN")

# specifiy global path to shelve files
inverse_index_path = 'shelve/inverse_index.shelve'
docid_index_path = 'shelve/docid_index.shelve'

# Shelves are preferable when: 
#     Dataset is large and does not fit in memory
#     Random access to specific terms in in index required 
#     Incremental update indexd with reloading entire data structure

def clear_index(restart: bool) -> None: 
    if restart: 
        with shelve.open(inverse_index_path) as db: 
            db.clear()

        with shelve.open(docid_index_path) as db: 
            db.clear()

def is_unique_url(url: str) -> bool: 
    with shelve.open(docid_index_path) as db: 
        return not url in set(db.values())

def update_docid_index(docid : int, url: str) -> None: 
    with shelve.open(docid_index_path, writeback=True) as db:
        db[str(docid)] = url

def update_inverse_index(docid: int, token_count: Counter) -> None: 
    """
    """
    with shelve.open(inverse_index_path, writeback=True) as db: 
        for token, count in token_count.items(): 
            if token in db.keys():
                db[token].append((docid, count))
            else: 
                db[token] = [(docid, count)]

def construct_token_freq_counter(tokens) -> Counter:
    counter = Counter()
    counter.update(tokens)
    return counter

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer() 
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem_tokens(tokens: list[str]) -> list[str]: 
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def tokenize_text(text: str) -> list[str]:
    return word_tokenize(text)

def extract_text_from_html_content(content: str) -> list[str]: 
    """
    """
    try:
        # Get the text from the html response
        soup = BeautifulSoup(content, 'html.parser')

        # Remove the text of CSS, JS, metadata, alter for JS, embeded websites
        for markup in soup.find_all(["style", "script", "meta", "noscript", "iframe"]):  
            markup.decompose()  # remove all markups stated above
        
        # soup contains only human-readable texts now to be compared near-duplicate
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        logger.error(f"An unexpected error has orccurred: {e}") 
        return None 

def read_json_file(filepath: str) -> dict[str, str]:
    """
    """
    try: 
        with open(filepath, 'r') as file: 
            data = json.load(file)
            return data
    except FileNotFoundError:
        logger.error(f"File note found at path: {filepath}")
        return None 
    except json.JSONDecodeError: 
        logger.error(f"Invalid JSON format in file:  {filepath}")
        return None 
    except Exception as e:
        logger.error(f"An unexpected error has orccurred: {e}") 
        return None 

def walk_directory(rootdir: str) -> None:
    """
    """
    docid = 0
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            data = read_json_file(os.path.join(root, file)) 
            if not data:
                continue
            
            url = clean_url(data['url'])
            if not is_unique_url(url):
                continue

            text = extract_text_from_html_content(data['content'])
            if not text: 
                continue
            
            tokens = stem_tokens(tokenize_text(text))
            token_count = construct_token_freq_counter(tokens)
            
            update_inverse_index(docid, token_count)
            update_docid_index(docid, url)

            docid += 1
"""
Entry point
Call 'python main.py' from the command line to run program
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="developer\DEV")
    parser.add_argument("--restart", action="store_true", default=False)
    args = parser.parse_args()

    clear_index(args.restart)
    walk_directory(args.rootdir)