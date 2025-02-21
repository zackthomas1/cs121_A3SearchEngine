import os
import json
import logging
import shelve
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urlparse

# Download NLTK resources
nltk.download('punkt_tab')

#TODO: These data structures should not be stored in memory. 
# They need to be stored in a file. The inverse index should be store across multiple files
docid_url_index : dict[int, str] = {}
inverse_index : dict[int, list[tuple[str, int]]]  = {}

def normalize_url(url):
    if url.endswith("/"):
        return url.rstrip("/")
    return url

def update_docid_index(docid : int, url: str) -> None: 
    docid_url_index[docid] = url

def update_inverse_index(docid: int, token_count: Counter) -> None: 
    for token, count in token_count.items(): 
        if token in inverse_index:
            inverse_index[token].append((docid, count))
        else: 
            inverse_index[token] = [(docid, count)]

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
        print(f"An unexpected error has orccurred: {e}") 
        return None 

def read_json_file(filepath: str) -> dict[str, str]:
    """
    """

    try: 
        with open(filepath, 'r') as file: 
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File note found at path: {filepath}")
        return None 
    except json.JSONDecodeError: 
        print(f"Invalid JSON format in file:  {filepath}")
        return None 
    except Exception as e:
        print(f"An unexpected error has orccurred: {e}") 
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

            url = data['url']
            parsed_url = urlparse(url)
            if 'index' in parsed_url.path:
                clean_url = normalize_url(parsed_url._replace(path="", query="", fragment="").geturl())
            else:
                clean_url = normalize_url(parsed_url._replace(query="", fragment="").geturl())

            if clean_url in set(docid_url_index.values()):
                continue

            text = extract_text_from_html_content(data['content'])

            if not text: 
                continue
            
            tokens = stem_tokens(tokenize_text(text))
            token_count = construct_token_freq_counter(tokens)
            
            update_inverse_index(docid, token_count)
            update_docid_index(docid, clean_url)

            docid += 1
"""
Entry point
Call 'python main.py' from the command line to run program
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="developer\DEV")
    args = parser.parse_args()

    walk_directory(args.rootdir)