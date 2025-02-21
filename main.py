import os
import json
import logging
import shelve
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from argparse import ArgumentParser
from bs4 import BeautifulSoup

# Download NLTK resources
nltk.download('punkt_tab')

"""
These are a few possible functions I thought of just off the top of my head.
They are not requirements for the project.
Please replace/delete them if you can think of a better architecture or high level functions.
"""

def stemm_tokens(tokens: list[str]) -> list[str]: 
    pass

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
    
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            data = read_json_file(os.path.join(root, file)) 
            
            if not data: 
                continue

            text = extract_text_from_html_content(data['content'])

            if not text: 
                continue
            
            tokens = tokenize_text(text)



"""
Entry point
Call 'python main.py' from the command line to run program
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rootdir", type=str, default="developer\DEV")
    args = parser.parse_args()

    walk_directory(args.rootdir)