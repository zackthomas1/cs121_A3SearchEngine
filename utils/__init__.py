import re
import os
import math
import json
import logging
from logging import Logger
from urllib.parse import urlparse
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

#
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def compute_tf_idf(tf: int, doc_freq: int, total_docs: int) -> int:
    """
    Computes the tf-idf score. 
    TF(Token Frequency): term_freq / doc_length 
    IDF (Inverse Document Frequency): math.log(total_docs / (1+doc_freq))

    Parameters:
        tf (int): The token frequency score calculated during document processing
        doc_freq (int): Number of documents that contain the token
        total_docs (int): Total number of documents in corpus

    Returns:
        int: The tf-idf score
    """

    idf = math.log(total_docs / (1+doc_freq))

    return tf * idf

def clean_url(url: str) -> str:
    """
    Given a url string, returns the normalized url.
    normalize: to standardize or make consistent (remove trailing slashes / index.html default pages)
    """
    parsed_url = urlparse(url)
    if 'index' in parsed_url.path:
        return normalize_url(parsed_url._replace(path="", query="", fragment="").geturl())
    else:
        return normalize_url(parsed_url._replace(query="", fragment="").geturl())

def normalize_url(url: str) -> str:
    """ strips trailing backslash"""
    if url.endswith("/"):
        return url.rstrip("/")
    return url

def get_logger(name, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not os.path.exists("logs"):
        os.makedirs("logs")
    fh = logging.FileHandler(f"logs/{filename if filename else name}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
       "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def read_json_file(file_path: str, logger: Logger) -> dict:
    """
    Parameters:
        file_path (str): File path to json document in local file storage
        logger (Logger):
    Returns:
        dict: returns the data stored in the json file as a python dictionary
    """
    try:
        with open(file_path, 'r') as file: 
            data = json.load(file)
            # logger.info(f"Success: Data read from json file: {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"File note found at path: {file_path}")
        return None 
    except json.JSONDecodeError: 
        logger.error(f"Invalid JSON format in file:  {file_path}")
        return None 
    except Exception as e:
        logger.error(f"An unexpected error has orccurred: {file_path} - {e}") 
        return None

def write_json_file(file_path: str, data: dict, logger: Logger) -> None:
    """
    Parameters:
        file_path (str): File path to json document in local file storage
        logger (Logger):
    """
    try:
        new_data = {}
        # Check if there is already data save in the file
        if os.path.exists(file_path):
            logger.warning(f"Existing data in json file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f: 
                new_data = json.load(f)

        # Combine existing and new data
        for key, value in data.items(): 
            new_data[key] = value

        # write data to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)
            logger.info(f"Successful data write to json file: {file_path}")
    except Exception as e:
        logger.error(f"Unable to write data to json file: {file_path} - {e}")

def remove_stop_words(tokens: list[str]) -> list[str]: 
    """
    Removes stop words from list of tokens
    Returns:
        list[str]: list of tokens with stop words removed
    """
    return [ token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Apply nltk lemmatization algorithm to extracted tokens
    
    Parameters:
        tokens (list[str]): a list of raw tokens 

    Returns:
        list[str]: a lemmatized list of tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Apply porters stemmer to tokens
    
    Parameters:
        tokens (list[str]): a list of raw tokens 

    Returns:
        list[str]: a lemmatized list of tokens
    """

    return [stemmer.stem(token) for token in tokens]

def tokenize_text(text: str) -> list[str]:
    """
    Use nltk to tokenize text. Remove stop words and non alphanum
    
    Parameters:
        text (str): Text content parsed from an html document

    Returns:
        list[str]: a list of tokens extracted from the text content string
    """

    tokens =  word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum()]

def is_non_html_extension(url: str) -> bool: 
    parsed_url = urlparse(url)

    return re.match(
    r".*\.(css|js|bmp|gif|jpe?g|ico"
    + r"|png|tiff?|mid|mp2|mp3|mp4"
    + r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
    + r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
    + r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
    + r"|epub|dll|cnf|tgz|sha1"
    + r"|thmx|mso|arff|rtf|jar|csv"
    + r"|rm|smil|wmv|swf|wma|zip|rar|gz"
    + r"|img|java|war|sql|mpg|ff|sh|ppsx|py|apk|svg|conf|cpp|fig|cls|ipynb|bam|odp|odc|tsv|nb|bib|z|rpm|ma)$", parsed_url.path.lower())


def is_xml(content: str) -> bool: 
    """
    """

    stripped_content = content.lstrip().lower() 
    return  stripped_content.startswith("<?xml") or stripped_content.startswith("<xml")
