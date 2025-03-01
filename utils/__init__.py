import re
import os
import logging
from urllib.parse import urlparse

def clean_url(url: str) -> str:
    parsed_url = urlparse(url)
    if 'index' in parsed_url.path:
        return normalize_url(parsed_url._replace(path="", query="", fragment="").geturl())
    else:
        return normalize_url(parsed_url._replace(query="", fragment="").geturl())

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

def normalize_url(url):
    """ strips trailing backslash"""
    if url.endswith("/"):
        return url.rstrip("/")
    return url

