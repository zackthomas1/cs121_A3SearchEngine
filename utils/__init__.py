import os
import logging
from urllib.parse import urlparse

def clean_url(url: str) -> str:
    parsed_url = urlparse(url)
    if 'index' in parsed_url.path:
        return normalize_url(parsed_url._replace(path="", query="", fragment="").geturl())
    else:
        return normalize_url(parsed_url._replace(query="", fragment="").geturl())

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

