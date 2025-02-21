from urllib.parse import urlparse

def clean_url(url: str) -> str:
    parsed_url = urlparse(url)
    if 'index' in parsed_url.path:
        return normalize_url(parsed_url._replace(path="", query="", fragment="").geturl())
    else:
        return normalize_url(parsed_url._replace(query="", fragment="").geturl())

def normalize_url(url):
    """ strips trailing backslash"""
    if url.endswith("/"):
        return url.rstrip("/")
    return url