import nltk

"""
----------------IMPORTANT-----------------
Run this script before building the inverted index or searching
Downloads required nltk libraries
"""
if __name__ == "__main__":
    # Download NLTK resources
    nltk.download('punkt_tab')
    nltk.download('wordnet')