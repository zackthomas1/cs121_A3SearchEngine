import os
import gc
import json
import nltk
import shelve
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import clean_url, get_logger


# Constants 
DOC_THRESHOLD = 100
SHELVE_DB = "shelve/inverted_index"

#
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

class InvertedIndex: 
    def __init__(self): 
        self.index = defaultdict(list)
        self.doc_count = 0
        self.doc_id_map = {} # map file names to docid
        self.logger = get_logger("INVERTED_INDEX")

    def build_index(self, folder_path): 
        """
        Process all JSON files in folder and build index.
        """
        doc_id = 0
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                self.logger.info(f"Indexing doc: {doc_id}")
                if file_name.endswith(".json"):
                    self.__process_document(os.path.join(root, file_name), doc_id)
                doc_id += 1
                
        # Dump any remaining tokens to disk
        if self.index: 
            self.__dump_to_disk()
            self.index.clear()
            gc.collect()

    def search(self, query): 
        self.logger.info(f"Searching for query tokens in inverted index: {query}")
        tokens = InvertedIndex.__stem_tokens(InvertedIndex.__tokenize_text(query))

        return self.__merge_from_disk(tokens)

    def __process_document(self, file_path, doc_id):
        data = self.__read_json_file(file_path)
        if not data:
            self.logger.warning(f"Skipping empty JSON file: {file_path}")
            return
        
        url = clean_url(data['url'])
        # if not is_unique_url(url):
        #     self.logger.warning(f"Skipping non-unique Url: {os.path.join(root, file)} - {url}")
        #     return

        text = self.__extract_text_from_html_content(data['content'])
        if not text: 
            self.logger.warning(f"Skipping empty HTML text content: {file_path}")
            return

        self.logger.info(f"Tokenizing document content")
        tokens = InvertedIndex.__stem_tokens(InvertedIndex.__tokenize_text(text))
        token_freq = InvertedIndex.__construct_token_freq_counter(tokens)

        self.logger.info(f"Updating inverted index")
        for token, freq in token_freq.items(): 
            self.index[token].append((doc_id, freq))

        self.doc_count += 1

        # If threshould reached, store partial index and reset RAM
        if self.doc_count >= DOC_THRESHOLD: 
            self.__dump_to_disk()
            self.index.clear()
            self.doc_count = 0 
            gc.collect()

    def __dump_to_disk(self): 
        """Store current index to disk using shelve"""
        self.logger.info("Dumping in memory index to disk")
        with shelve.open(SHELVE_DB) as db:
            for token, postings in self.index.items(): 
                if token in db: 
                    db[token].extend(postings)
                else:
                    db[token] = postings

    def __merge_from_disk(self, query_tokens):
        """Loads only relevant part of index from disk for a given query."""
        merged_index = {}
        with shelve.open(SHELVE_DB) as db:
            for token in query_tokens:
                if token in db: 
                    merged_index[token] = db[token]
        return merged_index

    def __construct_token_freq_counter(tokens) -> Counter:
        counter = Counter()
        counter.update(tokens)
        return counter

    def __lemmatize_tokens(tokens: list[str]) -> list[str]:
        return [lemmatizer.lemmatize(token) for token in tokens]

    def __stem_tokens(tokens: list[str]) -> list[str]: 
        return [stemmer.stem(token) for token in tokens if token.isalnum()]

    def __tokenize_text(text: str) -> list[str]:
        # return word_tokenize(text)
        return tokenizer.tokenize(text)

    def __extract_text_from_html_content(self, content: str) -> list[str]: 
        """
        """
        try:
            #TODO: Check that the content is html before parsing. Document content may also be xml

            # Get the text from the html response
            soup = BeautifulSoup(content, 'html.parser')

            # Remove the text of CSS, JS, metadata, alter for JS, embeded websites
            for markup in soup.find_all(["style", "script", "meta", "noscript", "iframe"]):  
                markup.decompose()  # remove all markups stated above
            
            # soup contains only human-readable texts now to be compared near-duplicate
            text = soup.get_text(separator=" ", strip=True)
            return text
        except Exception as e:
            self.logger.error(f"An unexpected error has orccurred: {e}") 
            return None 
    
    def __read_json_file(self, file_path: str) -> dict[str, str]:
        """
        """
        try: 
            with open(file_path, 'r') as file: 
                data = json.load(file)
                self.logger.info(f"Success: Load JSON file: {file_path}")
                return data
        except FileNotFoundError:
            self.logger.error(f"File note found at path: {file_path}")
            return None 
        except json.JSONDecodeError: 
            self.logger.error(f"Invalid JSON format in file:  {file_path}")
            return None 
        except Exception as e:
            self.logger.error(f"An unexpected error has orccurred: {e}") 
            return None 