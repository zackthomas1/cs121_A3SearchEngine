import os
import re
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
from utils import clean_url, is_non_html_extension, get_logger


# Constants 
DOC_THRESHOLD = 250 # Dump index to latest JSON file every 100 docs
# NEW_FILE_THRESHOLD = 1000   # Create new index file every 1000 docs
PARTIAL_INDEX_DIR = "index/partial_index"
MASTER_INDEX_DIR = "index/master_index"
MASTER_INDEX_FILE = os.path.join(MASTER_INDEX_DIR, "master_index.json")

#
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

class InvertedIndex: 
    def __init__(self): 
        self.index = defaultdict(list)
        self.doc_count = 0
        # self.total_doc_count = 0
        # self.current_index_file = self.get_latest_index_file()
        self.doc_id_map = {} # map file names to docid
        self.logger = get_logger("INVERTED_INDEX")
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True) # Ensure index storage directory exist

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

    def build_master_index(self):
        """Combines all partial indexes into a single master index while preserving order."""
        self.logger.info(f"Building Master index...")

        master_index = defaultdict(list)

        # Iterate through all partial index files
        for file_name in sorted(os.listdir(PARTIAL_INDEX_DIR)):  # Ensure order is maintained
            self.logger.info(f"Adding: {file_name}")
            if file_name.startswith("index_part_") and file_name.endswith(".json"):
                file_path = os.path.join(PARTIAL_INDEX_DIR, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    partial_index = json.load(f)

                # Merge token postings while maintaining order
                for token, postings in partial_index.items():
                    master_index[token].extend(postings)

        # Save master index to disk
        with open(MASTER_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(master_index, f, indent=4)

        self.logger.info(f"Master index built successfully and saved to {MASTER_INDEX_FILE}")
    
    def get_unique_tokens(self):
        """Returns tokens that appear only once in a single document."""
        unique_tokens = set()

        # Load master index if it exists
        if os.path.exists(MASTER_INDEX_FILE):
            with open(MASTER_INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        else:
            index_data = {}

        # Iterate through all tokens
        for token, postings in index_data.items():
            # Check if the token appears in exactly one document and has a frequency of 1
            if len(postings) == 1 and postings[0][1] == 1:
                unique_tokens.add(token)

        return list(unique_tokens)

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
        if is_non_html_extension(url):
            self.logger.warning(f"Skipping url with non html extension")
            return
        # if not is_unique_url(url):
        #     self.logger.warning(f"Skipping non-unique Url: {os.path.join(root, file)} - {url}")
        #     return

        text = self.__extract_text_from_html_content(data['content'])
        if not text: 
            self.logger.warning(f"Skipping empty HTML text content: {file_path}")
            return

        # self.logger.info(f"Tokenizing document content")
        tokens = InvertedIndex.__stem_tokens(InvertedIndex.__tokenize_text(text))
        token_freq = InvertedIndex.__construct_token_freq_counter(tokens)

        # self.logger.info(f"Updating inverted index")
        for token, freq in token_freq.items(): 
            self.index[token].append((doc_id, freq))

        self.doc_count += 1
        # self.total_doc_count += 1

        # If threshould reached, store partial index and reset RAM
        if self.doc_count >= DOC_THRESHOLD: 
            self.__dump_to_disk()
            self.index.clear()
            self.doc_count = 0 
            gc.collect()

    def __dump_to_disk(self): 
        """Store current index to JSON file"""
        self.logger.info("Dumping index to disk")
        index_file = os.path.join(PARTIAL_INDEX_DIR, f"index_part_{len(os.listdir(PARTIAL_INDEX_DIR))}.json")

        # Load existing index
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f: 
                existing_data = json.load(f)
        else: 
            existing_data = {}

        for token, postings in self.index.items(): 
            if token in existing_data: 
                existing_data[token].extend(postings)
            else: 
                existing_data[token] = postings

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

    def __merge_from_disk(self, query_tokens):
        """Loads only relevant part of index from disk for a given query."""
        merged_index = {}
        for file_name in os.listdir(PARTIAL_INDEX_DIR): 
            file_path = os.path.join(PARTIAL_INDEX_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as f: 
                index_part = json.load(f)

            for token in query_tokens: 
                if token in index_part: 
                    if token in merged_index: 
                        merged_index[token].extend(index_part[token])
                    else: 
                        merged_index[token] = index_part[token]

        return merged_index

    def __construct_token_freq_counter(tokens) -> Counter:
        counter = Counter()
        counter.update(tokens)
        return counter

    def __lemmatize_tokens(tokens: list[str]) -> list[str]:
        return [lemmatizer.lemmatize(token) for token in tokens]

    def __stem_tokens(tokens: list[str]) -> list[str]: 
        return [stemmer.stem(token) for token in tokens if re.match(r"^[a-zA-Z]+$", token)]

    def __tokenize_text(text: str) -> list[str]:
        return word_tokenize(text)
        # return tokenizer.tokenize(text)

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
                # self.logger.info(f"Success: Load JSON file: {file_path}")
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