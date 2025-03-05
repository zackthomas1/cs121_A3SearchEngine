import os
import gc
import json
import math
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import clean_url, is_non_html_extension, get_logger, tokenize_text, stem_tokens
from typing import Dict, List, Tuple
from math import log as log_e

# Constants 
PARTIAL_INDEX_DOC_THRESHOLD = 250 # Dump index to latest JSON file every 100 docs
DOC_ID_DIR = "index/doc_id_map"            # "index/doc_id_map"
PARTIAL_INDEX_DIR = "index/partial_index"  # "index/partial_index"
MASTER_INDEX_DIR = "index/master_index"    # "index/master_index"
MASTER_INDEX_FILE = os.path.join(MASTER_INDEX_DIR, "master_index.json")
DOC_ID_MAP_FILE = os.path.join(DOC_ID_DIR, "doc_id_map.json")


class InvertedIndex: 
    def __init__(self):
        """ 
        Prepares to Index data by initializing storage directories and counter/keying variables.
        """
        
        self.index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # {token: [(docid, freq, tf)]}
        self.doc_id_map = {} # {doc_id: url}
        self.doc_count_partial_index = 0
        self.logger = get_logger("INVERTED_INDEX")

        # Initializes directories for index storage
        os.makedirs(DOC_ID_DIR, exist_ok=True) 
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        os.makedirs(MASTER_INDEX_DIR, exist_ok=True)

    def build_index(self, folder_path: str) -> None: 
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
            self.__save_index_to_disk()
            self.index.clear()
            gc.collect()

    def build_master_index(self) -> None:
        """
        Combines all partial indexes into a single master index while preserving order.
        """

        master_index = defaultdict(list)

        self.logger.info(f"Building Master index...")

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
    
    def construct_merged_index_from_disk(self, query_tokens: list[str]) -> dict[str, list[tuple[int, int]]]:
        """
        Constructs inverted index containing only query tokens from partial inverted index stored on disk
        
        Parameters:
        query_tokens (list[str]): 

        Returns:
        dict[str, list[tuple[int, int]]]: inverted index which contains only the query tokens entries from the partial index
        """
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
    
    def __process_document(self, file_path: str, doc_id: int):
        """
        Takes a file path to a document which stores an html page and updates the inverted index with tokens extracted from text content.
        Reads the file from disk. Extracts the html content. Updates the doc_id-url map. Tokenize the textual content.
        Update the inverted index with

        Parameters:
        file_path (str): The absolute file path to the document in the local file storage system
        doc_id (int): The unique id for the document at the provided file location 
        """

        # Read json file from disk
        data = self.__read_json_file(file_path)
        if not data:
            self.logger.warning(f"Skipping empty JSON file: {file_path}")
            return
        
        # Extract url and check that is valid
        url = clean_url(data['url'])
        if is_non_html_extension(url):
            self.logger.warning(f"Skipping url with non html extension")
            return

        # Extract textual content from html content
        text = self.__extract_text_from_html_content(data['content'])
        if not text: 
            self.logger.warning(f"Skipping empty HTML text content: {file_path}")
            return

        #
        self.__update_doc_id_map(doc_id, url)

        # Tokenize text
        # self.logger.info(f"Tokenizing document content")
        tokens = stem_tokens(tokenize_text(text))
        token_freq = InvertedIndex.__construct_token_freq_counter(tokens)

        # Update the inverted index with document tokens
        # self.logger.info(f"Updating inverted index")
        for token, freq in token_freq.items():
            tf = InvertedIndex.__compute_tf(freq, len(tokens))
            self.index[token].append((doc_id, freq, tf))

        # Update counters
        self.doc_count_partial_index += 1       # Used for Partial Indexing

        # Partial Indexing: If threshold is reached, store partial index and reset RAM
        if self.doc_count_partial_index >= PARTIAL_INDEX_DOC_THRESHOLD: 
            self.__dump_to_disk()

    def __update_doc_id_map(self, doc_id: int, url: str) -> None:
        """
        Updates the document id-url index with the provided doc_id url pair.
        Document id-url index records which url is associated with each doc_id number

        Parameters:
        doc_id (int): the unique identifier of the document
        url (str): url web address of the related document

        Returns:
        dict[str, list[tuple[int, int]]]
        """
        self.doc_id_map[doc_id] = url

    def __save_doc_id_map_to_disk(self) -> None: 
        """
        Saves the doc_id-url mapping to disk as a JSON file
        """

        if os.path.exists(DOC_ID_MAP_FILE):
            with open(DOC_ID_MAP_FILE, "r", encoding="utf-8") as f: 
                existing_map = json.load(f)
        else: 
            existing_map = {}

        for key, value in self.doc_id_map.items(): 
            existing_map[key] = value
        
        # write index to file
        with open(DOC_ID_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_map, f, indent=4)

    def __save_index_to_disk(self) -> None: 
        """
        Store current index to JSON file
        """

        self.logger.info("Dumping index to disk")
        
        # Create a new .json partial index file
        index_file = os.path.join(PARTIAL_INDEX_DIR, f"index_part_{len(os.listdir(PARTIAL_INDEX_DIR))}.json")

        # check if .json partial index file already existing index
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f: 
                existing_data = json.load(f)
        else: 
            existing_data = {}

        # Merge exisiting index with new data from in memory index
        for token, postings in self.index.items(): 
            if token in existing_data: 
                existing_data[token].extend(postings)
            else: 
                existing_data[token] = postings

        # write index to file
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

    def __dump_to_disk(self): 
        """
        Saves in memory partial inverted index and doc_id map to
        disk, then clears memory.
        """

        self.__save_index_to_disk()
        self.__save_doc_id_map_to_disk()
        self.index.clear()
        self.doc_id_map.clear()
        self.doc_count_partial_index = 0 
        gc.collect()

    def __construct_token_freq_counter(tokens: list[str]) -> Counter:  # NOTE: This is Not a member function
        """
        Counts the apparence frequency a token in a list of tokens from a single document
        
        Parameters:
        tokens (list[str]): A list of tokens from a single document

        Returns:
        Counter: A list of tuple pairs the token string and an integer of the frequency of token in tokens list
        """
        
        counter = Counter()
        counter.update(tokens)
        return counter

    def __extract_text_from_html_content(self, content: str) -> str: 
        """
        Extract the text content from a html content string. Ignores all of the markup and html tags
        and only returns the text content.

        Parameters:
        text (str): html content

        Returns:
        str: A string containing only the textual content from the html document.
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
        Parameters:
        file_path (str): File path to json document in local file storage

        Returns:
        dict[str, str]: returns the data stored in the json file as a python dictionary
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

    # Non-member functions
    @staticmethod
    def __compute_tf(term_freq: int, doc_length: int)->int: 
        return term_freq / doc_length

 