import os
import re
import gc
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import clean_url, is_non_html_extension, get_logger
from typing import Dict, List, Tuple
from math import log as log_e
import numpy as np

# Constants 
STOPWORDS = set(stopwords.words('english'))
DOC_THRESHOLD = 250 # Dump index to latest JSON file every 100 docs
# NEW_FILE_THRESHOLD = 1000   # Create new index file every 1000 docs
DOC_ID_DIR = "index/doc_id_map"            # "index/doc_id_map"
PARTIAL_INDEX_DIR = "index/partial_index"  # "index/partial_index"
MASTER_INDEX_DIR = "index/master_index"    # "index/master_index"
MASTER_INDEX_FILE = os.path.join(MASTER_INDEX_DIR, "master_index.txt")
DOC_ID_MAP_FILE = os.path.join(DOC_ID_DIR, "doc_id_map.txt")

#
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

class InvertedIndex: 
    def __init__(self):
        """ Prepares to Index data by initializing storage directories and counter/keying variables. """
        self.index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # {token: [(docid, tf_score)]}
        self.doc_count = 0
        self.total_doc_count = 0  # total number of documents
        self.idf_scores: Dict[str, float] = defaultdict(int)  # value is total number of documents where token(key) appears
        self.index_tfidf: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # TODO: temporary for testing; memory overload expected (DELETE later)
        # self.current_index_file = self.get_latest_index_file()
        self.doc_id_map = {} # map file names to docid
        self.logger = get_logger("INVERTED_INDEX")
        self.re_alnum = re.compile(r"^[a-z0-9]+$")

        # Initializes directories for index storage
        os.makedirs(DOC_ID_DIR, exist_ok=True) 
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        os.makedirs(MASTER_INDEX_DIR, exist_ok=True)

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
            self.__save_index_to_disk()
            self.index.clear()
            gc.collect()

        # Calculate IDF Score and Update self.idf_scores on Live
            # ln(# of doc / (denom + 1)) ; add +1 to smoothen value & prevent error (division by zero) in case
            # Use numpy np.float32 to cut-off 50% RAM take-up (without it's 64 bits)
            # Round to 4 decimal precision for JSON file saving; Wouldn't make difference on ranking performance
        for token, denom in self.idf_scores.items():
            self.idf_scores[token] = np.float32(round(np.log(self.total_doc_count / (denom + 1)), 4))

    def build_master_index(self):
        """Combines all partial indexes into a single master index while preserving order."""
        self.logger.info(f"Building Master index...")

        master_index = defaultdict(list)

        # Iterate through all partial index files
        for file_name in sorted(os.listdir(PARTIAL_INDEX_DIR)):  # Ensure order is maintained
            self.logger.info(f"Adding: {file_name}")
            if file_name.startswith("index_part_") and file_name.endswith(".txt"):
                file_path = os.path.join(PARTIAL_INDEX_DIR, file_name)
                partial_index = defaultdict(list)
                InvertedIndex.__load_txt(file_path, partial_index)

                # Merge token postings while maintaining order
                for token, postings in partial_index.items():
                    master_index[token].extend(postings)

        # Calculate TF-IDF Score and Save to TF-IDF Score Dictionary for Now
        # TODO: consider where to save this value (RAM or DISC? Inside master_index Posting or Separate Data Structure?)
        for token, posting in master_index.items():
            for post in posting:
                #print(post[0], f"{round(np.float32(post[1]) * self.idf_scores[token], 4):.4f}")
                #print(type(np.float32(post[1])), type(self.idf_scores[token]), type(np.float32(post[1]) * self.idf_scores[token]))
                self.index_tfidf[token].append(
                    ( post[0], "{:.4f}".format(np.float32(post[1]) * self.idf_scores[token]) ))

        # Save master index to disk
        InvertedIndex.__dump_txt(MASTER_INDEX_FILE, master_index)

        self.logger.info(f"Master index built successfully and saved to {MASTER_INDEX_FILE}")
    
    def search(self, query): 
        self.logger.info(f"Searching for query tokens in inverted index: {query}")
        tokens = InvertedIndex.__stem_tokens(self.__tokenize_text(query))
        return self.__merge_from_disk(tokens)

    def __process_document(self, file_path, doc_id):
        # Read File to Process On
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

        # Text Extraction
        text = self.__extract_text_from_html_content(data['content'])
        if not text: 
            self.logger.warning(f"Skipping empty HTML text content: {file_path}")
            return

        self.__update_doc_id_map(doc_id, url)

        # Text Preprocessing (Tokenize & Stem) and Conut Token Frequency
        # self.logger.info(f"Tokenizing document content")
        tokens: List[str] = InvertedIndex.__stem_tokens(self.__tokenize_text(text))
        token_freq: Dict[str, int] = InvertedIndex.__construct_token_freq_counter(tokens)

        # self.logger.info(f"Updating inverted index")

        # IDF-Score's Denominator Value Calculation
        for unique_token in set(tokens):
            self.idf_scores[unique_token] += 1
        
        # Inverted Index Construction and TF-Score Calculation
        word_count = len(tokens)  # Get total number of words in the current document
        for token, freq in token_freq.items():
            tf_score = round((freq / word_count), 4)  # Calculate tf score: (word freq in cur doc / word cnt of cur doc)
            self.index[token].append((doc_id, tf_score))

        # Update Counters
        self.doc_count += 1        # Used for Partial Indexing
        self.total_doc_count += 1  # Used for IDF score calculation

        # Partial Indexing: If threshold is reached, store partial index and reset RAM
        if self.doc_count >= DOC_THRESHOLD: 
            self.__dump_to_disk()

    def __update_doc_id_map(self, doc_id, url): 
        self.doc_id_map[doc_id] = url

    def __save_doc_id_map_to_disk(self): 
        """Saves the Doc_ID-URL mapping to disk as .txt file"""
        existing_docidmap = {}
        if os.path.exists(DOC_ID_MAP_FILE):
            InvertedIndex.__load_txt_docid_map_file(existing_docidmap)

        for key, value in self.doc_id_map.items(): 
            existing_docidmap[key] = value

        # Write to doc_id_map.txt file    
        InvertedIndex.__dump_txt_docid_map_file(existing_docidmap)


    def __save_index_to_disk(self): 
        """Store current index to .txt file"""
        self.logger.info("Dumping index to disk")
        
        # Create a new .json partial index file
        index_file = os.path.join(PARTIAL_INDEX_DIR, f"index_part_{len(os.listdir(PARTIAL_INDEX_DIR))}.txt")

        # Check if .txt partial index file already existing index
        existing_data = defaultdict(list)
        if os.path.exists(index_file):
            InvertedIndex.__load_txt(index_file, existing_data)

        # Merge exisiting index with new data from in memory index
        for token, postings in self.index.items(): 
            if token in existing_data: 
                existing_data[token].extend(postings)
            else: 
                existing_data[token] = postings

        # Write index to file
        InvertedIndex.__dump_txt(index_file, existing_data)


    def __dump_to_disk(self): 
        """
        Saves in memory partial inverted index and doc_id map to
        disk, then clears memory. 
        """
        self.__save_index_to_disk()
        self.__save_doc_id_map_to_disk()
        self.index.clear()
        self.doc_id_map.clear()
        self.doc_count = 0 
        gc.collect()

    def __merge_from_disk(self, query_tokens):
        """Loads only relevant part of index from disk for a given query."""
        merged_index = {}
        for file_name in os.listdir(PARTIAL_INDEX_DIR): 
            file_path = os.path.join(PARTIAL_INDEX_DIR, file_name)
            index_part = defaultdict(list)
            InvertedIndex.__load_txt(file_path, index_part)
            for token in query_tokens: 
                if token in index_part: 
                    if token in merged_index:
                        merged_index[token].extend(index_part[token])
                    else: 
                        merged_index[token] = index_part[token]
        return merged_index

    def __construct_token_freq_counter(tokens) -> Counter:  # NOTE: This is Not a member function
        counter = Counter()
        counter.update(tokens)
        return counter

    def __lemmatize_tokens(tokens: list[str]) -> list[str]:    # NOTE: This is Not a member function
        return [lemmatizer.lemmatize(token) for token in tokens]

    def __stem_tokens(tokens: list[str]) -> list[str]:    # NOTE: This is Not a member function
        """Apply porters stemmer to tokens"""
        return [stemmer.stem(token) for token in tokens]

    def __tokenize_text(self, text: str) -> list[str]:
        """Use nltk to tokenize text. Remove stop words and non alphanum"""
        tokens =  word_tokenize(text.lower())
        # NOTE: Why not using built-in ".isalnum()" which is faster?
        return [token for token in tokens if self.re_alnum.match(token) and token not in STOPWORDS]

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
        try:  # NOTE: developer\DEV is given as json files so use json.load() instead of __load_txt() here
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
        
        
    def __load_txt(file_path: str, inverted_index: Dict[str, List[Tuple[int, float]]]) -> None:
        """Dumps inverted index from designated .txt file (usually used with partial_index)"""
        # NOTE: [LINE FORMAT] token;docid1,posting1 docid2,posting2 docid3,posting3\n
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # if len(tok_post) < 2:  # NOTE: Omitting bound checking for performance
                #     continue  # Skip the line if data in wrong format (either token or posting doesn't exist)
                token, postings_str = line.strip().split(";")
                postings = []
                for posting in postings_str.split():
                    docid, tfidf = posting.split(",")
                    postings.append((int(docid), np.float32(tfidf)))
                inverted_index[token] = postings


    def __dump_txt(file_path: str, inverted_index: Dict[str, List[Tuple[int, float]]]) -> None:
        """Dumps inverted index into designated .txt file (usually used with master_index)"""
        with open(file_path, "w", encoding="utf-8") as f:
            for token, postings in inverted_index.items():
                postings_str = " ".join([f"{docid},{tfidf}" for docid, tfidf in postings])
                f.write(f"{token};{postings_str}\n")

    
    def load_txt_docid_map_file(docid_map: Dict[str, str]) -> None:
        """Loads docid_url map from designated .txt file"""
        # NOTE: [LINE FORMAT] docid;url\n
        with open(DOC_ID_MAP_FILE, "r", encoding="utf-8") as f:
            for line in f:
                docid, url = line.strip().split(";")
                docid_map[docid] = url


    def __dump_txt_docid_map_file(docid_map: Dict[str, str]) -> None:
        """Dumps docid_url map into designated .txt file"""
        with open(DOC_ID_MAP_FILE, "w", encoding="utf-8") as f:
            for docid, url in docid_map.items():
                f.write(f"{docid};{url}\n")
    

    """
    # Pseudocode for tf-idf processing
    invIndex = InvertedIndex()
    number_of_doc_containing_token = {token: num_TinD}  // num_TinD: number of documents that contain token
    idf_score = {token: idf_score}

    for file in all_files:
        for token in text:  // text = file.extract["Content"]
            during building invIndex, save tf_score instead of frequency:
                tf_score = number_of_token_in_text / number_of_all_words_in_text
        
        if token in text:
            number_of_doc_containing_token[token] += 1
    
    for token, number in number_of_doc_containing_token:
        idf_val = math.log(CONST_TOTAL_NUM_DOC / (1 + number))
        idf_score[token] = idf_val    // Approach 1: Having separate invIndex and idfTable
        invIndex[token][0] = idf_val  // Approach 2: Change data structure so that it saves idf value in InvertedIndex
                                      //    Final Structure:  {token: [ idf, [ [docID, tf], ... ] ]}
    """