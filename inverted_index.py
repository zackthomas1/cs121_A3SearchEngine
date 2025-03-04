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
MASTER_INDEX_FILE = os.path.join(MASTER_INDEX_DIR, "master_index.json")
DOC_ID_MAP_FILE = os.path.join(DOC_ID_DIR, "doc_id_map.json")

#
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

class InvertedIndex: 
    def __init__(self):
        """ 
        Prepares to Index data by initializing storage directories and counter/keying variables.
        """
        
        self.index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # {token: [(docid, tf_score)]}
        self.doc_count = 0
        self.total_doc_count = 0  # total number of documents
        self.idf_scores: Dict[str, int] = defaultdict(int)  # value is total number of documents where token(key) appears
        self.index_tfidf: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # TODO: temporary for testing; memory overload expected (DELETE later)
        # self.current_index_file = self.get_latest_index_file()
        self.doc_id_map = {} # map file names to docid
        self.logger = get_logger("INVERTED_INDEX")
        self.re_alnum = re.compile(r"^[a-z0-9]+$")

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

        # Calculate IDF Score and Update self.idf_scores on Live
            # ln(# of doc / (count + 1)) ; add +1 to smoothen value & prevent error (division by zero) in case
            # Use numpy np.float32 to cut-off 50% RAM take-up (without it's 64 bits)
            # Round to 4 decimal precision for JSON file saving; Wouldn't make difference on ranking performance
        for token, count in self.idf_scores.items():
            self.idf_scores[token] = np.float32(round(np.log(self.total_doc_count / (count + 1)), 4))

    def build_master_index(self) -> None:
        """
        Combines all partial indexes into a single master index while preserving order.
        """
        
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

        # Calculate TF-IDF Score and Save to TF-IDF Score Dictionary for Now
        # TODO: consider where to save this value (RAM or DISC? Inside master_index Posting or Separate Data Structure?)
        for token, posting in master_index.items():
            for post in posting:
                self.index_tfidf[token].append(
                    ( post[0], np.float32(post[1]) * self.idf_scores[token] ))

        # Save master index to disk
        with open(MASTER_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(master_index, f, indent=4)

        self.logger.info(f"Master index built successfully and saved to {MASTER_INDEX_FILE}")
    
    def boolean_search(self, query: str) -> dict[str, list[tuple[int, int]]]:
        """
        Parameters:
        query (str): a query string 

        Returns:
        dict[str, list[tuple[int, int]]]: Inverted index containing only tokens formed from the query string
        """
        self.logger.info(f"Searching inverted index for query: {query}")
        
        tokens = InvertedIndex.__stem_tokens(self.__tokenize_text(query))
        query_token_index = self.__merge_from_disk(tokens)

        merged_results = {}

        # AND boolean implementation: merge docId results on token occurances
        for token in query_token_index:
            self.logger.info(f"\tProcessing token: {token}")
            # Initialize 'merged_results' if empty
            if not merged_results:
                merged_results = {docId: token_freq for docId, token_freq in query_token_index[token]}

            # Find and merge relevent documents
            else:
                relevent_documents = query_token_index[token]

                for docId, token_freq in relevent_documents:
                    if docId in merged_results:
                        merged_results[docId] += token_freq

        # TODO: tf-idf implementation would be somewhere here!
        # Sort the merged results by their "quality" [# of token frequency]
        return sorted(merged_results.items(), key=lambda kv: (-kv[1], kv[0]))

    def __process_document(self, file_path: str, doc_id: int):
        """
        Takes a file path to a document which stores an html page and updates the inverted index with tokens extracted from text content.
        Reads the file from disk. Extracts the html content. Updates the doc_id-url map. Tokenize the textual content.
        Update the inverted index with

        Parameters:
        file_path (str): The absolute file path to the document in the local file storage system
        doc_id (int): The unique id for the document at the provided file location 
        """

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
        Saves the Doc_ID-URL mapping to disk as a JSON file
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
        self.doc_count = 0 
        gc.collect()

    def __merge_from_disk(self, query_tokens: list[str]) -> dict[str, list[tuple[int, int]]]:
        """
        Loads only relevant part of index from disk for a given query.
        
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

    def __tokenize_text(self, text: str) -> list[str]:
        """
        Use nltk to tokenize text. Remove stop words and non alphanum
        
        Parameters:
        text (str): Text content parsed from an html document

        Returns:
        list[str]: a list of tokens extracted from the text content string
        """

        tokens =  word_tokenize(text.lower())
        # NOTE: Why not using built-in ".isalnum()" which is faster?
        return [token for token in tokens if self.re_alnum.match(token) and token not in STOPWORDS]
        # return tokenizer.tokenize(text)

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
        
        
    def __calculate_tfscore(self, token: str, text: str):
        """
        Call this function in loop for all documents like below.
            tfidf_table = 
            For text[contents] in all document files:
                for each token in each text[contents]:
                    tf = __calculate_tfscore(token, text)

        """
        text = self.__tokenize_text(text)
        tokenToFind = word_tokenize(token.lower())


    def __calculate_idfscore(self):
        pass


    
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

    # Non-member functions
    def __lemmatize_tokens(tokens: list[str]) -> list[str]:    # NOTE: This is Not a member function
        """
        Apply nltk lemmatization algorithm to extracted tokens
        
        Parameters:
        tokens (list[str]): a list of raw tokens 

        Returns:
        list[str]: a lemmatized list of tokens
        """
        return [lemmatizer.lemmatize(token) for token in tokens]

    def __stem_tokens(tokens: list[str]) -> list[str]:    # NOTE: This is Not a member function
        """
        Apply porters stemmer to tokens
        
        Parameters:
        tokens (list[str]): a list of raw tokens 

        Returns:
        list[str]: a lemmatized list of tokens
        """
        
        return [stemmer.stem(token) for token in tokens]