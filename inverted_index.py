import os
import gc
import math
import json
import simhash
import sys
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import clean_url, compute_tf_idf, get_logger, read_json_file, write_json_file, tokenize_text, is_non_html_extension, is_xml
from datastructures import IndexCounter
from pympler.asizeof import asizeof

# Constants 
PARTIAL_INDEX_SIZE_THRESHOLD_KB = 20000  # set threshold to 20000 KB (margin of error: 5000 KB)
DOC_THRESHOLD_COUNT = 125


META_DIR            = "index/meta_data"    # "index/doc_id_map"
PARTIAL_INDEX_DIR   = "index/partial_index" # "index/partial_index"
MASTER_INDEX_DIR    = "index/master_index"  # "index/master_index"

MASTER_INDEX_FILE   = os.path.join(MASTER_INDEX_DIR, "master_index.json")
DOC_ID_MAP_FILE     = os.path.join(META_DIR, "doc_id_map.json")
META_DATA_FILE      = os.path.join(META_DIR, "meta_data.json")
DOC_NORMS_FILE      = os.path.join(META_DIR, "doc_norms.json")

class InvertedIndex: 
    def __init__(self):
        """ 
        Prepares to Index data by initializing storage directories and counter/keying variables.
        """
        # Note, modify the Tuple[] in the case you want to add more attributes to the posting
        self.alphanumerical_index: dict[str, dict[str, list[tuple[int, int, float]]]] = defaultdict(lambda: defaultdict(list)) # {letter/num: {token: [(docid, freq, tf_score)]}}
        self.alphanumerical_counts: dict[str, IndexCounter] = dict() # {letter/num: [number of current documents, current partial index num]}
        
        self.doc_id_map = {} # {doc_id: url}
        self.visited_content_simhashes = set()
        self.doc_count_partial_index = 0
        self.doc_id = 0
        self.total_doc_indexed = 0  # Tracks the number of documents successfully processed/indexed (not skipped).

        self.logger = get_logger("INVERTED_INDEX")

        # Initializes directories for index storage
        os.makedirs(META_DIR, exist_ok=True) 
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        os.makedirs(MASTER_INDEX_DIR, exist_ok=True)

        # Initializes directories a-z within the partial index
        for letter_ascii in range(ord('a'), ord('z') + 1):
            os.makedirs(PARTIAL_INDEX_DIR + '/' + chr(letter_ascii), exist_ok=True)

        # Initializes directories 0-9 within the partial index
        for num in range(10):
            os.makedirs(PARTIAL_INDEX_DIR + '/' + str(num), exist_ok=True)

    def build_index(self, folder_path: str) -> None: 
        """
        Process all JSON files in folder and build index.
        """

        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                self.logger.info(f"Indexing doc: {self.doc_id}")
                if file_name.endswith(".json"):
                    self.__process_document(os.path.join(root, file_name), self.doc_id)
                else:
                    self.logger.warning(f"File not does not end with .json extention: {file_name}")
                
                # Update counters
                self.doc_id += 1
                
        # Dump any remaining tokens to disk
        alphanumerical_indexes_modified = [alphanum_char for alphanum_char, partial_index in self.alphanumerical_index.items() if partial_index]
        self.__dump_to_disk(set(alphanumerical_indexes_modified), override=True)

        # Save index meta data disk
        self.__save_meta_data_to_disk()

    def build_master_index(self) -> None:
        """
        Combines all partial indexes into a single master index while preserving order.
        """

        master_index = defaultdict(list)

        self.logger.info(f"Building Master index...")

        # Iterate through all partial index files
        for dir_name in sorted(os.listdir(PARTIAL_INDEX_DIR)):  # Ensure order is maintained
            dir_path = os.path.join(PARTIAL_INDEX_DIR, dir_name)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                self.logger.info(f"Adding: {file_path}")

                partial_index = self.__read_partial_index_from_disk(file_path)

                # Merge token postings while maintaining order
                for token, postings in partial_index.items():
                    master_index[token].extend(postings)

        write_json_file(MASTER_INDEX_FILE, master_index, self.logger)
    
    def construct_merged_index_from_disk(self, query_tokens: list[str]) -> dict:
        """
        Constructs inverted index containing only query tokens from partial inverted index stored on disk
        
        Parameters:
            query_tokens (list[str]): 

        Returns:
            dict: inverted index which contains only the query tokens entries from the partial index
        """

        merged_index = {}

        for token in query_tokens:
            partial_index_char = token[0]
            dir_name = os.path.join(PARTIAL_INDEX_DIR + '/' + partial_index_char)
            for root, dirs, files in os.walk(dir_name):
                for file_name in files:
                    # Read in partial index from file
                    partial_index = self.__read_partial_index_from_disk(os.path.join(dir_name, file_name))
                    
                    # Search for token in partial index and add found query tokens to merged_index
                    if token in partial_index: 
                        if token in merged_index: 
                            merged_index[token].extend(partial_index[token])
                        else: 
                            merged_index[token] = partial_index[token]

        return merged_index

    def load_meta_data_from_disk(self) -> dict:
        """Load index meta file from disk"""
        return read_json_file(META_DATA_FILE, self.logger)

    def load_doc_id_map_from_disk(self) -> dict:
        """Load the doc_id map from disk to get urls"""
        return read_json_file(DOC_ID_MAP_FILE, self.logger)
    
    def load_doc_norms_from_disk(self) -> dict:
        """
        Loads the precomputed document norms from disk.

        Returns:
            dict[int, float]: A dictionary mapping document IDs(int) to their norm values(float).
        """
        doc_norms = read_json_file(DOC_NORMS_FILE, self.logger)
        doc_norms = {int(key): float(value) for key, value in doc_norms.items()}
        return doc_norms

    def load_master_index_from_disk(self) -> dict[str, list]:
        """
        Load master index from dist

        Returns: 
            dict[str, list]: A dictionary mapping token(str) to postings(list[tuple[int, int, float])
        """
        return read_json_file(MASTER_INDEX_FILE, self.logger)

    def precompute_document_norms(self) -> None:
        """
        Precomputes the Euclidean norm of each document's vector using tf-idf weighting. 

        The norm of a document is the squer root of the sum of squared tf-idf weights
        of all the tokens which are in that document. 

        Store precomputed norms in JSON file for look up at query time.
        """

        self.logger.info("Precomputing document normals...")

        # NOTE: do no use the master index
        master_index = self.load_master_index_from_disk()
        total_docs = self.total_doc_indexed
        doc_norms = defaultdict(float)
        
        # Compute document vector norms by summing squared token weights
        for token, postings in master_index.items():
            doc_freq = len(postings)
            for posting in postings:
                doc_id, freq, tf = posting
                weight = compute_tf_idf(tf, doc_freq, total_docs)
                doc_norms[doc_id] += weight ** 2

        # Take square root for each document
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

        try:
            # Save computed doc norms to disk
            with open(DOC_NORMS_FILE, "w", encoding="utf-8") as f: 
                json.dump(doc_norms, f, indent=4)
        except Exception as e:
            self.logger.error(f"Unable to save precomputed document norms to file: {e}")


        self.logger.info(f"Document norms saved to: {DOC_NORMS_FILE}")
        
    def __process_document(self, file_path: str, doc_id: int) -> None:
        """
        Takes a file path to a document which stores an html page and updates the inverted index with tokens extracted from text content.
        Reads the file from disk. Extracts the html content. Updates the doc_id-url map. Tokenize the textual content.
        Update the inverted index with

        Parameters:
            file_path (str): The absolute file path to the document in the local file storage system
            doc_id (int): The unique id for the document at the provided file location 
        """

        # Read json file from disk
        data = read_json_file(file_path, self.logger)
        if not data:
            self.logger.warning(f"Skipping empty JSON file: {file_path}")
            return
        
        # Extract url and check that is valid
        url = clean_url(data['url'])
        if is_non_html_extension(url):
            self.logger.warning(f"Skipping url with non html extension - {url}")
            return

        content = data['content']

        if not content: 
            self.logger.warning(f"Skipping doc {doc_id}: empty content - {url}")
            return
        if is_xml(content):
            self.logger.warning(f"Skipping doc {doc_id}: content is XML - {url}")
            return

        # Extract tokens from html content
        tokens = self.__extract_tokens_with_weighting(content)

        # Check for near and exact duplicate content (Simhash); Simhash also covers exact duplicate which has dist == 0
        current_page_hash = simhash.compute_simhash(tokens)
        for visited_page_hash in self.visited_content_simhashes:
            dist = simhash.calculate_hash_distance(current_page_hash, visited_page_hash)
            if dist == 0:  # Exact-duplicate
                self.logger.warning(f"Skipping doc {doc_id}: Exact Duplicate Content Match with Dist={dist} - {url}")
                return []
            elif dist < simhash.THRESHOLD:  # Near-duplicate
                self.logger.warning(f"Skipping doc {doc_id}: Near Duplicate Content Match with Dist={dist} - {url}")
                return []
        self.visited_content_simhashes.add(current_page_hash)

        # Update doc id map
        self.__update_doc_id_map(doc_id, url)

        # Tokenize text
        token_freq = InvertedIndex.__construct_token_freq_counter(tokens)

        # Update the inverted index with document tokens
        alphanumerical_indexes_modified = set() # Track which partial indexes are being updated this document
        for token, freq in token_freq.items():
            tf = InvertedIndex.__compute_tf(freq, len(tokens))

            # Append token to alphanumerical_index
            # Only process tokens that are alphanumerical
            if (token[0].lower().isalnum() and token[0].lower().isascii()):
                first_char = token[0].lower()
                self.alphanumerical_index[first_char][token].append((doc_id, freq, tf))

                # Track # of documents counted per alphanumerical character
                alphanumerical_indexes_modified.add(first_char)
        
        self.__dump_to_disk(alphanumerical_indexes_modified)

        self.total_doc_indexed += 1 

    def __update_doc_id_map(self, doc_id: int, url: str) -> None:
        """
        Updates the document id-url index with the provided doc_id url pair.
        Document id-url index records which url is associated with each doc_id number

        Parameters:
            doc_id (int): the unique identifier of the document
            url (str): url web address of the related document
        """

        self.doc_id_map[doc_id] = url

    def __save_meta_data_to_disk(self) -> None: 
        """
        Saves meta data about the inverted index to disk to be read back when preforming query
        """

        meta_data = {
            "corpus_size" : self.doc_id,
            "total_doc_indexed": self.total_doc_indexed
        }
        write_json_file(META_DATA_FILE, meta_data, self.logger)

    def __save_doc_id_map_to_disk(self) -> None: 
        """
        Saves the Doc_ID-URL mapping to disk as a JSON file
        """
        write_json_file(DOC_ID_MAP_FILE, self.doc_id_map, self.logger)

    def __save_index_to_disk(self, partial_index_char: str) -> None: 
        """
        Store the current in-memory partial index to a .txt file.
        Each line in the file is formatted as:
        token;doc_id1,freq1,tf1 doc_id2,freq2,tf2 ...
        """
        self.logger.info(f"Saving '{partial_index_char}' index to disk...")

        # Create a new .txt partial index file
        filepath = PARTIAL_INDEX_DIR + '/' + partial_index_char
        index_file = os.path.join(filepath, f"index_part_{self.alphanumerical_counts[partial_index_char].indexNum}.txt")
        try: 
            with open(index_file, "w", encoding="utf-8") as f:
                partial_index = self.alphanumerical_index[partial_index_char].items()
                # Merge exisiting index with new data from in memory index
                for token, postings in partial_index: 
                    # Serialize posting: each posting is represented as doc_id,freq,tf
                    postings_str = " ".join([f"{docid},{freq},{tf}" for docid, freq, tf in postings])
                    f.write(f"{token};{postings_str}\n")
            self.logger.info(f"Partial index saved to {index_file}")
        except Exception as e:
            self.logger.error(f"Unable to save partial index to disk: {index_file} - {e}")

    def __dump_to_disk(self, alphanumerical_indexes_modified: set, override: bool = False) -> None:
        """
        Saves partial inverted index and doc_id map to
        disk, then clears them from memory.
        """

        is_disk_index_updated = False
        # Note which partial indexes were updated and dump if necessary
        for char_modified in alphanumerical_indexes_modified:
            # Get the existing tuple of counts, initialize (0, 0) if none
            currentIndexCounter = self.alphanumerical_counts.get(char_modified, IndexCounter(docCount = 0, indexNum = 0))

            # Increment number of documents stored per character
            currentIndexCounter = IndexCounter(docCount = currentIndexCounter.docCount + 1, indexNum=currentIndexCounter.indexNum)
            self.alphanumerical_counts[char_modified] = currentIndexCounter

            # Dump partial index if it exceeds PARTIAL_INDEX_DOC_THRESHOLD
            if override or self.alphanumerical_counts[char_modified].docCount % DOC_THRESHOLD_COUNT == 0:
              if override or (asizeof(self.alphanumerical_index[char_modified]) / 1024) >= PARTIAL_INDEX_SIZE_THRESHOLD_KB:  # Compare in KB
                  is_disk_index_updated = True
                  self.__save_index_to_disk(char_modified)

                  # Reset count
                  currentIndexCounter = IndexCounter(docCount = 0, indexNum = currentIndexCounter.indexNum + 1)
                  self.alphanumerical_counts[char_modified] = currentIndexCounter

                  # Reset the partial index within memory
                  self.alphanumerical_index[char_modified].clear()
        
        if is_disk_index_updated: 
            # Update the doc_id map if index on disk is updated
            self.__save_doc_id_map_to_disk()
            self.doc_id_map.clear()

        gc.collect()

    def __extract_tokens_with_weighting(self, content: str, weigh_factor: int = 2) -> list[str]: 
        """
        Extract toekns from HTML content and applies extra wieght to tokens that 
        appear in important HTML tags (titles, h1, h2, h3, and strong). 

        Parameters:
            text (str): html content
            weight_factor (int): how much importance to assign tags

        Returns:
            list[str]: A combined list of tokens. Toeksn from important sections are replicated 
        """

        try:
            # Get the text from the html response
            soup = BeautifulSoup(content, 'html.parser')

            # Remove the text of CSS, JS, metadata, alter for JS, embeded websites
            for tag in soup.find_all(["style", "script", "meta", "noscript", "iframe"]):  
                tag.decompose()  # remove all markups stated above
            
            # soup contains only human-readable texts now to be compared near-duplicate
            general_text = soup.get_text(separator=" ", strip=True)
            general_tokens = tokenize_text(general_text)

            # Extract important text from specific tags and tokenize 
            important_tags = ["title", "h1", "h2", "h3", "strong"]
            important_text = ""
            for tag in important_tags: 
                for element in soup.find_all(tag): 
                    important_text += " " + element.get_text(separator=" ", strip=True)

            important_tokens = tokenize_text(important_text)

            # Weight important tokens by replicating them
            # Tokens from the important sections are multiplied by a weight factor. 
            # This effectively increases their frequency count.
            
            weighted_tokens = general_tokens + (important_tokens * weigh_factor)

            return weighted_tokens
        except Exception as e:
            self.logger.error(f"An unexpected error has orccurred: {e}") 
            return None
    
    def __read_partial_index_from_disk(self, file_path: str) -> dict:
        """
        Reads/deserializes partial inverted index from file
        [LINE FORMAT] token;docid1,freq1,tf1 docid2,freq2,tf2 docid3,freq3,tf3\n

        Parameters: 
            file_path (str): A file path to a partial index serialized in a .txt file 

        Returns:
            dict[str, list[tuple[int, int, float]]]:

        """
        inverted_index = defaultdict(list[tuple[int, int, float]])
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # if len(tok_post) < 2:  # NOTE: Omitting bound checking for performance
                    #     continue  # Skip the line if data in wrong format (either token or posting doesn't exist)
                    token, postings_str = line.strip().split(";")
                    postings = []
                    for posting in postings_str.split():
                        docid, freq, tf = posting.split(",")
                        postings.append((int(docid), int(freq), float(tf)))
                    inverted_index[token] = postings
        except Exception as e:
            self.logger.error(f"Unable to read .txt file: {e}")

        return inverted_index

    # Non-member functions
    @staticmethod
    def __compute_tf(term_freq: int, doc_length: int)->int: 
        return term_freq / doc_length
    
    @staticmethod
    def __construct_token_freq_counter(tokens: list[str]) -> Counter:
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
    