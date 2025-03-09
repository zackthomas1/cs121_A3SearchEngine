import os
import gc
import math
import simhash
import pickle
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import clean_url, compute_tf_idf, get_logger, read_pickle_file, write_pickle_file, read_json_file, write_json_file, tokenize_text, stem_tokens, is_non_html_extension, is_xml
from datastructures import IndexCounter
from pympler.asizeof import asizeof

# Constants 
PARTIAL_INDEX_SIZE_THRESHOLD_KB = 14000  # set threshold to 20000 KB (margin of error: 5000 KB)
DOC_THRESHOLD_COUNT = 125

MASTER_INDEX_DIR        = "index/master_index"  # "index/master_index"
META_DIR                = "index/meta_data"    # "index/doc_id_map"
PARTIAL_INDEX_DIR       = "index/partial_index" # "index/partial_index"
TOKEN_TO_FILE_MAP_DIR   = "index/meta_data/token_to_file_map"

DOC_ID_MAP_FILE     = os.path.join(META_DIR, "doc_id_map.json")
DOC_LENGTH_FILE     = os.path.join(META_DIR, "doc_length.json")
DOC_NORMS_FILE      = os.path.join(META_DIR, "doc_norms.json")
MASTER_INDEX_FILE   = os.path.join(MASTER_INDEX_DIR, "master_index.json")
META_DATA_FILE      = os.path.join(META_DIR, "meta_data.json")

class InvertedIndex: 
    def __init__(self):
        """ 
        Prepares to Index data by initializing storage directories and counter/keying variables.
        """
        # Note, modify the Tuple[] in the case you want to add more attributes to the posting
        self.alphanumerical_index: dict[str, dict[str, list[tuple[int, int, float]]]] = defaultdict(lambda: defaultdict(list)) # {letter/num: {token: [(docid, freq, tf_score)]}}
        self.alphanumerical_counts: dict[str, IndexCounter] = dict() # {letter/num: [number of current documents, current partial index num]}
        
        self.doc_id_map                 = defaultdict(str) # {doc_id: url}
        self.doc_lengths                = defaultdict()
        self.visited_content_simhashes  = set()

        self.doc_id                     = 0
        self.doc_count_partial_index    = 0 # Rerset to zero every time a partial index is created
        self.total_doc_indexed          = 0 # Tracks the number of documents successfully processed/indexed (not skipped).
        self.doc_lengths_sum            = 0

        self.logger = get_logger("INVERTED_INDEX")

        # Initializes directories for index storage
        os.makedirs(MASTER_INDEX_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True) 
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        os.makedirs(TOKEN_TO_FILE_MAP_DIR, exist_ok=True) 

        # Initializes directories a-z within the partial index
        for letter_ascii in range(ord('a'), ord('z') + 1):
            os.makedirs(PARTIAL_INDEX_DIR + '/' + chr(letter_ascii), exist_ok=True)

        # Initializes directories 0-9 within the partial index
        for num in range(10):
            os.makedirs(PARTIAL_INDEX_DIR + '/' + str(num), exist_ok=True)

    def build_index(self, corpus_dir: str) -> None: 
        """
        Process all JSON files in folder and build index.
        """
        for root, dirs, files in os.walk(corpus_dir):
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
    
    def construct_merged_index_from_disk(self, query_tokens: list[str], token_to_file_map: dict) -> dict:
        """
        Constructs inverted index containing only query tokens from partial inverted index stored on disk
        
        Parameters:
            query_tokens (list[str]): 

        Returns:
            dict: inverted index which contains only the query tokens entries from the partial index
        """

        merged_index = {}

        for token in query_tokens:
            if token in token_to_file_map:
                    file_list = token_to_file_map[token]
                    self.logger.info(f"'{token}' found in {len(file_list)} file/s")
                    for file_path in file_list:
                        # Read in partial index from file
                        partial_index = self.__read_partial_index_from_disk(file_path)

                        # Search for token in partial index and add found query tokens to merged_index
                        if token in partial_index: 
                            if token in merged_index: 
                                merged_index[token].extend(partial_index[token])
                            else: 
                                merged_index[token] = partial_index[token]

        return merged_index

    def load_doc_id_map_from_disk(self) -> dict:
        """
        Load the doc_id map from disk to get urls
        Returns: 
            dict: A dictionary mapping doc id numbers to url strings
        """
        return read_json_file(DOC_ID_MAP_FILE, self.logger)
    
    def load_doc_lengths_from_disk(self) -> dict:
        """
        Load the doc length map from disk
        Returns: 
            dict: A dictionary mapping doc id numbers to length of the document
        """
        return read_json_file(DOC_LENGTH_FILE, self.logger)

    def load_doc_norms_from_disk(self) -> dict:
        """
        Loads the precomputed document norms from disk.
        Returns:
            dict: A dictionary mapping document IDs(int) to their norm values(float).
        """
        doc_norms = read_json_file(DOC_NORMS_FILE, self.logger)
        doc_norms = {int(key): float(value) for key, value in doc_norms.items()}
        return doc_norms

    def load_master_index_from_disk(self) -> dict:
        """
        Load master index from dist
        Returns: 
            dict: A dictionary mapping token(str) to postings(list[tuple[int, int, float])
        """
        return read_json_file(MASTER_INDEX_FILE, self.logger)

    def load_meta_data_from_disk(self) -> dict:
        """
        Load index meta file from disk
        
        Returns: 
            dict: A dictionary of inverted index meta data
        """
        return read_json_file(META_DATA_FILE, self.logger)
    
    def load_token_to_file_map_from_disk(self) -> dict:
        """
        Load token to file map from disk
        Returns: 
            dict: A dictionary mapping tokens(str) to files(str)
        """
        token_to_file_map = defaultdict(list)
        for file_name in os.listdir(TOKEN_TO_FILE_MAP_DIR):
            file_path = os.path.join(TOKEN_TO_FILE_MAP_DIR, file_name)
            char_token_to_file_map = read_pickle_file(file_path, self.logger)
            for token, files in char_token_to_file_map.items():
                token_to_file_map[token] = files

        return token_to_file_map

    def precompute_doc_norms(self) -> None:
        """
        Precomputes the Euclidean norm of each document's vector using tf-idf weighting. 

        The norm of a document is the squer root of the sum of squared tf-idf weights
        of all the tokens which are in that document. 

        Store precomputed norms in JSON file for look up at query time.
        """

        self.logger.info("Precomputing document normals...")

        total_docs = self.load_meta_data_from_disk()["total_doc_indexed"]
        
        # Compute global document frequency for each token
        global_df = defaultdict(int)
        for subdir in os.listdir(PARTIAL_INDEX_DIR):
            subdir_path = os.path.join(PARTIAL_INDEX_DIR, subdir)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                partial_index = self.__read_partial_index_from_disk(file_path)
                for token, postings in partial_index.items():
                    global_df[token] += len(postings)

        self.logger.info("Gloabl document frequencies computed.")

        # Compute document norms using tf-idf weights
        doc_norms = defaultdict(float)
        for subdir in os.listdir(PARTIAL_INDEX_DIR):
            subdir_path = os.path.join(PARTIAL_INDEX_DIR, subdir)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                partial_index = self.__read_partial_index_from_disk(file_path)
                for token, postings in partial_index.items():
                    df = global_df[token]
                    for doc_id, freq, tf, structural_weight in postings: 
                        weight = compute_tf_idf(tf, df, total_docs)
                        doc_norms[doc_id] += weight ** 2

        # Take square root to compute Euclidean norm
        for doc_id in doc_norms: 
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

        write_json_file(DOC_NORMS_FILE, doc_norms, self.logger)
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
        # tokens = self.__extract_tokens(content)
        tokens_with_freq_and_weight = self.__extract_content_structure(content)
        tokens = tokens_with_freq_and_weight.keys()

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
        self.__update_doc_lengths(doc_id, len(tokens))

        # Extract structure of content and tokens

        # Update the inverted index with document tokens
        alphanumerical_indexes_modified = set() # Track which partial indexes are being updated this document
        for token, (freq, structural_weight) in tokens_with_freq_and_weight.items():
            tf = InvertedIndex.__compute_tf(freq, len(tokens))

            # Append token to alphanumerical_index
            # Only process tokens that are alphanumerical
            if (token[0].lower().isalnum() and token[0].lower().isascii()):
                first_char = token[0].lower()
                self.alphanumerical_index[first_char][token].append((doc_id, freq, tf, structural_weight))

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

    def __update_doc_lengths(self, doc_id: int, doc_length: int) -> None: 
        """
        
        """
        self.doc_lengths_sum += doc_length
        self.doc_lengths[doc_id] = doc_length

    def __save_meta_data_to_disk(self) -> None: 
        """
        Saves meta data about the inverted index to disk to be read back when preforming query
        """
        total_length = self.doc_lengths_sum
        num_docs = self.total_doc_indexed
        doc_length_avg = total_length / num_docs if num_docs > 0 else 0.0

        meta_data = {
            "avg_doc_length": doc_length_avg,
            "corpus_size" : self.doc_id,
            "total_doc_indexed": self.total_doc_indexed,
        }
        write_json_file(META_DATA_FILE, meta_data, self.logger)

    def __save_doc_id_map_to_disk(self) -> None: 
        """
        Saves the Doc_ID-URL mapping to disk as a JSON file
        """
        write_json_file(DOC_ID_MAP_FILE, self.doc_id_map, self.logger)

    def __save_doc_lengths_to_disk(self) -> None: 
        """
        Saves the Doc_ID-URL mapping to disk as a JSON file
        """
        write_json_file(DOC_LENGTH_FILE, self.doc_lengths, self.logger)

    def __save_index_to_disk(self, partial_index_char: str) -> None: 
        """
        Store the current in-memory partial index to a .pkl file
        """
        self.logger.info(f"Saving '{partial_index_char}' index to disk...")
        
        # Create a new .txt partial index file
        filepath = PARTIAL_INDEX_DIR + '/' + partial_index_char
        index_file = os.path.join(filepath, f"index_part_{self.alphanumerical_counts[partial_index_char].indexNum}.pkl")
        write_pickle_file(index_file, self.alphanumerical_index[partial_index_char], self.logger)
            
        def save_token_to_file_map_disk() -> None: 
            # Update token-to-file mapping
            token_to_file_path = os.path.join(TOKEN_TO_FILE_MAP_DIR, f"token_to_file_map_{partial_index_char}.pkl")
            
            char_token_to_file_map = defaultdict(list)
            if os.path.exists(token_to_file_path): 
                char_token_to_file_map = read_pickle_file(token_to_file_path, self.logger)

            for token in self.alphanumerical_index[partial_index_char]:
                char_token_to_file_map[token].append(index_file)
            write_pickle_file(token_to_file_path, char_token_to_file_map, self.logger, True)

        save_token_to_file_map_disk()

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
            self.__save_doc_lengths_to_disk()
            self.doc_lengths.clear()

        gc.collect()

    def __extract_content_structure(self, content: str, weigh_factor: int = 2) -> dict[int, tuple[int,float]]: 
        """
        Extract tokens from HTML content and applies extra wieght to tokens that 
        appear in important HTML tags (titles, h1, h2, h3, and strong). 

        Parameters:
            text (str): html content
            weight_factor (int): how much importance to assign tags

        Returns:
            dict[int, tuple[int,float]]: A combined list of tokens. Toeksn from important sections are replicated 
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
            general_tokens = stem_tokens(general_tokens)  # Apply stemming

            token_counts = defaultdict(int)
            structural_weights = defaultdict(lambda: 1.0)

            # Count all tokens in the main body.
            for token in general_tokens:
                token_counts[token] += 1

            # Increase weight for tokens in the title.
            if soup.title:
                title_text = soup.title.get_text()
                title_tokens = tokenize_text(title_text)
                for token in title_tokens:
                    structural_weights[token] += 1.0  # bonus for title

            # Increase weight for tokens in header tags.
            for header in soup.find_all(['h1', 'h2', 'h3']):
                header_text = header.get_text()
                header_tokens = tokenize_text(header_text)
                for token in header_tokens:
                    structural_weights[token] += 0.75  # bonus for headers
            
            # Increase weight for tokens in bold text.
            for bold in soup.find_all(['b', 'strong']):
                bold_text = bold.get_text()
                bold_tokens = tokenize_text(bold_text)
                for token in bold_tokens:
                    structural_weights[token] += 0.5  # bonus for bold text

            # Combine counts and structural weights.
            tokens_structural_weights = {}
            for token in token_counts:
                tokens_structural_weights[token] = (token_counts[token], structural_weights[token])
            
            return tokens_structural_weights
        except Exception as e:
            self.logger.error(f"An unexpected error has orccurred: {e}") 
            return None

    def __extract_tokens(self, content: str, weigh_factor: int = 2) -> list[str]: 
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

            return general_tokens
        except Exception as e:
            self.logger.error(f"An unexpected error has orccurred: {e}") 
            return []
    
    def __read_partial_index_from_disk(self, file_path: str) -> dict:
        """
        Reads/deserializes partial inverted index from file
        [LINE FORMAT] token;docid1,freq1,tf1 docid2,freq2,tf2 docid3,freq3,tf3\n

        Parameters: 
            file_path (str): A file path to a partial index serialized in a .txt file 

        Returns:
            dict[str, list[tuple[int, int, float]]]:

        """
        return read_pickle_file(file_path, self.logger)

    # Non-member functions
    @staticmethod
    def __compute_tf(term_freq: int, doc_length: int)->float: 
        return term_freq / doc_length