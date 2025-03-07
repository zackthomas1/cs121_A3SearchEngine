import os
import gc
import math
import json
import simhash
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import clean_url, compute_tf_idf, get_logger, tokenize_text, is_non_html_extension, is_xml

# Constants 
PARTIAL_INDEX_DOC_THRESHOLD = 250 # Dump index to latest JSON file every 100 docs

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
        
        self.index: dict[str, list[tuple[int, int, float]]] = defaultdict(list)  # {token: [(docid, freq, tf)]}
        self.doc_id_map = {} # {doc_id: url}
        self.visited_content_simhashes = []
        self.doc_count_partial_index = 0
        self.doc_count_total = 0

        self.logger = get_logger("INVERTED_INDEX")

        # Initializes directories for index storage
        os.makedirs(META_DIR, exist_ok=True) 
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        os.makedirs(MASTER_INDEX_DIR, exist_ok=True)

    def build_index(self, folder_path: str) -> None: 
        """
        Process all JSON files in folder and build index.
        """

        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                self.logger.info(f"Indexing doc: {self.doc_count_total}")
                if file_name.endswith(".json"):
                    self.__process_document(os.path.join(root, file_name), self.doc_count_total)
                else:
                    self.logger.warning(f"File not does not end with .json extention: {file_name}")
                
                # Update counters
                self.doc_count_partial_index += 1       # Used for Partial Indexing
                self.doc_count_total += 1

                # Partial Indexing: If threshold is reached, store partial index and reset RAM
                if self.doc_count_partial_index >= PARTIAL_INDEX_DOC_THRESHOLD: 
                    self.__dump_to_disk()
                
        # Dump any remaining tokens to disk
        if self.index: 
            self.__dump_to_disk()

        # Save index meta data disk
        self.__save_meta_data_to_disk()

    def build_master_index(self) -> None:
        """
        Combines all partial indexes into a single master index while preserving order.
        """

        master_index = defaultdict(list)

        self.logger.info(f"Building Master index...")

        # Iterate through all partial index files
        for file_name in sorted(os.listdir(PARTIAL_INDEX_DIR)):  # Ensure order is maintained
            self.logger.info(f"Adding: {file_name}")
            
            file_path = os.path.join(PARTIAL_INDEX_DIR, file_name)
            partial_index = self.__read_partial_index_from_disk(file_path)

            # Merge token postings while maintaining order
            for token, postings in partial_index.items():
                master_index[token].extend(postings)

        # Save master index to disk
        self.logger.info(f"Saving to file...")
        with open(MASTER_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(master_index, f, indent=4)

        self.logger.info(f"Master index built successfully and saved to {MASTER_INDEX_FILE}")
    
    def construct_merged_index_from_disk(self, query_tokens: list[str]) -> dict[str, list[tuple[int, int, float]]]:
        """
        Constructs inverted index containing only query tokens from partial inverted index stored on disk
        
        Parameters:
            query_tokens (list[str]): 

        Returns:
            dict[str, list[tuple[int, int, float]]]: inverted index which contains only the query tokens entries from the partial index
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
    
    def load_doc_id_map_from_disk(self) -> dict[str, str]:
        """Load the doc_id map to get urls"""

        if os.path.exists(DOC_ID_MAP_FILE):
            with open(DOC_ID_MAP_FILE, "r", encoding="utf-8") as f: 
                doc_id_map = json.load(f)
        else:
            self.logger.error("Unable to load doc id map. Path does not exist: {DOC_ID_MAP_FILE}")
            doc_id_map = {}

        return doc_id_map
    
    def load_doc_norms_from_disk(self) -> dict[int, float]:
        """
        Loads the precomputed document norms from disk.

        Returns:
            dict[int, float]: A dictionary mapping document IDs(int) to their norm values(float).
        """

        try:
            with open(DOC_NORMS_FILE, "r", encoding="utf-8") as f:
                doc_norms = json.load(f)

            # Coverty keys to int and values to float
            doc_norms = {int(key): float(value) for key, value in doc_norms.items()}
        except Exception as e:
            self.logger.error("Unable to load document norms. Path does not exist: {DOC_NORMS_FILE}")
            doc_norms = {}

        return doc_norms

    def load_master_index_from_disk(self) -> dict[str, list[tuple[int, int, float]]]:
        """
        Load master index from dist

        Returns: 
            dict[str, list[tuple[int, int, float]]]: A dictionary mapping token(str) to postings(list[tuple[int, int, float])
        """
        
        try:
            with open(MASTER_INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception as e:
            self.logger.error("Unable to load master index. Path does not exist: {MASTER_INDEX_FILE}")
            index_data = {}

        return index_data

    def precompute_document_norms(self) -> None:
        """
        Precomputes the Euclidean norm of each document's vector using tf-idf weighting. 

        The norm of a document is the squer root of the sum of squared tf-idf weights
        of all the tokens which are in that document. 

        Store precomputed norms in JSON file for look up at query time.
        """

        self.logger.info("Precomputing document normals...")

        master_index = self.load_master_index_from_disk()
        total_docs = self.doc_count_total
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
        data = self.__read_json_file(file_path)
        if not data:
            self.logger.warning(f"Skipping empty JSON file: {file_path}")
            return
        
        # Extract url and check that is valid
        url = clean_url(data['url'])
        if is_non_html_extension(url):
            self.logger.warning(f"Skipping url with non html extension")
            return

        content = data['content']
        if not content: 
            self.logger.warning(f"Skipping empty content: {file_path}")
            return
        if is_xml(content):
            self.logger.warning(f"Skipping content is XML: {file_path}")
            return

        # Extract tokens from html content
        tokens = self.__extract_tokens_with_weighting(content)

        # Check for near and exact duplicate content (Simhash); Simhash also covers exact duplicate which has dist == 0
        current_page_hash = simhash.compute_simhash(tokens)
        for visited_page_hash in self.visited_content_simhashes:
            dist = simhash.calculate_hash_distance(current_page_hash, visited_page_hash)
            if dist == 0:  # Exact-duplicate
                self.logger.warning(f"Skipping URL {url}: Exact Duplicate Content Match with Dist={dist}")
                return []
            elif dist < simhash.THRESHOLD:  # Near-duplicate
                self.logger.warning(f"Skipping URL {url}: Near Duplicate Content Match with Dist={dist}")
                return []
        self.visited_content_simhashes.add(current_page_hash)

        # Update doc id map
        self.__update_doc_id_map(doc_id, url)

        # Tokenize text
        token_freq = InvertedIndex.__construct_token_freq_counter(tokens)

        # Update the inverted index with document tokens
        # self.logger.info(f"Updating inverted index")
        for token, freq in token_freq.items():
            tf = InvertedIndex.__compute_tf(freq, len(tokens))
            self.index[token].append((doc_id, freq, tf))

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
            "doc_count_total": self.doc_count_total,
        }
        try:
            # write index to file
            with open(META_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Unable to save meta data to file: {e}")

    def __save_doc_id_map_to_disk(self) -> None: 
        """
        Saves the Doc_ID-URL mapping to disk as a JSON file
        """
        try: 
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
        except Exception as e:
            self.logger.error(f"Unable to save doc_id map to file: {e}")

    def __save_index_to_disk(self) -> None: 
        """
        Store the current in-memory partial index to a .txt file.
        Each line in the file is formatted as:
        token;doc_id1,freq1,tf1 doc_id2,freq2,tf2 ...
        """

        self.logger.info("Saving index to disk...")
        
        # Create a new .txt partial index file
        index_file = os.path.join(PARTIAL_INDEX_DIR, f"index_part_{len(os.listdir(PARTIAL_INDEX_DIR)):04d}.txt")
        try: 
            with open(index_file, "w", encoding="utf-8") as f:
                # Merge exisiting index with new data from in memory index
                for token, postings in self.index.items(): 
                    # Serialize posting: each posting is represented as doc_id,freq,tf
                    postings_str = " ".join([f"{docid},{freq},{tf}" for docid, freq, tf in postings])
                    f.write(f"{token};{postings_str}\n")

            self.logger.info(f"Partial index saved to {index_file}")
        except Exception as e:
            self.logger.error(f"Unable to save partial index to disk: {index_file} - {e}")

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
       
    def __read_partial_index_from_disk(self, file_path: str) -> dict[str, list[tuple[int, int, float]]]:
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
            self.logger.error(f"Unable to read file: {e}")

        return inverted_index

    # Non-member functions
    @staticmethod
    def __compute_tf(term_freq: int, doc_length: int)->int: 
        return term_freq / doc_length