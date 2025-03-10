import os
import gc
import math
import simhash
import pickle
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from utils import (clean_url, compute_tf_idf, get_logger, read_pickle_file,
                   write_pickle_file, read_json_file, write_json_file, tokenize_text,
                   is_non_html_extension, is_xml, generate_ngrams)
from datastructures import IndexCounter
from pympler.asizeof import asizeof

# Constants 
PARTIAL_INDEX_SIZE_THRESHOLD_KB = 9000  # set threshold to 20000 KB (margin of error: 5000 KB)
DOC_THRESHOLD_COUNT = 125

MASTER_INDEX_DIR        = "index/master_index"   # "index/master_index"
META_DIR                = "index/meta_data"        # "index/doc_id_map"
PARTIAL_INDEX_DIR       = "index/partial_index"    # "index/partial_index"
TOKEN_TO_FILE_MAP_DIR   = "index/meta_data/token_to_file_map"

DOC_ID_MAP_FILE     = os.path.join(META_DIR, "doc_id_map.json")
DOC_LENGTH_FILE     = os.path.join(META_DIR, "doc_length.json")
DOC_NORMS_FILE      = os.path.join(META_DIR, "doc_norms.json")
MASTER_INDEX_FILE   = os.path.join(MASTER_INDEX_DIR, "master_index.json")
META_DATA_FILE      = os.path.join(META_DIR, "meta_data.json")


class InvertedIndex:
    def __init__(self):
        """Prepares to index data by initializing storage directories and counter/keying variables."""
        # The inverted index: {letter/num: {token: [(doc_id, freq, tf, structural_weight)]}}
        self.alphanumerical_index: dict[str, dict[str, list[tuple[int, int, float, float]]]] = defaultdict(lambda: defaultdict(list))
        self.alphanumerical_counts: dict[str, IndexCounter] = dict()
        
        self.doc_id_map = defaultdict(str)
        self.doc_lengths = defaultdict()
        self.visited_content_simhashes = set()

        self.doc_id = 0
        self.doc_count_partial_index = 0
        self.total_doc_indexed = 0

        self.logger = get_logger("INVERTED_INDEX")

        os.makedirs(MASTER_INDEX_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True)
        os.makedirs(PARTIAL_INDEX_DIR, exist_ok=True)
        os.makedirs(TOKEN_TO_FILE_MAP_DIR, exist_ok=True)

        for letter_ascii in range(ord('a'), ord('z') + 1):
            os.makedirs(os.path.join(PARTIAL_INDEX_DIR, chr(letter_ascii)), exist_ok=True)
        for num in range(10):
            os.makedirs(os.path.join(PARTIAL_INDEX_DIR, str(num)), exist_ok=True)

    def build_index(self, corpus_dir: str) -> None:
        """
        Process all JSON files in a folder and build the index.
        """
        for root, dirs, files in os.walk(corpus_dir):
            for file_name in files:
                self.logger.info(f"Indexing doc: {self.doc_id}")
                if file_name.endswith(".json"):
                    # Call the private method __process_document
                    self.__process_document(os.path.join(root, file_name), self.doc_id)
                else:
                    self.logger.warning(f"File does not end with .json extension: {file_name}")
                self.doc_id += 1

        # Dump any remaining tokens to disk
        alphanumerical_indexes_modified = [
            char for char, index in self.alphanumerical_index.items() if index
        ]
        self.__dump_to_disk(set(alphanumerical_indexes_modified), override=True)
        self.__save_meta_data_to_disk()

    def build_master_index(self) -> None:
        """
        Combines all partial indexes into a single master index while preserving order.
        """
        master_index = defaultdict(list)
        self.logger.info("Building Master index...")
        # Iterate only over .pkl files in each subdirectory
        for dir_name in sorted(os.listdir(PARTIAL_INDEX_DIR)):
            dir_path = os.path.join(PARTIAL_INDEX_DIR, dir_name)
            for file_name in os.listdir(dir_path):
                if not file_name.endswith(".pkl"):
                    continue  # Skip non-pkl files
                file_path = os.path.join(dir_path, file_name)
                self.logger.info(f"Adding: {file_path}")
                partial_index = self.__read_partial_index_from_disk(file_path)
                for token, postings in partial_index.items():
                    master_index[token].extend(postings)
        write_json_file(MASTER_INDEX_FILE, master_index, self.logger)

    def construct_merged_index_from_disk(self, query_tokens: list[str], token_to_file_map: dict) -> dict:
        """
        Constructs an inverted index containing only query tokens from partial indexes stored on disk.
        """
        merged_index = {}
        for token in query_tokens:
            if token in token_to_file_map:
                file_list = token_to_file_map[token]
                self.logger.info(f"'{token}' found in {len(file_list)} file(s)")
                for file_path in file_list:
                    if not file_path.endswith(".pkl"):
                        continue
                    partial_index = self.__read_partial_index_from_disk(file_path)
                    if token in partial_index:
                        if token in merged_index:
                            merged_index[token].extend(partial_index[token])
                        else:
                            merged_index[token] = partial_index[token]
        return merged_index

    def load_doc_id_map_from_disk(self) -> dict:
        return read_json_file(DOC_ID_MAP_FILE, self.logger)

    def load_doc_lengths_from_disk(self) -> dict:
        return read_json_file(DOC_LENGTH_FILE, self.logger)

    def load_doc_norms_from_disk(self) -> dict:
        doc_norms = read_json_file(DOC_NORMS_FILE, self.logger)
        return {int(key): float(value) for key, value in doc_norms.items()}

    def load_master_index_from_disk(self) -> dict:
        return read_json_file(MASTER_INDEX_FILE, self.logger)

    def load_meta_data_from_disk(self) -> dict:
        return read_json_file(META_DATA_FILE, self.logger)

    def load_token_to_file_map_from_disk(self) -> dict:
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
        """
        self.logger.info("Precomputing document normals...")
        total_docs = self.load_meta_data_from_disk()["total_doc_indexed"]

        # Compute global document frequency for each token
        global_df = defaultdict(int)
        for subdir in os.listdir(PARTIAL_INDEX_DIR):
            dir_path = os.path.join(PARTIAL_INDEX_DIR, subdir)
            for file_name in os.listdir(dir_path):
                if not file_name.endswith(".pkl"):
                    continue
                file_path = os.path.join(dir_path, file_name)
                partial_index = self.__read_partial_index_from_disk(file_path)
                for token, postings in partial_index.items():
                    global_df[token] += len(postings)
        self.logger.info("Global document frequencies computed.")

        # Compute document norms using tf-idf weights
        doc_norms = defaultdict(float)
        for subdir in os.listdir(PARTIAL_INDEX_DIR):
            dir_path = os.path.join(PARTIAL_INDEX_DIR, subdir)
            for file_name in os.listdir(dir_path):
                if not file_name.endswith(".pkl"):
                    continue
                file_path = os.path.join(dir_path, file_name)
                partial_index = self.__read_partial_index_from_disk(file_path)
                for token, postings in partial_index.items():
                    df = global_df[token]
                    for doc_id, freq, tf, _ in postings:
                        weight = compute_tf_idf(tf, df, total_docs)
                        doc_norms[doc_id] += weight ** 2
        for doc_id in doc_norms:
            doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])
        write_json_file(DOC_NORMS_FILE, doc_norms, self.logger)
        self.logger.info(f"Document norms saved to: {DOC_NORMS_FILE}")

    def __update_doc_id_map(self, doc_id: int, url: str) -> None:
        self.doc_id_map[doc_id] = url

    def __update_doc_lengths(self, doc_id: int, doc_length: int) -> None:
        self.doc_lengths[doc_id] = doc_length

    def __save_meta_data_to_disk(self) -> None:
        total_length = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        doc_length_avg = total_length / num_docs if num_docs > 0 else 0.0
        meta_data = {
            "avg_doc_length": doc_length_avg,
            "corpus_size": self.doc_id,
            "total_doc_indexed": self.total_doc_indexed,
        }
        write_json_file(META_DATA_FILE, meta_data, self.logger)

    def __save_doc_id_map_to_disk(self) -> None:
        write_json_file(DOC_ID_MAP_FILE, self.doc_id_map, self.logger)

    def __save_doc_lengths_to_disk(self) -> None:
        write_json_file(DOC_LENGTH_FILE, self.doc_lengths, self.logger)

    def __save_index_to_disk(self, partial_index_char: str) -> None:
        self.logger.info(f"Saving '{partial_index_char}' index to disk...")
        filepath = os.path.join(PARTIAL_INDEX_DIR, partial_index_char)
        index_file = os.path.join(filepath, f"index_part_{self.alphanumerical_counts[partial_index_char].indexNum}.pkl")
        write_pickle_file(index_file, self.alphanumerical_index[partial_index_char], self.logger)

        def save_token_to_file_map_disk() -> None:
            token_to_file_path = os.path.join(TOKEN_TO_FILE_MAP_DIR, f"token_to_file_map_{partial_index_char}.pkl")
            char_token_to_file_map = defaultdict(list)
            if os.path.exists(token_to_file_path):
                char_token_to_file_map = read_pickle_file(token_to_file_path, self.logger)
            for token in self.alphanumerical_index[partial_index_char]:
                char_token_to_file_map[token].append(index_file)
            write_pickle_file(token_to_file_path, char_token_to_file_map, self.logger, True)
        save_token_to_file_map_disk()

    def __dump_to_disk(self, alphanumerical_indexes_modified: set, override: bool = False) -> None:
        is_disk_index_updated = False
        for char_modified in alphanumerical_indexes_modified:
            currentIndexCounter = self.alphanumerical_counts.get(char_modified, IndexCounter(docCount=0, indexNum=0))
            currentIndexCounter = IndexCounter(docCount=currentIndexCounter.docCount + 1, indexNum=currentIndexCounter.indexNum)
            self.alphanumerical_counts[char_modified] = currentIndexCounter
            if override or self.alphanumerical_counts[char_modified].docCount % DOC_THRESHOLD_COUNT == 0:
                if override or (asizeof(self.alphanumerical_index[char_modified]) / 1024) >= PARTIAL_INDEX_SIZE_THRESHOLD_KB:
                    is_disk_index_updated = True
                    self.__save_index_to_disk(char_modified)
                    currentIndexCounter = IndexCounter(docCount=0, indexNum=currentIndexCounter.indexNum + 1)
                    self.alphanumerical_counts[char_modified] = currentIndexCounter
                    self.alphanumerical_index[char_modified].clear()
        if is_disk_index_updated:
            self.__save_doc_id_map_to_disk()
            self.doc_id_map.clear()
            self.__save_doc_lengths_to_disk()
            self.doc_lengths.clear()
        gc.collect()

    def __extract_tokens_with_weighting(self, content: str, weigh_factor: int = 2) -> list[str]:
        """
        Extract tokens from HTML content and apply extra weight to tokens that 
        appear in important HTML tags (titles, h1, h2, h3, and strong). Additionally,
        generates bigrams and trigrams from both the general text and the important text,
        and combines them into a single list of tokens.
        
        Parameters:
            content (str): HTML content.
            weigh_factor (int): Multiplier for tokens extracted from important tags.
        
        Returns:
            list[str]: Combined list of tokens (unigrams, bigrams, trigrams) ready for indexing.
        """
        try:
            # Parse the HTML and remove unwanted tags
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup.find_all(["style", "script", "meta", "noscript", "iframe"]):
                tag.decompose()
            
            # --- Process general text ---
            general_text = soup.get_text(separator=" ", strip=True)
            general_tokens = tokenize_text(general_text)
            
            # Generate bigrams and trigrams for general tokens
            bigrams = generate_ngrams(general_tokens, 2)
            trigrams = generate_ngrams(general_tokens, 3)
            bigram_strings = [' '.join(bigram) for bigram in bigrams]
            trigram_strings = [' '.join(trigram) for trigram in trigrams]
            
            # --- Process important text ---
            important_tags = ["title", "h1", "h2", "h3", "strong"]
            important_text = ""
            for tag in important_tags:
                for element in soup.find_all(tag):
                    important_text += " " + element.get_text(separator=" ", strip=True)
            important_tokens = tokenize_text(important_text)
            
            # Generate n-grams for important tokens
            important_bigrams = generate_ngrams(important_tokens, 2)
            important_trigrams = generate_ngrams(important_tokens, 3)
            important_bigram_strings = [' '.join(bigram) for bigram in important_bigrams]
            important_trigram_strings = [' '.join(trigram) for trigram in important_trigrams]
            
            # --- Combine everything ---
            weighted_tokens = (
                general_tokens +
                bigram_strings +
                trigram_strings +
                (important_tokens * weigh_factor) +
                (important_bigram_strings * weigh_factor) +
                (important_trigram_strings * weigh_factor)
            )
            
            return weighted_tokens
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return None

    def __read_partial_index_from_disk(self, file_path: str) -> dict:
        return read_pickle_file(file_path, self.logger)

    @staticmethod
    def __compute_tf(term_freq: int, doc_length: int) -> int:
        return term_freq / doc_length

    @staticmethod
    def __construct_token_freq_counter(tokens: list[str]) -> Counter:
        counter = Counter()
        counter.update(tokens)
        return counter

    def __process_document(self, file_path: str, doc_id: int) -> None:
        """
        Takes a file path to a document which stores an HTML page and updates the inverted index
        with tokens extracted from the text content. It reads the file from disk, extracts the HTML content,
        updates the doc_id-URL map, tokenizes the textual content (including generating n-grams),
        and updates the inverted index.
        
        Parameters:
            file_path (str): The absolute file path to the document.
            doc_id (int): The unique id for the document.
        """
        # Read JSON file from disk
        data = read_json_file(file_path, self.logger)
        if not data:
            self.logger.warning(f"Skipping empty JSON file: {file_path}")
            return

        # Extract URL and check that it is valid
        url = clean_url(data['url'])
        if is_non_html_extension(url):
            self.logger.warning(f"Skipping doc {doc_id}: URL with non HTML extension - {url}")
            return

        content = data['content']
        if not content:
            self.logger.warning(f"Skipping doc {doc_id}: empty content - {url}")
            return
        if is_xml(content):
            self.logger.warning(f"Skipping doc {doc_id}: content is XML - {url}")
            return

        # --- Use the new n-gram token extraction ---
        tokens = self.__extract_tokens_with_weighting(content, weigh_factor=2)
        if tokens is None:
            return

        # Update document ID map and document length
        self.__update_doc_id_map(doc_id, url)
        self.__update_doc_lengths(doc_id, len(tokens))

        # Count token frequencies from tokens (including unigrams, bigrams, and trigrams)
        token_freq = InvertedIndex.__construct_token_freq_counter(tokens)

        # Update the inverted index with token frequencies
        for token, freq in token_freq.items():
            tf = InvertedIndex.__compute_tf(freq, len(tokens))
            if token and token[0].lower().isalnum() and token[0].lower().isascii():
                first_char = token[0].lower()
                # Structural weight is set to 1 here
                self.alphanumerical_index[first_char][token].append((doc_id, freq, tf, 1))
        
        # Dump updated partial indexes to disk
        modified_chars = {token[0].lower() for token in token_freq.keys() if token}
        self.__dump_to_disk(modified_chars)

        self.total_doc_indexed += 1
