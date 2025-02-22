from collections import defaultdict
from utils import get_logger
import json

# Constants 
DOC_THRESHOLD = 100
SHELVE_DB = "shelve/inverted_index.shelve"

class InvertedIndex: 
    def __init__(self): 
        self.index = defaultdict(list)
        self.doc_count = 0
        self.doc_id_map = {} # map file names to docid
        self.logger = get_logger("INVERTEDINDEX")

    def process_document(self, file_path, doc_id):
        pass

    def __read_json_file(self, filepath: str) -> dict[str, str]:
        """
        """
        try: 
            with open(filepath, 'r') as file: 
                data = json.load(file)
                self.logger.info(f"Json file loaded")
                return data
        except FileNotFoundError:
            self.logger.error(f"File note found at path: {filepath}")
            return None 
        except json.JSONDecodeError: 
            self.logger.error(f"Invalid JSON format in file:  {filepath}")
            return None 
        except Exception as e:
            logger.error(f"An unexpected error has orccurred: {e}") 
            return None 