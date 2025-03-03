import os
import json
import time
from inverted_index import InvertedIndex
from inverted_index import PARTIAL_INDEX_DIR, MASTER_INDEX_DIR, MASTER_INDEX_FILE, DOC_ID_DIR, DOC_ID_MAP_FILE
from typing import List
from utils import get_logger
from summary import top_n_result_urls


if __name__ == "__main__":
    RESULT_NUM = 5

    index = InvertedIndex()
    logger = get_logger("SEARCHER")
    input_text = ""
    while (input_text != "quit"):
        print("Please enter the query you'd like to switch (or type 'quit' to exit)")
        input_text = input()

        if (input_text != "quit"):
            print(f'Searching for "{input_text}"')

            # Begin timing after recieving search query
            start_time = time.perf_counter() * 1000

            logger.info(f'Searching "{input_text}"')
            results = top_n_result_urls(input_text, RESULT_NUM, index, logger)
            for count, url in enumerate(results, start=1):
                print(f"{count}: {url}")
            end_time = time.perf_counter() * 1000
            logger.info(f"Completed search: {end_time - start_time:.0f} ms")

    print("'quit' detected, exiting...")
    logger.info("User ended searching.")
