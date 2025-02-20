import os
import json
from typing import Dict


def load_data(rootdir: str) -> Dict[str, Dict[str, str]]:
    """
    Use os.walk to traverse through subfolders and access .json files

    Parameter:
        The relative path to directory to start crawling local files (e.g. ..\DEV)

    Returns:
        data dictionary that has following elements:
            key: relative path to each JSON files
                ..\DEV\aiclub_ics_uci_edu\906c2...ecc3b.json
            value: dictionary that contains following elements:
                key: data type
                    ['url', 'content', 'encoding']
                value: corresponding values
                    "url": "https://aiclub.ics.uci.edu/"
                    "content": "<!DOCTYPE html>...</body>\r\n</html>"
                    "encoding": "utf-8"
                    NOTE: not yet exhasutively checked that content types are always same for all JSON files

    Reference Python Library Documents:
        os.walk:   https://docs.python.org/3.13/library/os.html#os.walk
        json.load: https://docs.python.org/3/library/json.html#json.load

    NOTE: few questions with assumptions is annotated with (ask TA). Delete when resolved.
    """
    data: Dict[str, Dict[str, str]] = {}  # {filepath: {data type (e.g. url, html content): corresponding data value}}

    # Read all JSON files in all subfolders and load into dictionary
    for root, _, files in os.walk(rootdir):  # os.walk returns [current folder, subfolders, files] and if subfolders exists, it 
        for file in files:
            if file.endswith(".json"):  # NOTE: If all files are guaranteed to be JSON, we can get rid of this. (ask TA)
                filepath = os.path.join(root, file)  # Generates full path (adaptive to operating system)
                with open(filepath, "r", encoding="utf-8") as fp:
                    data[filepath] = json.load(fp)  # json.load converts JSON file to Python Dictionary

    return data

