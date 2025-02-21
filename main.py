import os
from argparse import ArgumentParser

"""
These are a few possible functions I thought of just off the top of my head.
They are not requirements for the project.
Please replace/delete them if you can think of a better architecture or high level functions.
"""

def stemm_tokens(tokens: list[str]) -> list[str]: 
    pass

def tokenize_content(content: str) -> list[str]: 
    pass 

def parse_json_file(filename: str) -> str:
    pass 

def walk_directory(root_dir: str) -> None: 
    # directory_content = os.walk(root_dir)
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs: 
            walk_directory(dir)

        for file in files:
            parse_json_file(file) 

"""
Entry point
Call 'python main.py' from the command line to run program
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, default="developer")
    args = parser.parse_args()
    walk_directory(args.filename)