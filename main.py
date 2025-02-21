import os
from argparse import ArgumentParser

"""
These are a few possible functions I thought of just off the top of my head.
They are not requirements for the project.
Please replace/delete them if you can think of a better architecture or high level functions.
"""

def read_json(filepath: str) -> None: 
    pass 

def parse_content() -> None:
    pass 

def crawl_data_set(root_dir: str) -> None: 
    directory_content = os.listdir(root_dir)
    for dir in directory_content: 
        print(dir) 

"""
Entry point
Call 'python main.py' from the command line to run program
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, default="developer/Dev")
    args = parser.parse_args()
    crawl_data_set(args.filename)