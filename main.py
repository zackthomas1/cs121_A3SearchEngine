import os
from argparse import ArgumentParser

def read_json(filepath: str) -> None: 
    pass 

def parse_content() -> None:
    pass 

def crawl_data_set(root_dir: str) -> None: 
    directory_content = os.listdir(root_dir)
    for dir in directory_content: 
        print(dir) 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, default="developer/Dev")
    args = parser.parse_args()
    crawl_data_set(args.filename)