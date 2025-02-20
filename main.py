import sys
from preprocessor import Preprocessor


def run(rootpath: str) -> None:
    pp = Preprocessor(rootpath)
    pp.load_data()

    # NOTE: This code is testing/debugging purposes
    # TODO: DELETE at least before final submission
    for path, v in pp.get_data().items():
        print(f"{path}, {v.keys()}")
        break


if __name__ == "__main__":
    rootpath = sys.argv[1]  # relative path to root directory name ("/DEV")
    run(sys.argv[1])

