import sys
import data_loader


def run(rootpath: str) -> None:
    data = data_loader.load_data(rootpath)

    # NOTE: This is code for testing & debugging
    # TODO: DELETE at least before submission
    for i, (k, v) in enumerate(data.items()):
        print(f"{k}, {v}")
        print(f"KEYS: {k}")
        print(f"Val's keys: {v.keys()}")
        if i == 1:
            break



if __name__ == "__main__":
    rootpath = sys.argv[1]  # relative path to root directory name ("/DEV")
    run(sys.argv[1])


