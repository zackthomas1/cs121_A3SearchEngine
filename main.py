import sys
import nltk
from preprocessor import Preprocessor
from indexer import InvertedIndex
import os
from typing import Dict
import json
from posting import Posting


NUMFILE = 1400


def run(rootpath: str, trialNo: int, onlyWriting: bool) -> None:
    if not onlyWriting:
        pp = Preprocessor(rootpath)
        #pp.load_data()
        iiwc = InvertedIndex()
        pp.genereate_inverted_index_from_json(iiwc)
        iiwc.merge_all_indexes()

        print(f"Token Count: {iiwc.getTokenCount()}")
        print(f"File Count: {pp.getFileCounts()}")
        print(f"File Size: {os.path.getsize('RESULT_grand_final_index.json') / 1024} KB")

        # # NOTE: This code is testing/debugging purposes
        # # TODO: DELETE at least before final submission
        # docid = 0
        # for path, data in pp.get_data().items():
        #     text = pp.parse_html(data["content"])
        #     tokens = pp.tokenize(text)
        #     tok_freq = pp.get_tok_freq(text)  # {token: freq}

        #     for tok, freq in tok_freq.items():
        #         iiwc.add(tok, (docid, freq))

        #     docid += 1
        #     # i += 1

        #     # if i >= threshold:
        #     #     break
        # iiwc.merge_all_indexes()
        # print(f"[SYSMSG] Merging process complete.")

        # print(f"[OUTPUT] Indexing Result of First {i} Files")
        
        # with open(f"output/trial{trialNo}.txt", "w") as f:
        #     print(iiwc, file=f)
    else:
        final_index: Dict[str, "Posting"] = {}
        for i in range(NUMFILE):
            filename = f"partial_index/index_{i}.json"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    partial: Dict[str, Dict[str, int]] = json.load(f)

                for token, post_info in partial.items():
                    if token in final_index:
                        final_index[token].update_posting(dict(post_info))
                    else:
                        final_index[token] = Posting.convert_to_posting_object(dict(post_info))


        with open(f"output/trial{trialNo}.txt", "w") as f:
            for tok, pos in final_index.items():
                json.dump({tok: pos.get_posting()}, f)

        with open(f"output/stat{trialNo}.txt", "w") as f:
            print(f"Total number of tokens: {len(final_index)}", file=f)



if __name__ == "__main__":
    #nltk.download("punkt")  # Download punkt (needed to run once)
    #nltk.download("punkt_tab")  # Download punkt_tab (needed to run once)

    sys.stdout.reconfigure(encoding='utf-8')

    rootpath = sys.argv[1]  # relative path to root directory name ("/DEV")
    trialNumber = sys.argv[2]

    run(rootpath, int(trialNumber), False)

