from posting import Posting
from typing import List, Tuple, Dict
from collections import Counter
import gc
import json
import os


class InvertedIndex:
    """
    InvertedIndex that represents {token: Posting}
    """
    def __init__(self, threshold=100000):
        self.indexDict = dict()  # {token: Posting}
        self.threshold = threshold      # threshold for RAM->DISC conversion (NOTE: can declare as global variable or constant)
        self.size = 0
        self.file_cnt = 0    # filename ID for each files (inverted index files)


    def __str__(self):
        output = []
        output.append("Printing InvertedIndex")
        for tok, post in self.indexDict.items():
            output.append(f"Posting for token: {tok}")
            output.append(str(post))  # Ensure post is converted to a string
        output.append("Printing Finished")
        return "\n".join(output)
    

    def __generateFilename(self) -> str:
        name = f"partial_index\index_{self.file_cnt}.json"
        self.file_cnt += 1
        return name


    def add(self, token: str, id_freq: Tuple[int, int]) -> None:
        # Check if size is above threshold
        # if self.size >= self.threshold:
        #     self.partialize()
        #     self.size = 0
        #print(f"[LOG] Adding token: {token} to IIWC as DOC {id_freq[0]} with freq {id_freq[1]}")
        if token in self.indexDict:
            self.indexDict[token].add(*id_freq)
        else:
            self.indexDict[token] = Posting(*id_freq)
        self.size += 1

        if self.size >= self.threshold:
            self.partialize()


    def partialize(self) -> str:
        """
        Does "partial indexing"
        Return: filename of generated JSON file
        """
        # How are we partializing?
        # 1. On-Air: read through data and modify data dynamically
        # 2. Load-Merge-Dump (LMD): load data into temp dictionary (local scope),
        #       merge with current data, and dump it back
        #    -> might be memory inefficient (consider putting threshold <= currentRAM * 0.5)
        filename = self.__generateFilename()

        try:
            data = {token: posting.get_posting() for token, posting in self.indexDict.items()}
            with open(filename, "w") as f:
                json.dump(data, f)
            print(f"[LOG] Saved Partial Index: {filename}")
            self.clear_indexDict()
        except Exception as e:
            print(f"[ERROR] {type(e)}: Partialize failed for {filename}")

    
    def clear_indexDict(self) -> None:
        for posting in self.indexDict.values():
            posting.clear_all()
        self.indexDict.clear()
        gc.collect()
        self.size = 0

    
    def merge_all_indexes(self) -> None:
        # Move remaining data to DISC before merging
        if len(self.indexDict) > 0:
            last_filename = self.partialize()
        print(f"[SYSMSG] Saved remaining RAM index to {last_filename}")

        # Merge all data into one dictionary
        final_index: Dict[str, "Posting"] = {}
        for i in range(self.file_cnt):
            filename = f"index_{i}.json"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    partial: Dict[str, Dict[str, int]] = json.load(f)

                for token, post_info in partial.items():
                    if token in final_index:
                        final_index[token].update_posting(post_info)
                    else:
                        final_index[token] = Posting.convert_to_posting_object(post_info)

        self.indexDict = final_index

        # with open("RESULT_grand_final_index.txt", "w") as f:
        #     json.dump(self.indexDict, f)

        self.writeOnFile("RESULT_grand_final_index.json", "RESULT_grand_final_RESULT.txt")

        print(f"[SYSMSG] Successfully Merged All Index Files {0}-{i}")


    def writeOnFile(self, finalIndexPath: str, statPath: str) -> None:
        with open(finalIndexPath, "w") as f:
            for tok, pos in self.indexDict.items():
                json.dump({tok: pos.get_posting()}, f)

        with open(statPath, "w") as f:
            print(f"Total number of tokens: {self.getTokenCount()}", file=f)


    def getTokenCount(self) -> int:
        return len(self.indexDict)


    # def __serialize(self, filepath: str) -> bool:
    #     """
    #     Parameter: filepath to shelve/json file that contains InvertedIndex's dictionary data
    #     Return: True if migration of data was successful
    #     Functionality:
    #         1. Opens filepath to write on.
    #         2. Merge its existing content with current InvertedIndex's dictionary data.
    #         3. Write it back to the file.
    #         4. Deallocate all data in current dictionary (and its elements including Posting) like C/C++ Destructor to clear RAM memory.
    #     """
    #     try:
    #         with open(filepath, "w") as f:
    #             pass  # TODO: Write code to save to to shelve/json file

    #         self.indexDict.clear()  # Remove all elements in indexDict
    #         gc.collect()            # Force garbage collection (deallocation)
    #         return True
    #     except Exception as e:
    #         print(f"[ERROR] {type(e)} from serialize(): Data failed to be serialized")
    #         return False
        

    def calculate_docScore(self) -> float:
        """
        Calculate document score (TF-IDF score)
        """
        pass


    # NOTE: Consider implementing ExtentList class in extent.py
    def add_to_extent(self, speciality: str) -> None:
        """
        Add element to extent list's entry depending on speciality (title, header, bold, etc)
            Example code: extent_list[speciality].append({docID: calculate_weight(speciality)})
        """
        pass


    # NOTE: Consider to move this implementation under ExtentList class
    def calculate_weight(self, speciality: str) -> int:
        """
        Calculate weight depending on sepciality of token
            For example, if title, weight=10; header, weight=5; bold, weight=3
        """
        pass
    
