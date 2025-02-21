from posting import Posting
from typing import List, Tuple
from collections import Counter
import gc
import shelve
import json


class InvertedIndex:
    """
    InvertedIndex that represents {token: Posting}
    """
    def __init__(self):
        self.indexDict = dict()  # {token: Posting}
        self.threshold = 10      # threshold for RAM->DISC conversion (NOTE: can declare as global variable or constant)
        self.size = 0


    def add(self, token: str, id_freq: Tuple[int, int]) -> None:
        # Check if size is above threshold
        if self.size >= self.threshold:
            self.partialize()
            self.size = 0

        if token in self.indexDict:
            self.indexDict[token].add(*id_freq)
        else:
            self.indexDict[token] = Posting(*id_freq)
        self.size +=1


    def partialize(self, filepath: str) -> bool:
        # How are we partializing?
        # 1. On-Air: read through data and modify data dynamically
        # 2. Load-Merge-Dump (LMD): load data into temp dictionary (local scope),
        #       merge with current data, and dump it back
        #    -> might be memory inefficient (consider putting threshold <= currentRAM * 0.5)
        self.__serialize(filepath)
        # TODO: Figure out what else we need to do other than serializing


    def __serialize(self, filepath: str) -> bool:
        """
        Parameter: filepath to shelve/json file that contains InvertedIndex's dictionary data
        Return: True if migration of data was successful
        Functionality:
            1. Opens filepath to write on.
            2. Merge its existing content with current InvertedIndex's dictionary data.
            3. Write it back to the file.
            4. Deallocate all data in current dictionary (and its elements including Posting) like C/C++ Destructor to clear RAM memory.
        """
        try:
            with open(filepath, "w") as f:
                pass  # TODO: Write code to save to to shelve/json file

            self.indexDict.clear()  # Remove all elements in indexDict
            gc.collect()            # Force garbage collection (deallocation)
            return True
        except Exception as e:
            print(f"[ERROR] {type(e)} from serialize(): Data failed to be serialized")
            return False
        

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
    
