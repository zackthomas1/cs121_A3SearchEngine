from typing import List, Tuple


class Posting:
    """
    Posting that represents {Document ID: frequency of token in that document}
    Used in Indexer class as following: {token: Posting}
    """
    def __init__(self, id: int = None, freq: int = None):
        """
        If id and freq are provided, initializes with one entry; otherwise, empty.
        """
        if id is not None and freq is not None:
            self.posting = {id: freq}
            self.size = 1
        else:
            self.posting = {}  # Initialize empty dictionary
            self.size = 0
        # NOTE: we can add self.token that saves corresponding token if needed (consider memeory problem)


    def add(self, id: int , freq: int) -> None:
        if id not in self.posting:
            self.posting[id] = freq
        else:
            self.posting[id] += freq
            # NOTE: Otherwise, consider rasing error depending on implementation
        self.size += 1


    def get_posting(self, sorted=True) -> List[Tuple[int, int]]:
        return sorted(self.posting.items()) if sorted else list(self.posting.items())


    def get_freq(self, id: int) -> int:
        return self.posting[id] if id in self.posting else -1
    

    def get_size(self) -> int:
        return self.size
    
