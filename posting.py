from typing import List, Tuple


class Posting:
    """
    Posting that represents {Document ID: frequency of token in that document}
    """
    def __init__(self):
        self.posting = dict()
        self.size = 0


    def add(self, id, freq) -> None:
        if id not in self.posting:
            self.posting[id] = freq
        else:
            self.posting[id] += freq
            # NOTE: Otherwise, consider rasing error depending on implementation
        self.size += 1


    def get_posting(self, sorted=True) -> List[Tuple[int, int]]:
        return sorted(self.posting.items()) if sorted else list(self.posting.items())


    def get_freq(self, id) -> int:
        return self.posting[id] if id in self.posting else -1
    

    def get_size(self) -> int:
        return self.size
    
