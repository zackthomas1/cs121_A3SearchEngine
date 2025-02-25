from typing import List, Tuple, Dict


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


    def __str__(self):
        output = []
        for tok, post in self.posting.items():
            output.append(f"    Doc {tok} - Freq {post}")
        return "\n".join(output)


    def add(self, id: int , freq: int) -> None:
        if id not in self.posting:
            self.posting[id] = freq
        else:
            self.posting[id] += freq
        self.size += 1


    def get_posting(self, sorting=True) -> List[Tuple[int, int]]:
        """
        Exports data by returning posting in dictionary (choose sorted/unsorted)
        """
        return sorted(self.posting.items()) if sorting else list(self.posting.items())
    

    def update_posting(self, data: Dict[str, int]) -> None:
        """
        Updates current Posting object's data
        """
        for id, frq in data.items():
            self.add(int(id), frq)
        self.size += len(data)


    @classmethod
    def convert_to_posting_object(cls, data: Dict[str, int]) -> "Posting":
        """
        Create new Posting object and returns it with data filled
        """
        postingObj = cls()  # Create new Posting object
        postingObj.posting = {int(id): frq for id, frq in data.items()}
        postingObj.size = len(data)
        return postingObj   # Return object with data filled


    def get_freq(self, id: int) -> int:
        return self.posting[id] if id in self.posting else -1
    

    def get_size(self) -> int:
        return self.size
    

    def clear_all(self) -> None:
        """
        Clear all posting dictionary data
        """
        self.posting.clear()
    
