from collections import namedtuple

# Used for count Partial Index numbers
#   docCount = number of memory stored documents
#   indexNum = number of partial index being constructed
IndexCounter = namedtuple("IndexCounter", [docCount, indexNum])