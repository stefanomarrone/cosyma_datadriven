import pickle
from io import BytesIO
import polars as pl

class TrainedModel():
    def __init__(self):
        self.core = dict()

    def load(self, bytes):
        binary_buffer = BytesIO(bytes)
        self.core = pickle.load(binary_buffer)

    def getModel(self):
        return self.core['model']

    def getCartID(self):
        temporary_dictionary = self.core['cartID']
        lazy_df = pl.LazyFrame(temporary_dictionary)
        return lazy_df

    def getRanges(self):
        return self.core['ranges']

    def addModel(self, mdl):
        self.core['model'] = mdl

    def addCartID(self, cid):
        temporary_dictionary = cid.collect().to_dict(as_series = False)
        self.core['cartID'] = temporary_dictionary

    def addRanges(self, ranges):
        self.core['ranges'] = ranges

    def serialize(self):
        binary_buffer = BytesIO()
        pickle.dump(self.core, binary_buffer)
        return binary_buffer.getvalue()
