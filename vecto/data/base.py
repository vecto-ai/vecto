import os
from vecto.utils.metadata import WithMetaData


class Dataset(WithMetaData):
    """
    Container class for stock datasets.
    Arguments:
        path (str): local path to place files
    """

    def __init__(self, path):
        if not os.path.exists(path):
            raise Exception("test dataset dir does not exist:" + path)
        super().__init__(path)
        self.path = path

    # define iterators?
    # download
    # abd for description
