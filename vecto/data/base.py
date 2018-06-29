from vecto.utils.metadata import WithMetaData


class Dataset(WithMetaData):
    """
    Container class for stock datasets.
    Arguments:
        path (str): local path to place files
    """

    def __init__(self, path):
        self.path = path

    # define iterators?
    # download
    # abd for description
