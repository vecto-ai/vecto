from vecto.utils.metadata import WithMetaData


class Dataset(WithMetaData):
    """
    Container class for stock datasets.
    Arguments:
        path (str): local path to place files
    """
    def __init__(self, filename, url, size, path='.', subset_pct=100):
        # parameters to use in dataset config serialization
        super(Dataset, self).__init__(name=None)
        self.filename = filename
        self.url = url
        self.size = size
        self.path = path

    # define iterators?
    # download
    # abd for description
