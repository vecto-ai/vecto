class Resources(dict):
    def __init__(self, *args, **kwargs):
        super(Resources, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def wrap(data):
        if not isinstance(data, dict):
            return data
        else:
            return Resources({key: Resources.wrap(data[key])
                              for key in data})
