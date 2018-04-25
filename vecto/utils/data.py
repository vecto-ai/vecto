import json
import gzip
import bz2


def detect_archive_format_and_open(path):
    if path.endswith(".bz2"):
        return bz2.open(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode='rt')
    return open(path)


def save_json(data, path):
    # if not os.path.isdir(path):
        # os.makedirs(path)
    s = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True)
    f = open(path, 'w')
    f.write(s)
    f.close()


def load_json(path):
    f = open(path)
    s_data = f.read()
    data = json.loads(s_data)
    f.close()
    return data
