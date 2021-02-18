import bz2
import gzip
import json
import lzma
import os


def detect_archive_format_and_open(path):
    if path.endswith(".xz"):
        return lzma.open(path, mode="rt", encoding="utf-8", errors="replace")
    if path.endswith(".bz2"):
        return bz2.open(path, mode="rt", encoding="utf-8", errors="replace")
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf8", errors="replace")


def get_uncompressed_size(path):
    with detect_archive_format_and_open(path) as f:
        size = f.seek(0, 2)
    return size


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False))


def save_json(data, path):
    basedir = os.path.dirname(path)
    os.makedirs(basedir, exist_ok=True)
    str_data = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False)
    file_out = open(path, "w")
    file_out.write(str_data)
    file_out.close()


def load_json(path):
    f = open(path)
    s_data = f.read()
    data = json.loads(s_data)
    f.close()
    return data


def jsonify(data):
    if isinstance(data, list):
        return [jsonify(item) for item in data]
    if isinstance(data, dict):
        return {jsonify(key): jsonify(value) for key, value in data.items()}
    if isinstance(data, int):
        return str(data)
    if type(data).__module__ == "numpy":
        return data.tolist()
    return str(data)
