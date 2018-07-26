import json
import gzip
import bz2
import os


def detect_archive_format_and_open(path):
    if path.endswith(".bz2"):
        return bz2.open(path, mode='rt')
    if path.endswith(".gz"):
        return gzip.open(path, mode='rt')
    return open(path)


def save_json(data, path):
    basedir = os.path.dirname(path)
    os.makedirs(basedir, exist_ok=True)
    str_data = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=False)
    file_out = open(path, 'w')
    file_out.write(str_data)
    file_out.close()


def load_json(path):
    f = open(path)
    s_data = f.read()
    data = json.loads(s_data)
    f.close()
    return data


def jsonify(data):
    json_data = dict()
    for key, value in data.items():
        if isinstance(value, list):  # for lists
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):  # for nested lists
            value = jsonify(value)
        if isinstance(key, int):  # if key is integer: > to string
            key = str(key)
        if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data
