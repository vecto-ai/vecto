from vecto.corpus.tokenization import word_tokenize_txt


# TODO: move this to corpus module
def normalize_text(text):
    return text.strip().lower()


def read_first_col_is_label_format(path):
    dataset = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f):
            if len(l.strip()) < 3:
                continue
            label, text = l.strip().split(None, 1)
            label = int(label)
            tokens = word_tokenize_txt(normalize_text(text))
            dataset.append((tokens, label))
    return dataset
