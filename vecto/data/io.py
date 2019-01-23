from vecto.corpus.tokenization import word_tokenize_txt


# TODO: move this to corpus module
def normalize_text(text):
    return text.strip().lower()


def read_first_col_is_label_format(path, char_based=False):
    dataset = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f):
            if len(l.strip()) < 3:
                continue
            label, text = l.strip().split(None, 1)
            # TODO: make lower-casing optional
            text = normalize_text(text)
            label = int(label)
            if char_based:
                tokens = list(text)
            else:
                tokens = word_tokenize_txt(text)
            dataset.append((tokens, label))
    return dataset
