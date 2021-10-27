# import spacy
# import numpy as np
# from nltk.tokenize import sent_tokenize
# import nltk
import json
import sys

from transformers import AutoTokenizer
from vecto.corpus import Corpus


def simple_char_iter(text):
    for c in text:
        yield c


# def sentencize(text):
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # sents = [sent.text for sent in doc.sents]
    # return [s for s in sentence_iter(char_iter(text))]
    # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # return sent_detector.tokenize(text)


# TODO: ok let's do streaming sentence splitter
# ingest character by character
# append to sentence, unles
other_delimiters = {"?", "!", "。"}

known_abbreviations = {"md", "bs", "mr", "ms"}


def is_abbreviation(token):
    if "." in token:
        return True
    if len(token) == 1:
        return True
    if token.lower() in known_abbreviations:
        return True
    return False


def sentence_iter(char_iter):
    size_buffer = 10000
    buffer = [" "] * size_buffer
    pos = 0
    prev_char = ""
    prev_token = ""
    for c in char_iter:
        is_sentence_end = False
        if c == " " and prev_char == ".":
            # print(prev_token)
            if not is_abbreviation(prev_token[:-1]):
                is_sentence_end = True
        if prev_char in other_delimiters and c != "\"":
            is_sentence_end = True
            #buffer[pos] = c
            #pos += 1
        if is_sentence_end:
            if pos > 0:
                yield "".join(buffer[: pos]).strip()
            buffer = [" "] * size_buffer
            pos = 0
            continue
        prev_char = c
        if pos >= len(buffer):
            print("buffer overflow:")
            # print("".join(buffer[:100]))
            print("".join(buffer[-100:]))
            pos = 0
        buffer[pos] = c
        prev_token += c
        if c == " ":
            prev_token = ""
        pos += 1
    if pos > 0:
        yield "".join(buffer[: pos])


def preprocess():
    pass
    # TODO: ok read line by line, for time being let's ignore 


def main():
    # samples = []
    # samples.append("Hey how do you do? M.D. Bob is my friend. Mr. John too.")
    # samples.append("А по-русски слабо? Что делать с гос. служащими?")
    # samples.append("富士山が見える。こんにちは")
    # for s in samples:
    #     tokenized = sentencize(s)
    #     print(tokenized)
    path = "./tests/data/corpora/sentencise"
    # path = "/mnt/storage/Data/NLP/corpora/wiki_clean.txt"
    path = "/mnt/storage/Data/NLP/corpora/toronto_clean.txt"
    # path = "./quotes/13th_Reality-1.txt"
    name_tokenizer = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
    corpus = Corpus(path)
    corpus.load_dir_strucute()
    char_iter = corpus.get_character_iterator()
    sent_iter = sentence_iter(char_iter)
    # cnt = 0
    sample = []
    max_length = 128
    with open("lines.jsonl", "w") as f_out:
        for line in sent_iter:
            tokens = tokenizer(line, return_attention_mask=False)["input_ids"]
            if len(sample) + len(tokens) > max_length:
                sample = sample[:max_length]
                # print(len(sample))
                f_out.write(json.dumps(sample))
                f_out.write("\n")
                sample = []
            sample += tokens
            # print(tokenizer.convert_ids_to_tokens(tokens))
            # print(line)
            # print()
            # if cnt > 100:
                # break
            # cnt += 1


if __name__ == "__main__":
    main()
