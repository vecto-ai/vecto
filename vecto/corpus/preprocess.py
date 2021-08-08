# import spacy
# import numpy as np
# from nltk.tokenize import sent_tokenize
import nltk
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
delimiters = {".", "?"}


def sentence_iter(char_iter):
    size_buffer = 10000
    buffer = [" "] * size_buffer
    pos = 0
    for c in char_iter:
        buffer[pos] = c
        pos += 1
        if c in delimiters:
            yield "".join(buffer[: pos])
            buffer = [" "] * size_buffer
            pos = 0
    if pos > 0:
        yield "".join(buffer[: pos])


def preprocess():
    pass
    # TODO: ok read line by line, for time being let's ignore 


def main():
    samples = []
    samples.append("Hey how do you do? M.D. Bob is my friend. Mr. John too.")
    samples.append("А по-русски слабо? Что делать с гос. служащими?")
    samples.append("富士山が見える。こんにちは")
    # for s in samples:
    #     tokenized = sentencize(s)
    #     print(tokenized)
    path = "/home/blackbird/Projects_heavy/NLP/langmo/data/sense_small"
    corpus = Corpus(path)
    corpus.load_dir_strucute()
    char_iter = corpus.get_character_iterator()
    sent_iter = sentence_iter(char_iter)
    for line in sent_iter:
        print(line)
        print()


if __name__ == "__main__":
    main()
