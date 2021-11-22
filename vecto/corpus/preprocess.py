# import spacy
# import numpy as np
# from nltk.tokenize import sent_tokenize
# import nltk
import json
import random
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


def main():
    # samples = []
    # samples.append("Hey how do you do? M.D. Bob is my friend. Mr. John too.")
    # samples.append("А по-русски слабо? Что делать с гос. служащими?")
    # samples.append("富士山が見える。こんにちは")
    # for s in samples:
    #     tokenized = sentencize(s)
    #     print(tokenized)
    # path = "./tests/data/corpora/sentencise"
    path = sys.argv[1]
    # path = "/mnt/storage/Data/NLP/corpora/wiki_clean.txt"
    # path = "/mnt/storage/Data/NLP/corpora/toronto_clean.txt"
    # path = "./quotes/13th_Reality-1.txt"
    name_tokenizer = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
    corpus = Corpus(path)
    corpus.load_dir_strucute()
    char_iter = corpus.get_character_iterator()
    sent_iter = sentence_iter(char_iter)
    # cnt = 0
    sample = [tokenizer.cls_token_id]
    max_length = 128
    cnt = 0
    proba_shortening = 0.1
    with open("lines.jsonl", "w") as f_out:
        for line in sent_iter:
            tokens = tokenizer(line,
                               add_special_tokens=False,
                               return_attention_mask=False,)["input_ids"]
            sample += tokens
            if len(sample) > max_length - 10:
                sample = sample[:max_length - 1]
                min_length = 5
                if random.random() < proba_shortening:
                    sample = sample[: random.randint(min_length, len(sample))]
                sample += [tokenizer.sep_token_id]
                sample += [tokenizer.pad_token_id] * (max_length - len(sample))
                # print(len(sample))
                serialized = json.dumps(sample) 
                if ":" in serialized:
                    print(sample)
                    print(serialized)
                f_out.write(serialized)
                f_out.write("\n")
                #print(tokenizer.decode(sample))
                #print(len(sample))
                #print()
                sample = [tokenizer.cls_token_id]
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt, "last line", len(tokens))
            # print(tokenizer.convert_ids_to_tokens(tokens))
            # print(line)
            # print()
            # if cnt > 100:
                # break
            # cnt += 1


if __name__ == "__main__":
    main()
