import nltk
import re
import abc

nltk.download('punkt')
nltk.download('stopwords')

_WORD_TOKENIZER_IMPL = nltk.tokenize.WordPunctTokenizer()

# we should not probably do it on module level
_SENT_SPLITTER_IMPL = nltk.data.load('tokenizers/punkt/english.pickle')

DEFAULT_GOOD_TOKEN_RE = re.compile(r'^\w+$')

# TODO: moved from corpus, rename and use or remove
_default_tokenizer_patter = r"[\w\-']+|[.,!?…]"


def default_token_normalizer(t):
    return t.lower()


def word_tokenize_txt(txt,
                      token_normalizer=default_token_normalizer,
                      good_token_re=DEFAULT_GOOD_TOKEN_RE,
                      min_token_len=3,
                      stopwords=nltk.corpus.stopwords.words('english')):
    norm_tokens = map(token_normalizer, _WORD_TOKENIZER_IMPL.tokenize(txt))
    return [t for t in norm_tokens
            if len(t) >= min_token_len
            and (t not in stopwords)
            and good_token_re.match(t)]


class BaseTokenizer(object):
    @abc.abstractmethod
    def __call__(self, txt):
        '''
        :param txt: text to tokenize
        :return: list of lists of tokens
        '''
        pass


class Tokenizer(BaseTokenizer):
    def __init__(self,
                 token_normalizer=default_token_normalizer,
                 good_token_re=DEFAULT_GOOD_TOKEN_RE,
                 min_token_len=3,
                 stopwords=nltk.corpus.stopwords.words('english')):
        self.token_normalizer = token_normalizer
        self.good_token_re = good_token_re
        self.min_token_len = min_token_len
        self.stopwords = stopwords

    def __call__(self, txt):
        return [word_tokenize_txt(txt,
                                  self.token_normalizer,
                                  self.good_token_re,
                                  self.min_token_len,
                                  self.stopwords)]


DEFAULT_TOKENIZER = Tokenizer()


class SentenceTokenizer(BaseTokenizer):
    def __init__(self,
                 word_tokenizer=DEFAULT_TOKENIZER,
                 sentence_splitter=_SENT_SPLITTER_IMPL,
                 min_sent_words=2):
        self.word_tokenizer = word_tokenizer
        self.sentence_splitter = sentence_splitter
        self.min_sent_words = min_sent_words

    def __call__(self, txt):
        for sent in self.sentence_splitter.tokenize(txt.strip()):
            for sent_tokens in self.word_tokenizer(sent):
                if len(sent_tokens) >= self.min_sent_words:
                    yield sent_tokens


DEFAULT_SENT_TOKENIZER = SentenceTokenizer()
