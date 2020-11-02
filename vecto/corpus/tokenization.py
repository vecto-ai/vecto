import nltk
import re
import abc

from vecto.utils.metadata import WithMetaData, get_full_typename

# TODO: ckeck id the data is there
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

_DEFAULT_WORD_SPLITTER = nltk.tokenize.WordPunctTokenizer().tokenize
_WHITESPACE_TOKEN_SPLITTER = re.compile(r'[^\s]+').findall

# we should not probably do it on module level
_SENT_SPLITTER_IMPL = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

DEFAULT_GOOD_TOKEN_RE = re.compile(r'^\w+$')
ANY_TOKEN_IS_GOOD_RE = re.compile(r'.*')

# TODO: moved from corpus, rename and use or remove
_default_tokenizer_patter = r"[\w\-']+|[.,!?…]"


def default_token_normalizer(token):
    return token.lower()


def word_tokenize_txt(txt,
                      token_splitter=_DEFAULT_WORD_SPLITTER,
                      token_normalizer=default_token_normalizer,
                      good_token_re=DEFAULT_GOOD_TOKEN_RE,
                      min_token_len=1,
                      stopwords=[]):
    # stopwords = nltk.corpus.stopwords.words('english')
    norm_tokens = map(token_normalizer, token_splitter(txt))
    return [token for token in norm_tokens
            if len(token) >= min_token_len and
            token not in stopwords]
            # and good_token_re.match(token)]


class BaseTokenizer(WithMetaData):
    """
    Base class for all tokenizer. It's a simple callable (functor) with metadata management infrastructure.
    """

    @abc.abstractmethod
    def __call__(self, txt):
        '''
        :param txt: text to tokenize
        :return: list of lists of tokens
        '''
        pass


class Tokenizer(BaseTokenizer):
    """
    Tokenizes text, normalizes each token with `token_normalizer`,
    filters tokens by length and regex `good_token_re`.
    Returns a list with the only element: list of tokens.
    This nesting is necessary to unify output with SentenceTokenizer,
    which returns list of sentences (each is a list of tokens).
    """

    def __init__(self,
                 token_splitter=_DEFAULT_WORD_SPLITTER,
                 token_normalizer=default_token_normalizer,
                 good_token_re=DEFAULT_GOOD_TOKEN_RE,
                 min_token_len=1,
                 stopwords=nltk.corpus.stopwords.words('english')):
        # TODO: decide how to save stopwords to metadata
        super().__init__(normalizer=get_full_typename(token_normalizer),
                         good_token_re=good_token_re.pattern,
                         min_token_len=min_token_len,
                         stopwords='too long to be saved to metadata')
        self.token_splitter = token_splitter
        self.token_normalizer = token_normalizer
        self.good_token_re = good_token_re
        self.min_token_len = min_token_len
        self.stopwords = stopwords

    def __call__(self, txt):
        return [word_tokenize_txt(txt,
                                  self.token_splitter,
                                  self.token_normalizer,
                                  self.good_token_re,
                                  self.min_token_len,
                                  self.stopwords)]


DEFAULT_TOKENIZER = Tokenizer()

ANNOTATED_TEXT_TOKENIZER = Tokenizer(token_splitter=_WHITESPACE_TOKEN_SPLITTER,
                                     good_token_re=ANY_TOKEN_IS_GOOD_RE,
                                     min_token_len=0)

DEFAULT_JAP_TOKENIZER = Tokenizer(min_token_len=0)


class SentenceTokenizer(BaseTokenizer):
    """
    Splits text into sentences, tokenizes each sentence, normalizes each token with `token_normalizer`,
    filters tokens by length and regex `good_token_re`.
    Returns a list of sentences (each is a list of tokens).
    """

    def __init__(self,
                 word_tokenizer=DEFAULT_TOKENIZER,
                 sentence_splitter=_SENT_SPLITTER_IMPL,
                 min_sent_words=2):
        super(SentenceTokenizer, self).__init__(word_tokenizer=word_tokenizer.metadata,
                                                sentence_splitter=get_full_typename(sentence_splitter),
                                                min_sent_words=min_sent_words)
        self.word_tokenizer = word_tokenizer
        self.sentence_splitter = sentence_splitter
        self.min_sent_words = min_sent_words

    def __call__(self, txt):
        for sent in self.sentence_splitter(txt.strip()):
            for sent_tokens in self.word_tokenizer(sent):
                if len(sent_tokens) >= self.min_sent_words:
                    yield sent_tokens


DEFAULT_SENT_TOKENIZER = SentenceTokenizer()
