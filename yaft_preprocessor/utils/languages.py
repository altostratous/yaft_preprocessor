import re
from string import ascii_letters
from hazm import Normalizer, WordTokenizer, Stemmer
from hazm.utils import stopwords_list
from nltk.corpus import stopwords
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

FA = 'fa'
EN = 'en'
LANGUAGES = (FA, EN)

download('stopwords')
download('punkt')
LANGUAGE_STOPWORDS = {
    FA: set(stopwords_list()),
    EN: set(stopwords.words('english')),
}


class EnglishPunctuationSet:
    def __contains__(self, item):
        return not item.isalpha()


LANGUAGE_PUNCTUATION = {
    FA: '*\.:!،؛؟»\]\)\}«\[\(\{',
    EN: EnglishPunctuationSet(),
}


LANGUAGE_STEMMER = {
    FA: Stemmer().stem,
    EN: PorterStemmer().stem,
}

LANGUAGE_NORMALIZER = {
    EN: lambda x: x.lower(),
    FA: Normalizer().normalize,
}

LANGUAGE_TOKENIZER = {
    FA: WordTokenizer().tokenize,
    EN: word_tokenize,
}


def remove_stop_words(tokens, language):
    return {i: token for i, token in enumerate(tokens) if token not in LANGUAGE_STOPWORDS[language]}


def remove_punctuation(tokens, language):
    return [token for token in tokens if token not in LANGUAGE_PUNCTUATION[language]]


def preprocess_document_of_language(document: str, lang):
    normalized_document = LANGUAGE_NORMALIZER[lang](document)
    tokens = LANGUAGE_TOKENIZER[lang](normalized_document)
    words = remove_punctuation(tokens, lang)
    non_stop_word_words = remove_stop_words(words, lang)
    stemmed_words = {position: LANGUAGE_STEMMER[lang](word) for position, word in non_stop_word_words.items()}
    return stemmed_words


LANGUAGE_CONTAINS_LETTER = {
    EN: lambda letter: letter in ascii_letters,
    FA: lambda letter: re.compile('^[آ-ی]$').match(letter),
}


def get_document_language(document):
    for letter in document:
        for language in LANGUAGE_CONTAINS_LETTER:
            if LANGUAGE_CONTAINS_LETTER[language](letter):
                return language


def process_document_of_unknown_language(document):
    return preprocess_document_of_language(document, get_document_language(document))
