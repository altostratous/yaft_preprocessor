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
    FA: '\.:!،؛؟»\]\)\}«\[\(\{',
    EN: EnglishPunctuationSet(),
}


def remove_stop_words(tokens, language):
    return [token for token in tokens if token not in LANGUAGE_STOPWORDS[language]]


def remove_punctuation(tokens, language):
    return [token for token in tokens if token not in LANGUAGE_PUNCTUATION[language]]


def preprocess_fa_language(document: str):
    normalizer = Normalizer()
    normalized_document = normalizer.normalize(document)
    tokenizer = WordTokenizer()
    tokens = tokenizer.tokenize(normalized_document)
    words = remove_punctuation(tokens, FA)
    non_stop_word_words = remove_stop_words(words, FA)
    stemmer = Stemmer()
    stemmed_words = [stemmer.stem(word) for word in non_stop_word_words]
    return stemmed_words


def preprocess_en_document(document: str):
    normalized_document = document.lower()
    tokens = word_tokenize(normalized_document)
    words = remove_punctuation(tokens, EN)
    non_stop_word_words = remove_stop_words(words, EN)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in non_stop_word_words]
    return stemmed_words


LANGUAGE_PREPROCESSORS = {
    FA: preprocess_fa_language,
    EN: preprocess_en_document,
}


def preprocess_document_of_language(document: str, lang):
    preprocessor = LANGUAGE_PREPROCESSORS[lang]
    return preprocessor(document)


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
