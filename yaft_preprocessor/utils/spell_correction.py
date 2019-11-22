from collections import defaultdict
import editdistance
from django.core.cache import cache

from yaft_preprocessor.utils.languages import process_document_of_unknown_language


class BiwordIndex:

    dictionary = defaultdict(list)
    words = []
    word_set = set()

    def index_words(self, words: list):
        words = [word for word in words if word not in self.word_set]
        for word in words:
            self.word_set.add(word)
        self.update_index(*self.create_index(words))

    @staticmethod
    def create_index(words):
        dictionary = defaultdict(list)
        for i, word in enumerate(words):
            biwords = BiwordIndex.get_biwords(word)
            for biword in biwords:
                dictionary[biword].append(i)
        return dictionary, words

    @staticmethod
    def get_biwords(word):
        extended_word = '${}$'.format(word)
        biwords = [extended_word[j - 1: j + 1] for j in range(1, len(extended_word))]
        return biwords

    def update_index(self, dictionary, words):
        offset = len(self.words)
        self.words.extend(words)
        for biword, word_ids in dictionary.items():
            self.dictionary[biword].extend([offset + word_id for word_id in word_ids])

    def jaccard_lookup(self, word):
        biwords = self.get_biwords(word)
        set_of_query_biwords = set(biwords)
        results = defaultdict(int)
        for biword in biwords:
            for word_id in self.dictionary[biword]:
                results[word_id] += 1
        for word_id in list(results.keys()):
            dictionary_word = self.words[word_id]
            results[word_id] /= len(
                set_of_query_biwords | set(self.get_biwords(dictionary_word))
            )
        return {self.words[word_id]: results[word_id] for word_id in results}


def index_words(words: list, reset=False):
    biword_index = BiwordIndex() if reset else cache.get('biword_index', BiwordIndex())
    biword_index.index_words(words)
    cache.set('biword_index', biword_index)


def correct_spelling(word):
    index = cache.get('biword_index')
    words_similarity = index.jaccard_lookup(word)
    ten_most_similar_words = sorted(words_similarity.items(), key=lambda x: x[1], reverse=True)[:10]
    if not ten_most_similar_words:
        return word
    ten_most_similar_words = [
        (
            similar_word,
            (
                -editdistance.eval(word, similar_word),
                jaccard_similarity
            )
        ) for similar_word, jaccard_similarity in ten_most_similar_words
    ]
    return sorted(ten_most_similar_words, key=lambda x: x[1], reverse=True)[0][0]


def preprocess_query(query):
    words = get_preprocessed_words_in_order(query)
    result = []
    for word in words:
        word = correct_spelling(word)
        result.append(word)
    return result


def get_preprocessed_words_in_order(query):
    return [i[1] for i in sorted(process_document_of_unknown_language(query).items(), key=lambda x: x[0])]
