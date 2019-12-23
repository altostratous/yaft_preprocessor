from collections import defaultdict

import numpy

from yaft_preprocessor.utils.classification import Classifier


class NaiveBayesClassifier(Classifier):

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        self.p_c = {}
        self.p_t_c = defaultdict(dict)

    def classify_document(self, document) -> int:
        scores = []
        for c in self.p_c:
            scores.append((self.calculate_document_score_for_class(document, c), c))
        return sorted(scores, reverse=True)[0][1]

    def train_using_training_set(self, x, y):
        label_counts = defaultdict(int)
        for label in y:
            label_counts[label] += 1
        for label in label_counts:
            self.p_c[label] = label_counts[label] / sum(label_counts.values())
        term_counts = defaultdict(lambda: defaultdict(float))
        for i in range(self.n):
            for j, v in enumerate(x):
                term_counts[i][y[j]] += v[i]
        for i in range(self.n):
            summation = sum(term_counts[i].values())
            for label in label_counts:
                self.p_t_c[i][label] = term_counts[i][label] / summation if summation else 0.5

    def calculate_document_score_for_class(self, document, c):
        return numpy.log(self.p_c[c]) + sum(map(numpy.log, [
            self.p_t_c.get(int(i), {}).get(c, 0.5) for i, w in document.items() if float(w) > 0
        ]))
