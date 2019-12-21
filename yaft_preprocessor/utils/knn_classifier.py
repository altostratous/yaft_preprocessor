from collections import defaultdict

import numpy

from yaft_preprocessor.utils.classification import Classifier


class KNNClassifier(Classifier):

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        self.x = []
        self.y = []

    def classify_document(self, document) -> int:
        document_vector = numpy.array(self.expand_vector(document))
        distance_values = [(sum((v - document_vector) ** 2), self.y[i]) for i, v in enumerate(self.x)]
        k = int(self.param)
        nearest_neighbours = sorted(distance_values[:k], key=lambda x: x[0])
        for distance, value in distance_values[k:]:
            if distance < nearest_neighbours[-1][0]:
                nearest_neighbours.append((distance, value))
                nearest_neighbours = sorted(nearest_neighbours, key=lambda x: x[0])
                nearest_neighbours = nearest_neighbours[:k]
        labels = defaultdict(int)
        for near_neighbour in nearest_neighbours:
            labels[near_neighbour[1]] += 1
        return sorted(labels.items(), key=lambda x: x[1], reverse=True)[0][0]

    def train_using_training_set(self, x, y):
        self.x = [numpy.array(v) for v in x]
        self.y = y
