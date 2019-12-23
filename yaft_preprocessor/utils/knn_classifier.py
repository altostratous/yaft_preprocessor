from collections import defaultdict

import numpy

from yaft_preprocessor.utils.classification import Classifier


class KNNClassifier(Classifier):

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        self.intermediate = None
        self.t = 2478
        self.x = []
        self.x_shards = {}
        self.y_shards = {}
        self.y = []
        self.minimum = None
        self.maximum = None
        self.plains = None

    def classify_document(self, document) -> int:
        document_vector = numpy.array(self.expand_vector(document))
        key = self.get_key(document_vector)
        distance_values = [
            (sum((v - document_vector) ** 2), self.y_shards[key][i])
            for i, v in enumerate(self.x_shards[key])
        ]
        k = int(self.param)
        nearest_neighbours = sorted(distance_values[:k], key=lambda x: x[0])
        for distance, value in distance_values[k:]:
            if distance < nearest_neighbours[-1][0]:
                nearest_neighbours.append((distance, value))
                del nearest_neighbours[-1]
                nearest_neighbours.sort(key=lambda x: x[0])
        labels = defaultdict(int)
        for near_neighbour in nearest_neighbours:
            labels[near_neighbour[1]] += 1
        return sorted(labels.items(), key=lambda x: x[1], reverse=True)[0][0]

    def train_using_training_set(self, x, y):
        self.x = [numpy.array(v) for v in x]
        self.y = y
        # minimum = self.x[0]
        # maximum = self.x[0]
        # for v in self.x:
        #     maximum = numpy.maximum(maximum, v)
        #     minimum = numpy.minimum(minimum, v)
        self.intermediate = sum(self.x) / len(self.x)
        # self.intermediate = (maximum + minimum) / 2
        self.plains = [
            sum(self.intermediate[i:i + self.t])
            for i in range(0, len(self.intermediate), self.t)
        ]
        for i, v in enumerate(self.x):
            key = self.get_key(v)
            if key in self.x_shards:
                self.x_shards[key].append(v)
                self.y_shards[key].append(self.y[i])
            else:
                self.x_shards[key] = [v]
                self.y_shards[key] = [self.y[i]]
        # self.minimum = minimum
        # self.maximum = maximum

        self.y = y

    def get_key(self, v):
        key = ''
        for i in range(0, len(v), self.t):
            if sum(v[i: i + self.t]) > self.plains[int(i / self.t)]:
                key += '1'
            else:
                key += '0'
        return key
