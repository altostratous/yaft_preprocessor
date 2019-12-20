from abc import abstractmethod
from typing import Any

from sklearn.svm import LinearSVC

from django.core.cache import caches


def collect_documents(documents, reset):
    classification_dataset = [] if reset else caches['classification'].get('classification_dataset', [])
    classification_dataset.extend(documents)
    caches['classification'].set('classification_dataset', classification_dataset)


class Classifier:

    def __new__(cls, method: str, param: float) -> Any:
        if cls is Classifier:
            if method == 'svm':
                return super(Classifier, cls).__new__(SVMClassifier)
            if method == 'rndfrst':
                raise NotImplemented
        else:
            return super(Classifier, cls).__new__(cls, method, param)

    def __init__(self, method: str, param: float) -> None:
        self.method = method
        self.param = param
        self.n = None
        super().__init__()

    @abstractmethod
    def classify_document(self, document) -> int:
        raise NotImplemented

    def classify_documents(self, documents: dict):
        return {doc_id: self.classify_document(document) for doc_id, document in documents.items()}

    def train(self):
        training_set = caches['classification'].get('classification_dataset', [])
        self.n = max(max(map(int, document['vector'].keys())) for document in training_set) + 1
        x = []
        y = []
        for document in training_set:
            x.append(self.expand_vector(document['vector']))
            y.append(int(document['class']))
        self.train_using_training_set(x, y)

    @abstractmethod
    def train_using_training_set(self, x, y):
        raise NotImplemented

    def expand_vector(self, positional_vector):
        result = []
        for i in range(self.n):
            result.append(int(positional_vector.get(str(i), 0)))
        return result


class SVMClassifier(Classifier):

    def train_using_training_set(self, x, y):
        self.classifier.fit(x, y)

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        self.classifier = LinearSVC(C=self.param)

    def classify_document(self, document) -> int:
        return self.classifier.predict([self.expand_vector(document)])[0]
