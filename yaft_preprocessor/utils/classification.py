from abc import abstractmethod
from typing import Any

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier

from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering

from django.core.cache import caches


def collect_documents(documents, reset):
    classification_dataset = [] if reset else caches['classification'].get('classification_dataset', [])
    classification_dataset.extend(documents)
    caches['classification'].set('classification_dataset', classification_dataset)


class Classifier:

    @staticmethod
    def factory(method, param):
        from yaft_preprocessor.utils.knn_classifier import KNNClassifier
        from .naive_base_classifier import NaiveBayesClassifier
        if method == 'svm':
            return SVMClassifier(method, param)
        if method == 'rndfrst':
            return RandomForestClassifier(method, param)
        if method == 'naivebayes':
            return NaiveBayesClassifier(method, param)
        if method == 'knn':
            return KNNClassifier(method, param)
        if method == 'kmeans':
            return KMeansClustering(method, param)
        if method == 'gmm':
            return GMMClustering(method, param)
        if method == 'hierarchical':
            return HierarchicalClustering(method, param)

    def __init__(self, method: str, param: float) -> None:
        self.method = method
        self.param = param
        self.n = None
        super().__init__()

    @abstractmethod
    def classify_document(self, document) -> int:
        raise NotImplemented

    def classify_documents(self, documents: dict):
        result = {}
        for i, document in enumerate(documents):
            if i % 100 == 0:
                print(i)
            result[document['id']] = self.classify_document(document['vector'])
        return result

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
        return self.expend_vector_of_size(positional_vector, self.n)

    @staticmethod
    def expend_vector_of_size(positional_vector, n):
        result = []
        for i in range(n):
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


class RandomForestClassifier(Classifier):

    def train_using_training_set(self, x, y):
        self.classifier.fit(x, y)

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        self.classifier = SKRandomForestClassifier()

    def classify_document(self, document) -> int:
        return self.classifier.predict([self.expand_vector(document)])[0]


class KMeansClustering(Clusterer):
    def train_using_training_set(self, x, y):
        self.classifier.fit(x, y)

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
        self.classifier = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

    def classify_document(self, document) -> int:
        return self.classifier.predict([self.expand_vector(document)])[0]


class GMMClustering(Clusterer):
    def train_using_training_set(self, x, y):
        self.classifier.fit(x, y)

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
        self.classifier = mixture.GaussianMixture(n_components=5, covariance_type='spherical')

    def classify_document(self, document) -> int:
        return self.classifier.predict([self.expand_vector(document)])[0]


class HierarchicalClustering(Clusterer):
    def train_using_training_set(self, x, y):
        self.classifier.fit(x, y)

    def __init__(self, method: str, param: float) -> None:
        super().__init__(method, param)
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
        self.classifier = AgglomerativeClustering()

    def classify_document(self, document) -> int:
        return self.classifier.predict([self.expand_vector(document)])[0]

