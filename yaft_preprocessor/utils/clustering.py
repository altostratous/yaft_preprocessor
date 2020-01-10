from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from yaft_preprocessor.utils.common import expend_vector_of_size

CLUSTERING_METHODS = {
    'kmeans': KMeans,
    'gmm': GaussianMixture,
    'hierarchical': AgglomerativeClustering
}


def cluster(documents, classifier_slug, k):
    ids, vectors = zip(*[(d['id'], d['vector']) for d in documents])
    n = max(max(map(int, vector.keys() or [-1])) + 1 for vector in vectors)
    vectors = [expend_vector_of_size(vector, n) for vector in vectors]
    return dict(zip(
        ids, CLUSTERING_METHODS[classifier_slug](k).fit_predict(vectors)
    ))
