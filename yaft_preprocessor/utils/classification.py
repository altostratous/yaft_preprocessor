from django.core.cache import caches, cache


def collect_documents(documents, reset):
    classification_dataset = [] if reset else caches['classification'].get('classification_dataset', [])
    classification_dataset.extend(documents)
    cache.set('classification_dataset', classification_dataset)
