import django
django.setup()
from celery import shared_task
from django.core.cache import caches, cache

from yaft_preprocessor.utils.classification import Classifier


@shared_task
def train(key, method, param):
    classifier = Classifier(method, param)
    classifier.train()
    caches['classification'].set(key, classifier)
    cache.set('classification_is_under_process', False)