from celery import Celery
import django
django.setup()
from django.core.cache import caches, cache

from yaft_preprocessor.utils.classification import Classifier

app = Celery('tasks', broker='pyamqp://guest@localhost//')


@app.task
def train(key, method, param):
    print('computing {}'.format(key))
    classifier = Classifier.factory(method, param)
    classifier.train()
    print('setting {}'.format(key))
    caches['classification'].set(key, classifier)
    caches['classification'].set('classification_is_under_process', False)