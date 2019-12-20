from django.core.cache import caches, cache
from rest_framework.status import HTTP_400_BAD_REQUEST
from rest_framework.views import APIView
from rest_framework.response import Response

from yaft_preprocessor.celery import train
from yaft_preprocessor.utils.classification import collect_documents
from yaft_preprocessor.utils.compression import COMPRESSION_TYPES, compress_lists, decompress_values
from yaft_preprocessor.utils.languages import LANGUAGES
from yaft_preprocessor.utils.preprocess import preprocess_documents
from yaft_preprocessor.utils.spell_correction import index_words, preprocess_query


class PreprocessView(APIView):

    def post(self, request):
        lang = request.GET.get('lang')
        if not lang or lang not in LANGUAGES:
            return Response({'error': 'Bad lang query parameter value.'}, status=400)
        data = request.data
        documents_dict = data.get('documents')
        return Response(preprocess_documents(documents_dict, lang))


class CompressView(APIView):

    def post(self, request):
        compression_type = request.GET.get('type')
        if not compression_type or compression_type not in COMPRESSION_TYPES:
            return Response({'error': 'Bad type query parameter value.'}, status=400)
        data = request.data
        integers_dict = data.get('integer_lists')
        return Response(compress_lists(integers_dict, compression_type))


class DecompressView(APIView):

    def post(self, request):
        compression_type = request.GET.get('type')
        if not compression_type or compression_type not in COMPRESSION_TYPES:
            return Response({'error': 'Bad type query parameter value.'}, status=400)
        data = request.data
        compressed_values = data.get('compressed_values')
        return Response(decompress_values(compressed_values, compression_type))


class IndexWordsView(APIView):

    def post(self, request):
        data = request.data
        words = data.get('words')
        reset = False
        if request.GET.get('reset') == 'true':
            reset = True
        index_words(words, reset)
        return Response({'status': 'success'}, 200)


class CollectDataSetView(APIView):

    def post(self, request):
        data = request.data
        documents = data.get('vectors')
        reset = False
        if request.GET.get('reset') == 'true':
            reset = True
        collect_documents(documents, reset)
        if reset:
            for key in caches['classification'].keys('classifier:*'):
                caches['classification'].delete(key)
        return Response({'status': 'success'}, 200)


def prepare_model(key, method, param):
    is_under_process = cache.get('classification_is_under_process')
    if is_under_process:
        return False
    train.delay(key, method, param)
    cache.set('classification_is_under_process', True)
    return True


class ClassifyView(APIView):

    def post(self, request):
        data = request.data
        documents = data
        try:
            method = request.GET.get('method')
            param = float(request.GET.get('param'))
        except ValueError:
            return Response(
                {'status': 'Please give `method` as string and `param` a float.'},
                status=HTTP_400_BAD_REQUEST
            )
        key = 'classifier:{}:{}'.format(method, param)
        already_classifier = caches['classification'].get(key)
        if already_classifier:
            Response(already_classifier.classify_documents(documents['vectors']), 200)
        scheduled = prepare_model(key, method, param)
        if scheduled:
            return Response({"status": "accepted", "detail": "Model is not ready yet"}, 202)
        else:
            return Response({"status": "to_many_requests", "detail": "Resource not enough."}, 429)


class PreprocessQueryView(APIView):

    def post(self, request):
        data = request.data
        query = data.get('query')
        return Response(preprocess_query(query), 200)
