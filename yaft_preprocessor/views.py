from rest_framework.views import APIView
from rest_framework.response import Response

from yaft_preprocessor.utils.compression import COMPRESSION_TYPES, compress_lists, decompress_values
from yaft_preprocessor.utils.languages import LANGUAGES
from yaft_preprocessor.utils.preprocess import preprocess_documents


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
