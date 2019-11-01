from rest_framework.views import APIView
from rest_framework.response import Response

from yaft_preprocessor.utils.languages import LANGUAGES
from yaft_preprocessor.utils.preprocess import preprocess_documents


class PreprocessView(APIView):

    def post(self, request):
        lang = request.GET.get('lang')
        if not lang or lang not in LANGUAGES:
            return Response({'error': 'Bad lang query parameter value.'}, status=400)
        data = request.data
        document_list = data.getlist('documents') if hasattr(data, 'getlist') else data.get('documents')
        return Response(preprocess_documents(document_list, lang))
