from rest_framework.views import APIView
from rest_framework.response import Response

from yaft_preprocessor.utils.preprocess import preprocess_documents


class PreprocessView(APIView):

    def post(self, request):
        data = request.data
        return Response(preprocess_documents(data))
