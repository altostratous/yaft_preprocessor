"""yaft_preprocessor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from yaft_preprocessor.views import PreprocessView, CompressView, DecompressView, IndexWordsView, PreprocessQueryView, \
    CollectDataSetView, ClassifyView, ClusterView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/preprocess_documents', PreprocessView.as_view()),
    path('api/v1/compress', CompressView.as_view()),
    path('api/v1/decompress', DecompressView.as_view()),
    path('api/v1/index_words', IndexWordsView.as_view()),
    path('api/v1/preprocess_query', PreprocessQueryView.as_view()),
    path('api/v1/collect_data_set', CollectDataSetView.as_view()),
    path('api/v1/classify', ClassifyView.as_view()),
    path('api/v1/cluster', ClusterView.as_view()),
]
