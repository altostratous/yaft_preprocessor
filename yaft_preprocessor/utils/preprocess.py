from yaft_preprocessor.utils.languages import preprocess_document_of_language


def preprocess_document(document: str, lang):
    return preprocess_document_of_language(document, lang)


def preprocess_documents(data: list, lang):
    return [preprocess_document(document, lang) for document in data]
