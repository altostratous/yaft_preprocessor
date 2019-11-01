from yaft_preprocessor.utils.languages import preprocess_document_of_language


def preprocess_document(document: str, lang):
    return preprocess_document_of_language(document, lang)


def preprocess_documents(data: dict, lang):
    return {doc_id: preprocess_document(document, lang) for doc_id, document in data.items()}
