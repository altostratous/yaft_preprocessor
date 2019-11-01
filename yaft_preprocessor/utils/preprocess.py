from yaft_preprocessor.utils.languages import preprocess_document_of_unknown_language


def preprocess_document(document: str):
    return preprocess_document_of_unknown_language(document)


def preprocess_documents(data: list):
    return [preprocess_document(document) for document in data]
