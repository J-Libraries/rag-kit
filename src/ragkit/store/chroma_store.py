from langchain_chroma import Chroma


def create_vector_store(docs, embeddings, persist_directory="db", collection_name="default"):
    return Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def load_vector_store(embeddings, persist_directory="db", collection_name="default"):
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )