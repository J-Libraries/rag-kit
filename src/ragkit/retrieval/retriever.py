from langchain_classic.retrievers.multi_query import MultiQueryRetriever


def build_retriever(vector_store, llm, config):
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.top_k},
    )

    if config.use_multi_query:
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            include_original=True,
        )

    return base_retriever