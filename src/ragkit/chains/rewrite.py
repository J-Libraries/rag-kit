from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_rewrite_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a query rewriter.

Your task is to rewrite the latest user message into a standalone search query.

Rules:
- Do not answer the question
- Do not explain anything
- Do not generate a paragraph
- Output only a short standalone query
- If the question is already standalone, return it unchanged
""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    return prompt | llm