from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant.

Answer using:
1. the provided document context, and
2. the conversation history when the user refers to earlier messages.

Important follow-up rules:
- If the user says things like "hindi m batao", "tell me in english", "explain simply", "is it related to langchain?", or similar follow-ups, interpret them with respect to the immediately previous assistant answer or recent conversation.
- If the user asks about the conversation itself, answer from chat history.
- If the user asks about the document, answer from the provided context.
- If the answer is in neither context nor history, say exactly: "I don't know."
- Do not output <think> tags.
- Return only the final answer.
""",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )

    return prompt | llm