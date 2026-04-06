from ragkit import PDFRAG

rag = PDFRAG("data/sample.pdf")
session_id = "user1"

while True:
    query = input("Ask a question: ").strip()
    if query.lower() in {"exit", "quit"}:
        break

    answer = rag.chat(query, session_id=session_id, return_sources = True)
    print(answer)