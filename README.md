# rag-kit

`rag-kit` is a simple, modular Python library for building PDF-based RAG applications with conversational memory and flexible LLM provider support.

It is designed to hide most of the LangChain complexity behind a clean API:

```python
from ragkit import PDFRAG

rag = PDFRAG("data/sample.pdf")
print(rag.ask("What is LangChain?"))
```

---

## Features

- PDF-based RAG
- Conversational chat with session memory
- Follow-up handling for queries like:
  - `hindi m batao`
  - `tell me in english`
  - `what did I ask earlier?`
- Query rewriting for better retrieval
- Source return support
- Configurable chunking and retrieval
- Multiple LLM provider support:
  - Sarvam (default)
  - OpenAI
  - Anthropic / Claude
  - Custom LangChain-compatible chat models

---

## Installation

### Basic install

```bash
pip install rag-kit
```

### Optional provider extras

```bash
pip install "rag-kit[openai]"
pip install "rag-kit[anthropic]"
pip install "rag-kit[all]"
```

### Local development install

```bash
pip install -e .
```

---

## Environment Variables

Create a `.env` file in your project root:

```env
SARVAM_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

An example template is provided in `.env.example`.

---

## Quick Start

### Stateless Q&A

```python
from ragkit import PDFRAG

rag = PDFRAG("data/sample.pdf")
answer = rag.ask("What is memory?")
print(answer)
```

### Chat with memory

```python
from ragkit import PDFRAG

rag = PDFRAG("data/sample.pdf")

session_id = "user1"

print(rag.chat("What is memory?", session_id=session_id))
print(rag.chat("hindi m batao", session_id=session_id))
print(rag.chat("tell me in english", session_id=session_id))
```

### Return sources

```python
from ragkit import PDFRAG

rag = PDFRAG("data/sample.pdf")
result = rag.ask("What is memory?", return_sources=True)

print(result["answer"])
print(result["sources"])
```

Example shape:

```python
{
    "answer": "Memory in LangChain stores previous conversation turns...",
    "sources": [
        {
            "content": "Memory in chat applications is created by storing earlier conversation turns...",
            "page": 2,
            "source": "data/sample.pdf",
            "metadata": {
                "page": 2,
                "source": "data/sample.pdf"
            }
        }
    ]
}
```

---

## ask() vs chat()

| Method | Purpose |
|---|---|
| `ask()` | Stateless document Q&A |
| `chat()` | History-aware conversational interaction |

Use `ask()` when you want a direct answer from the document.

Use `chat()` when you want:
- follow-up questions
- translation of the previous answer
- history-based conversation

---

## LLM Providers

### Default: Sarvam

```python
from ragkit import PDFRAG

rag = PDFRAG("file.pdf")
```

### OpenAI

```python
from ragkit import PDFRAG

rag = PDFRAG(
    "file.pdf",
    llm_provider="openai",
    llm_config={
        "model": "gpt-4o-mini",
        "temperature": 0.1,
    },
)
```

### Claude

```python
from ragkit import PDFRAG

rag = PDFRAG(
    "file.pdf",
    llm_provider="claude",
    llm_config={
        "model": "claude-3-5-haiku-latest",
        "temperature": 0.2,
    },
)
```

### Custom LLM

```python
from langchain_openai import ChatOpenAI
from ragkit import PDFRAG

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag = PDFRAG("file.pdf", llm=llm)
```

---

## Configuration

```python
from ragkit import PDFRAG, RAGConfig

config = RAGConfig(
    chunk_size=800,
    chunk_overlap=150,
    top_k=5,
    use_multi_query=True,
    enable_query_rewrite=True,
)

rag = PDFRAG("file.pdf", config=config)
```

Configurable options currently include:

- `persist_directory`
- `chunk_size`
- `chunk_overlap`
- `top_k`
- `use_multi_query`
- `enable_query_rewrite`
- `collection_name`
- `verbose`
- `llm_provider`
- `llm_model`
- `llm_temperature`
- `llm_kwargs`

---

## Add More Documents

```python
rag.add_documents("data/another.pdf")
```

---

## Reset Chat

```python
rag.reset_chat("user1")
```

---

## Project Structure

```text
rag-kit/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── examples/
├── data/
├── src/
│   └── ragkit/
└── third_party/
```

---

## Do You Need `requirements.txt`?

Not necessarily.

For modern Python packaging, `pyproject.toml` is enough and should be the main source of dependencies.

Use `requirements.txt` only if you want one of these:

- easier local setup for teammates
- pinned development environment
- quick install for people who do not use packaging workflows

### Recommendation

Keep:

- `pyproject.toml` as the main dependency file

Optional:

- `requirements-dev.txt` for local development and testing

Example `requirements-dev.txt`:

```txt
pytest
black
ruff
build
twine
```

If you want, you can also generate a plain `requirements.txt`, but it should not replace `pyproject.toml`.

---

## Current Limitations

- Primarily optimized for PDF-based RAG
- Sarvam support may depend on vendored or local integration setup
- No streaming support yet
- No FastAPI server or UI layer yet
- Agent support is planned, but not included in the current public API

---

## Roadmap

- Better source citations
- Improved multi-file indexing isolation
- Streaming responses
- FastAPI server mode
- Playground / UI
- Agent support via `ragkit.agent`

---

## Examples

Check the `examples/` folder for runnable examples such as:

- `basic_ask.py`
- `chat_example.py`
- `provider_openai.py`

---

## License

MIT License

---

## Version

Current version: `0.1.0-beta`

APIs may evolve in future releases.
