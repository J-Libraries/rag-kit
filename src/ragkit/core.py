from __future__ import annotations

from pathlib import Path

from .config import RAGConfig
from .ingestion.loaders import load_pdf
from .ingestion.splitters import split_docs
from .embeddings.factory import get_embeddings
from .store.chroma_store import create_vector_store, load_vector_store
from .retrieval.retriever import build_retriever
from .chains.rewrite import build_rewrite_chain
from .chains.answer import build_answer_chain
from .memory.history import ChatHistoryManager
from .utils.text import clean_response, is_bad_rewrite
from .providers.llm import get_llm


class PDFRAG:
    def __init__(
        self,
        pdf_path: str | Path | list[str | Path],
        config: RAGConfig | None = None,
        llm=None,
        embeddings=None,
        llm_provider: str | None = None,
        llm_config: dict | None = None,
    ):
        self.config = config or RAGConfig()
        self.pdf_paths = self._normalize_paths(pdf_path)

        provider = llm_provider or self.config.llm_provider
        provider_config = llm_config or {}

        self.llm = llm or get_llm(
            provider=provider,
            model=provider_config.get("model", self.config.llm_model),
            temperature=provider_config.get("temperature", self.config.llm_temperature),
            **{
                **self.config.llm_kwargs,
                **{
                    k: v
                    for k, v in provider_config.items()
                    if k not in {"model", "temperature"}
                },
            },
        )

        self.embeddings = embeddings or get_embeddings()
        self.history = ChatHistoryManager()

        self.vector_store = self._build_or_load_vector_store()
        self.retriever = build_retriever(
            vector_store=self.vector_store,
            llm=self.llm,
            config=self.config,
        )

        self.rewrite_chain = build_rewrite_chain(self.llm)
        self.answer_chain = build_answer_chain(self.llm)

        

    def ask(self, query: str, return_sources: bool = False):
        docs = self.retriever.invoke(query)

        if not docs or all(not d.page_content.strip() for d in docs):
            if return_sources:
                return {
                    "answer": "I don't know.",
                    "sources": [],
                }
            return "I don't know."

        context = self._build_context(docs)

        response = self.answer_chain.invoke({
            "history": [],
            "context": context,
            "question": query,
        })

        answer = clean_response(response.content)

        if return_sources:
            return {
                "answer": answer,
                "sources": self._format_sources(docs),
            }

        return answer

    def chat(self, query: str, session_id: str = "default", return_sources: bool = False) -> str:
        history = self.history.get(session_id)
        standalone_question = query

        use_history_only = self._is_history_question(query)

        if self.config.enable_query_rewrite and history.messages and not use_history_only:
            rewritten = self.rewrite_chain.invoke({
                "history": history.messages,
                "question": query,
            })
            candidate = clean_response(rewritten.content)

            if not is_bad_rewrite(candidate):
                standalone_question = candidate

        if use_history_only:
            context = ""
        else:
            docs = self.retriever.invoke(standalone_question)

            if not docs or all(not d.page_content.strip() for d in docs):
                context = ""
            else:
                context = self._build_context(docs)

        response = self.answer_chain.invoke({
            "history": history.messages,
            "context": context,
            "question": query,
        })

        final_answer = clean_response(response.content)

        if not final_answer:
            final_answer = "I don't know."

        history.add_user_message(query)
        history.add_ai_message(final_answer)
        if return_sources:
            return {
                "answer": final_answer,
                "sources": self._format_sources(docs),
            }

        return final_answer


    def add_documents(self, pdf_path: str | Path | list[str | Path]) -> None:
        new_paths = self._normalize_paths(pdf_path)

        all_docs = []
        for path in new_paths:
            all_docs.extend(load_pdf(path))

        chunks = split_docs(
            all_docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        self.vector_store.add_documents(chunks)
        self.pdf_paths.extend(new_paths)

    def reset_chat(self, session_id: str = "default") -> None:
        self.history.clear(session_id)

    def rebuild_index(self) -> None:
        all_docs = []
        for path in self.pdf_paths:
            all_docs.extend(load_pdf(path))

        chunks = split_docs(
            all_docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        self.vector_store = create_vector_store(
            docs=chunks,
            embeddings=self.embeddings,
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
        )

        self.retriever = build_retriever(
            vector_store=self.vector_store,
            llm=self.llm,
            config=self.config,
        )

    def _build_or_load_vector_store(self):
        persist_directory = Path(self.config.persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        chroma_file = persist_directory / "chroma.sqlite3"

        if chroma_file.exists():
            if self.config.verbose:
                print(f"[ragkit] Loading existing vector store from: {persist_directory}")

            return load_vector_store(
                embeddings=self.embeddings,
                persist_directory=persist_directory,
                collection_name=self.config.collection_name,
            )

        if self.config.verbose:
            print(f"[ragkit] Creating new vector store at: {persist_directory}")

        all_docs = []
        for path in self.pdf_paths:
            all_docs.extend(load_pdf(path))

        chunks = split_docs(
            all_docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        return create_vector_store(
            docs=chunks,
            embeddings=self.embeddings,
            persist_directory=persist_directory,
            collection_name=self.config.collection_name,
        )

    @staticmethod
    def _build_context(docs) -> str:
        return "\n\n".join(
            d.page_content.strip()
            for d in docs
            if d.page_content and d.page_content.strip()
        )

    @staticmethod
    def _normalize_paths(pdf_path: str | Path | list[str | Path]) -> list[str]:
        if isinstance(pdf_path, (str, Path)):
            return [str(pdf_path)]
        return [str(p) for p in pdf_path]


    @staticmethod
    def _is_history_question(query: str) -> bool:
        q = query.strip().lower()

        triggers = [
            "last message",
            "last thing i asked",
            "what did i ask",
            "what are the 2 things i asked",
            "what are the two things i asked",
            "previous message",
            "previous question",
            "in last",
            "earlier",
            "above",
            "tell me in english",
            "english m batao",
            "hindi m batao",
            "explain in hindi",
            "explain it in hindi",
            "translate in hindi",
            "translate to hindi",
            "translate to english",
            "tell me in hindi",
            "is it related to",
            "is this related to",
            "does it relate to",
            "what is it about",
            "explain simply",
            "explain in simple words",
            "summarize what we discussed",
        ]

        return any(t in q for t in triggers)


    @staticmethod
    def _format_sources(docs) -> list[dict]:
        sources = []

        for doc in docs:
            metadata = doc.metadata or {}

            sources.append(
                {
                    "content": doc.page_content,
                    "page": metadata.get("page"),
                    "source": metadata.get("source"),
                    "metadata": metadata,
                }
            )

        return sources