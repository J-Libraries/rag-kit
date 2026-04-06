from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str | Path):
    loader = PyPDFLoader(str(path))
    return loader.load()