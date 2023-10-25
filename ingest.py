from pathlib import Path

from langchain.document_loaders import PDFPlumberLoader
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore import get_vectorstore

settings = get_settings()

record_manager = SQLRecordManager(
    namespace=f"weaviate/{settings.INDEX_NAME}",
    db_url=settings.INDEX_DATABASE_URL,
)

record_manager.create_schema()

vectorstore = get_vectorstore()


def get_pdf_files():
    pdf_files: list[Path] = []
    for path in Path("./data/").rglob("*.pdf"):
        pdf_files.append(path)
    return pdf_files


def load_pdf_file(pdf_file: str):
    pdf_loader = PDFPlumberLoader(
        file_path=pdf_file,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
    )

    docs = pdf_loader.load()
    docs = text_splitter.split_documents(docs)

    return docs


def load_data():
    pdf_files = get_pdf_files()

    docs: list[Document] = []
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}")
        docs.extend(load_pdf_file(str(pdf_file)))
    return docs


def index_data(docs: list[Document]):
    indexing_stats = index(
        docs_source=docs,
        record_manager=record_manager,
        vector_store=vectorstore,
        batch_size=100,
        cleanup="full",
        source_id_key="source",
    )

    return indexing_stats


def main():
    docs = load_data()
    indexing_stats = index_data(docs)
    print(indexing_stats)


if __name__ == "__main__":
    main()
