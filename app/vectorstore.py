import weaviate  # type: ignore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.weaviate import Weaviate

from app.config import get_settings


def get_vectorstore():
    settings = get_settings()

    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_organization=settings.OPENAI_API_ORGANIZATION_ID,
    )

    weaviate_client = weaviate.Client(
        url=settings.WEAVIATE_URL,
    )

    vectorstore = Weaviate(
        client=weaviate_client,
        index_name=settings.INDEX_NAME,
        text_key="text",
        by_text=False,
        embedding=embeddings,
        attributes=["source", "page", "total_pages"],
    )

    return vectorstore


def get_retriever():
    return get_vectorstore().as_retriever()
