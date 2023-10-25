import json
from contextlib import asynccontextmanager
from functools import partial
from operator import itemgetter
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.chat_models import ChatOpenAI
from langchain.load import dumps, loads
from langchain.output_parsers import MarkdownListOutputParser, NumberedListOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import BaseMemory, BaseRetriever, Document, StrOutputParser
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.runnable import RunnableLambda, RunnableMap
from pydantic import BaseModel

from app.config import get_settings
from app.memory import get_memory
from app.vectorstore import get_retriever

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatFollowUpRequest(BaseModel):
    session_id: str


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def reciprocal_rank_fusion(
    results: list[list[Document]], k: int = 60
) -> list[tuple[Document, float]]:
    fused_scores: dict[str, float] = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results: list[tuple[Document, float]] = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def get_top_k_from_reranked(
    results: list[tuple[Document, float]], k: int = 5
) -> list[Document]:
    return [doc for doc, _ in results[:k]]


def format_docs(docs: list[Document]) -> str:
    formatted_docs: list[str] = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


GENERATE_QUERIES_TEMPLATE = """\
Generate a diverse set of search queries related to a follow-up question. \
Each query should be self-contained and separated by a new line. \
Output the queries as a markdown list. \
Limit the number of queries to a maximum of 4.

Chat History:
{chat_history}
Follow Up Input: {question}
Generated Queries:"""

GENERATE_QUERIES_PROMPT = PromptTemplate.from_template(
    GENERATE_QUERIES_TEMPLATE,
)

RESPONSE_TEMPLATE = """\
You are an expert in legal studies, equipped with answering any questions about jurisprudence.

Generate a comprehensive and informative answer of 80 words or less for the given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities with the same name, write separate answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.

Anything between the following context html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

<context>
    {context} 
</context>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
"""

RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

GENERATE_FOLLOW_UPS_TEMPLATE = """\
Generate 5 follow-up questions that are relevant to the given conversation. \
Each question should be self-contained and separated by a new line. \
Output the questions as a numbered markdown list.

Chat History:
```
{chat_history}
```
Generated Follow Up Questions:"""

GENERATE_FOLLOW_UPS_PROMPT = PromptTemplate.from_template(
    GENERATE_FOLLOW_UPS_TEMPLATE,
)


def create_retriever_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
    rank_fusion_k: int = 60,
    k: int = 5,
):
    generate_queries_chain = (
        GENERATE_QUERIES_PROMPT | llm | MarkdownListOutputParser()
    ).with_config(run_name="GenerateQueries")

    reciprocal_rank_fusion_lambda = RunnableLambda(
        partial(reciprocal_rank_fusion, k=rank_fusion_k)
    ).with_config(run_name="ReciprocalRankFusion")

    get_top_lambda = RunnableLambda(partial(get_top_k_from_reranked, k=k)).with_config(
        run_name="GetTopK"
    )

    return (
        generate_queries_chain
        | retriever.map()
        | reciprocal_rank_fusion_lambda
        | get_top_lambda
    ).with_config(run_name="FindRelevantDocuments")


def get_answer_chain(memory: BaseMemory, llm: BaseLanguageModel):
    retriever_chain = create_retriever_chain(
        llm=llm,
        retriever=get_retriever(),
    )

    _context = RunnableMap(
        {
            "context": (
                retriever_chain
                | RunnableLambda(format_docs).with_config(run_name="FormatDocs")
            ).with_config(run_name="GetContext"),
            "question": RunnableLambda(itemgetter("question")).with_config(
                run_name="GetQuestion"
            ),
            "chat_history": RunnableLambda(itemgetter("chat_history")).with_config(
                run_name="GetChatHistory"
            ),
        }
    ).with_config(run_name="RetrieveContext")

    response_synthesizer = (RESPONSE_PROMPT | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse"
    )

    answer_chain_with_context = (
        RunnableMap(
            {
                "question": RunnableLambda(itemgetter("question")).with_config(
                    run_name="GetQuestion"
                ),
                "chat_history": (
                    RunnableLambda(memory.load_memory_variables).with_config(
                        run_name="LoadMemory"
                    )
                    | RunnableLambda(itemgetter("chat_history")).with_config(
                        run_name="GetChatHistory"
                    )
                ).with_config(run_name="GetChatHistoryFromMemory"),
            }
        ).with_config(run_name="GetQuestionAndChatHistory")
        | _context
        | response_synthesizer
    ).with_config(run_name="AnswerQuestion")

    return answer_chain_with_context


@app.post("/chat")
def chat(request: ChatRequest):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-16k",
    )
    memory = get_memory(llm=llm, session_id=request.session_id)
    answer_chain_with_context = get_answer_chain(llm=llm, memory=memory)
    inputs = {"question": request.message}
    response = answer_chain_with_context.invoke(input=inputs)
    memory.save_context(inputs=inputs, outputs={"output": response})
    return response


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-16k",
    )
    memory = get_memory(llm=llm, session_id=request.session_id)
    answer_chain_with_context = get_answer_chain(llm=llm, memory=memory)
    inputs = {"question": request.message}

    def stream_response():
        response = ""
        for token in answer_chain_with_context.stream(input=inputs):
            yield token
            response += token
        memory.save_context(inputs=inputs, outputs={"output": response})

    return StreamingResponse(stream_response(), media_type="text/event-stream")


async def transform_stream_for_client(
    stream: AsyncIterator[RunLogPatch],
    memory: BaseMemory,
) -> AsyncIterator[str]:
    inputs_and_outputs: dict[str, dict[str, str]] = {}
    async for chunk in stream:
        for c in chunk.ops:
            if c.get("path") in (
                "/logs/GenerateQueries/final_output",
                "/logs/FindRelevantDocuments/final_output",
                "/logs/ChatOpenAI:2/streamed_output_str/-",
            ):
                yield f"{json.dumps(jsonable_encoder(chunk))}\n"

            if c.get("path") == "/logs/GetQuestion/final_output":
                print(c)
                inputs_and_outputs["input"] = {
                    "question": c.get("value", {}).get("output")
                }
            if c.get("path") == "/final_output":
                print(c)
                inputs_and_outputs["output"] = {
                    "output": c.get("value", {}).get("output")
                }

    if (
        inputs_and_outputs.get("input", None) is not None
        and inputs_and_outputs.get("output", None) is not None
    ):
        memory.save_context(
            inputs=inputs_and_outputs["input"],
            outputs=inputs_and_outputs["output"],
        )


@app.post("/chat/stream-events")
async def chat_stream_events(request: ChatRequest):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-16k",
    )
    memory = get_memory(llm=llm, session_id=request.session_id)
    answer_chain_with_context = get_answer_chain(llm=llm, memory=memory)
    inputs = {"question": request.message}

    stream = answer_chain_with_context.astream_log(input=inputs)

    return StreamingResponse(
        transform_stream_for_client(stream, memory=memory),
        media_type="text/event-stream",
    )


@app.post("/chat/follow-up-suggestions")
async def chat_follow_up_suggestions(request: ChatFollowUpRequest):
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-16k",
    )
    memory = get_memory(
        llm=llm,
        session_id=request.session_id,
        return_messages=False,
    )

    generate_follow_ups_chain = (
        GENERATE_FOLLOW_UPS_PROMPT | llm | NumberedListOutputParser()
    ).with_config(run_name="GenerateFollowUps")

    response = generate_follow_ups_chain.invoke(input={"chat_history": memory.buffer})

    return {
        "follow_ups": response,
    }


@app.get("/")
def read_root():
    return {"Hello": "World"}
