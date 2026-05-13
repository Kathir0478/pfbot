import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

log = logging.getLogger("pfbot")
_log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
log.setLevel(_log_level)
if not logging.getLogger().handlers and not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    log.addHandler(_h)


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        log.info(
            "http %s %s -> status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            getattr(response, "status_code", "?"),
            ms,
        )
        return response


app = FastAPI()

app.add_middleware(AccessLogMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load fixed JSON at startup ---
vectorstore = None
qa_chain = None
startup_failure: str | None = None
JSON_FILE = os.path.join(BASE_DIR, "profile.json")
_embed_override = os.getenv("GOOGLE_EMBEDDING_MODEL")
# langchain-google-genai 2.x uses Generative Language v1beta embedContent; legacy
# text-embedding / embedding-001 ids often 404 there. See package example: gemini-embedding-001.
DEFAULT_EMBED_MODELS = [
    "gemini-embedding-001",
]
EMBED_MODEL_CANDIDATES = (
    [_embed_override] if _embed_override else DEFAULT_EMBED_MODELS
)
GROQ_CHAT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

try:
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not set (check .env next to app.py or the environment).")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set (check .env next to app.py or the environment).")

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Flatten JSON into text
    json_text = json.dumps(data, indent=2)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([json_text])

    # Embeddings: try API-supported model ids (override list with GOOGLE_EMBEDDING_MODEL)
    embedding = None
    vectorstore = None
    embed_model_used = None
    last_embed_error: Exception | None = None
    gkey = os.getenv("GOOGLE_API_KEY")
    for model_name in EMBED_MODEL_CANDIDATES:
        try:
            embedding = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=gkey,
            )
            vectorstore = FAISS.from_documents(chunks, embedding)
            embed_model_used = model_name
            break
        except Exception as ex:
            last_embed_error = ex
            log.warning("embedding init failed for model=%s: %s", model_name, ex)
    if vectorstore is None or embedding is None:
        raise last_embed_error or RuntimeError("No embedding model succeeded.")

    # LLM using Groq
    llm = ChatGroq(
        model=GROQ_CHAT_MODEL,
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        verbose=True
    )
    log.info(
        "startup state=ready vectorstore=ok qa_chain=ok embedding_model=%s groq_model=%s",
        embed_model_used,
        GROQ_CHAT_MODEL,
    )

except Exception as e:
    vectorstore = None
    qa_chain = None
    startup_failure = str(e)
    log.exception("startup state=rejected reason=init_failure: %s", e)

TEMPLATE = """
You are a knowledgeable assistant representing Kathiravan's professional portfolio. 
Answer questions clearly, concisely, and in Kathiravan's own voice, using ONLY the information available in the portfolio JSON provided. 

If the user asks about details that are not in the portfolio, respond politely and suggest contacting Kathiravan directly for more information.

Maintain a friendly, professional tone and provide helpful explanations when possible.

User's question:
{question}

Your response:
"""

prompt = PromptTemplate.from_template(TEMPLATE)


# Request model
class ChatRequest(BaseModel):
    question: str


# --- Chat Endpoint ---
@app.get("/")
def check():
    body = {"response": "Working"}
    log.info("health input=GET / output=%s status_code=200", body)
    return body


@app.post("/chat")
def chat(request: ChatRequest):
    global qa_chain, vectorstore
    q = request.question.strip()
    log.info("chat input question=%r", q)

    if qa_chain is None or vectorstore is None:
        detail = "Vectorstore not initialized."
        if startup_failure:
            detail = f"{detail} Cause: {startup_failure[:800]}"
        log.warning(
            "chat rejected state=vectorstore_uninitialized status_code=503 detail=%s",
            detail[:500],
        )
        raise HTTPException(status_code=503, detail=detail)

    try:
        formatted_prompt = prompt.format(question=request.question)
        result = qa_chain({"query": formatted_prompt})
        text = result["result"]
        log.info(
            "chat output status_code=200 response_len=%d preview=%r",
            len(text),
            text[:500] + ("..." if len(text) > 500 else ""),
        )
        return {"response": text}

    except Exception as e:
        log.exception("chat rejected state=handler_error status_code=500 error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)