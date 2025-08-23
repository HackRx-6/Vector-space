import os
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import re
from urllib.parse import urlparse
from config import config, AppConfig
import file_extractor
from text_chunker import chunk_text, create_in_memory_faiss_index
from qa_handler import generate_answers_in_parallel, generate_answers_in_without_embedding, generate_answers_for_url
from image_handler import get_answers_async
from language_normaliser import normalise_questions, normalise_language
import io
import httpx
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

# ------------------- Middleware to print raw POST requests -------------------
app = FastAPI()  # Temporarily create for middleware; will override later with lifespan

@app.middleware("http")
async def print_raw_request_body(request: Request, call_next):
    if request.method == "POST":
        try:
            body_bytes = await request.body()
            print("---- RAW REQUEST BODY ----")
            print(body_bytes.decode(errors="ignore"))
            print("--------------------------")
        except Exception as e:
            print(f"Failed to read request body: {e}")
    response = await call_next(request)
    return response

# ------------------- Global Async HTTP Client -------------------
global_http_client: httpx.AsyncClient = None

# ------------------- Download and Dispatcher -------------------
async def _download_file_in_parallel(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        head_resp = await client.head(url, follow_redirects=True)
        head_resp.raise_for_status()
        file_size = int(head_resp.headers.get('Content-Length', 0))
        accept_ranges = head_resp.headers.get("accept-ranges", "").lower()
        if "bytes" not in accept_ranges or file_size == 0:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content

        chunk_size = 10 * 1024 * 1024
        num_chunks = min(file_size // chunk_size + 1, 16)
        if num_chunks < 2:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content

        chunk_ranges = [(i * (file_size // num_chunks), (i + 1) * (file_size // num_chunks) - 1)
                        for i in range(num_chunks - 1)]
        chunk_ranges.append(((num_chunks - 1) * (file_size // num_chunks), file_size - 1))

        async def download_chunk(start: int, end: int):
            headers = {'Range': f'bytes={start}-{end}'}
            resp = await client.get(url, headers=headers, follow_redirects=True)
            resp.raise_for_status()
            return resp.content

        results = await asyncio.gather(*[download_chunk(s, e) for s, e in chunk_ranges], return_exceptions=True)
        if any(isinstance(r, Exception) for r in results):
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content
        return b"".join(r for r in results if isinstance(r, bytes))

    except Exception:
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except:
            return None

async def _download_file(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except:
        return None

async def get_text_from_document_url(url: str, file_ext: str, batch_size: int = 250) -> Optional[str]:
    match file_ext:
        case '.pdf':
            content_bytes = await _download_file_in_parallel(global_http_client, url)
            if not content_bytes:
                return None
            return file_extractor._process_pdf(content_bytes, batch_size)
        case '.docx':
            content_bytes = await _download_file_in_parallel(global_http_client, url)
            if not content_bytes:
                return None
            return file_extractor._process_docx(content_bytes)
        case '.eml':
            content_bytes = await _download_file_in_parallel(global_http_client, url)
            if not content_bytes:
                return None
            return file_extractor._process_eml(content_bytes)
        case '.xlsx' | '.xls' | '.xlsb':
            content_bytes = await _download_file_in_parallel(global_http_client, url)
            if not content_bytes:
                return None
            file_like = io.BytesIO(content_bytes)
            return file_extractor.process_excel(file_like)
        case '.pptx':
            content_bytes = await _download_file_in_parallel(global_http_client, url)
            if not content_bytes:
                return None
            return file_extractor.process_pptx(content_bytes)
        case '.bin':
            content_bytes = await _download_file(global_http_client, url)
            if not content_bytes:
                return None
            return file_extractor.process_bin(content_bytes)
        case '.zip':
            content_bytes = await _download_file_in_parallel(global_http_client, url)
            if not content_bytes:
                return None
            return file_extractor.process_zip_simple(content_bytes)
        case '.png' | '.jpg' | '.jpeg':
            return file_ext
        case _:
            return "Unsupported file type"
        
# ------------------- URL Checking -------------------
        
def is_valid_url(url_string):
        try:
            result = urlparse(url_string)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

# Load environment variables
load_dotenv()

# ------------------- Lifespan for Async HTTP Client -------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_http_client
    
    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=50,
        keepalive_expiry=30.0
    )
    
    global_http_client = httpx.AsyncClient(
        http2=True,
        limits=limits,
        timeout=httpx.Timeout(
            connect=30.0,
            read=600.0,
            write=30.0,
            pool=10.0
        )
    )
    yield
    await global_http_client.aclose()

# Override FastAPI app with lifespan
app = FastAPI(
    title="Parallel In-Memory Q&A API",
    description="A stateless API to answer questions about PDF documents using parallel processing and configurable settings.",
    version="4.0.0",
    lifespan=lifespan
)

# Re-add middleware to the final app
@app.middleware("http")
async def print_raw_request_body(request: Request, call_next):
    if request.method == "POST":
        try:
            body_bytes = await request.body()
            print("---- RAW REQUEST BODY ----")
            print(body_bytes.decode(errors="ignore"))
            print("--------------------------")
        except Exception as e:
            print(f"Failed to read request body: {e}")
    response = await call_next(request)
    return response

# --- API Models ---
class PolicyInquiry(BaseModel):
    documents: Optional[str] = None 
    url: Optional[str] = None   # Make url optional
    query: Optional[str] = None     # Add query field
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

security = HTTPBearer(auto_error=False)  # Do not throw error if token missing

# --- Core Endpoint ---
@app.post("/hackrx/run", response_model=QAResponse)
async def run_submission(
    inquiry: PolicyInquiry,
    token: HTTPAuthorizationCredentials = Depends(security)
):
    if token and token.credentials:
        print(f"Auth token received: {token.credentials}")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server configuration error: OPENAI_API_KEY is not set.")

    try:
        print("Incoming Inquiry:", inquiry.dict())
        inquiry.questions = await normalise_questions(inquiry.questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Language Normalisation: {e}")

    if inquiry.documents:
        try:
            path = urlparse(str(inquiry.documents)).path
        except:
            path = ""
        file_ext = os.path.splitext(path)[1].lower()
        print(file_ext)
        if not file_ext or file_ext == '.':
            print("Not Document")
        else:
            try:
                document_text = await get_text_from_document_url(str(inquiry.documents), file_ext)
                match document_text:
                    case ".png" | ".jpg" | ".jpeg":
                        found_answers = await get_answers_async(inquiry.questions, str(inquiry.url))
                        return {"answers": found_answers}
                    case "Unsupported file type":
                        raise HTTPException(status_code=500, detail="Unsupported file type")
                    case "" | None:
                        raise HTTPException(status_code=400, detail="Failed to download or extract text from the document.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during document processing: {e}")

            try:
                text_chunks = await asyncio.to_thread(chunk_text, document_text)
                if len(text_chunks) < config.top_k:
                    found_answers = await generate_answers_in_without_embedding(inquiry.questions, text_chunks)
                else:
                    faiss_index, chunks_for_qa = await create_in_memory_faiss_index(text_chunks)
                    found_answers = await generate_answers_in_parallel(inquiry.questions, faiss_index, chunks_for_qa)
                return {"answers": found_answers}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during embedding/answer generation: {e}")
    if inquiry.url:
         # Validate URL format
        if not is_valid_url(str(inquiry.url)):
            print("invalid url detected") 
        else:
            try:
                chunk = [str(inquiry.url)]
                found_answers = await generate_answers_for_url(inquiry.questions, chunk)
                return {"answers": found_answers}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during URL processing: {e}")
    if inquiry.query:
        try:
            inquiry.query = await normalise_language(str(inquiry.questions))
            text_chunks = await asyncio.to_thread(chunk_text, inquiry.query)
            found_answers = await generate_answers_in_without_embedding(inquiry.questions, text_chunks)
            return {"answers": found_answers}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during query handling: {e}")

    raise HTTPException(status_code=400, detail="Either 'url' or 'query' must be provided.")

# --- Configuration Endpoints ---
@app.get("/config", response_model=AppConfig)
async def get_config():
    return config

@app.post("/config", response_model=AppConfig)
async def update_config(new_config: Dict[str, Any]):
    updated_data = config.model_dump()
    updated_data.update(new_config)
    
    try:
        new_app_config = AppConfig(**updated_data)
        config.dict.update(new_app_config.dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {e}")
        
    return config