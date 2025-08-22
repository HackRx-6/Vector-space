import os
import tiktoken
import faiss
from openai import AsyncOpenAI
import numpy as np
import asyncio
from typing import List, Tuple
from dotenv import load_dotenv
from config import config

# Load env & pull your comma‑separated list of keys, e.g. "key1,key2,key3"
load_dotenv()
API_KEYS = os.environ["OPENAI_API_KEYS2"].split(",")

# Build one AsyncOpenAI client per key
ASYNC_CLIENTS = [AsyncOpenAI(api_key=key) for key in API_KEYS]

TOKENIZER = tiktoken.get_encoding("cl100k_base")



def chunk_text(text: str) -> List[str]:
    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + config.token_limit
        chunk_tokens = tokens[start:end]
        chunks.append(TOKENIZER.decode(chunk_tokens))
        start += config.token_limit - config.overlap
    print(f"Text split into {len(chunks)} chunks.")
    return chunks

async def create_in_memory_faiss_index(chunks: List[str]) -> Tuple[faiss.Index, List[str]]:
    print(f"Generating embeddings for {len(chunks)} text chunks using '{config.embedding_model}'...")
    
    batch_size = 350
    tasks = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(chunks), batch_size), start=0):
        batch = chunks[i : i + batch_size]
        # pick a client by round‑robin
        client = ASYNC_CLIENTS[batch_idx % len(ASYNC_CLIENTS)]
        print(f"[API CALL] Requesting embeddings for batch {batch_idx+1}/{total_batches} with {len(batch)} chunks")
        task = asyncio.create_task(
            client.embeddings.create(input=batch, model=config.embedding_model)
        )
        tasks.append(task)


    print("All batches dispatched; awaiting responses...")
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    all_embeddings = []
    for idx, resp in enumerate(responses):
        if isinstance(resp, Exception):
            print(f"Batch {idx} failed: {resp}")
            continue
        all_embeddings.extend(item.embedding for item in resp.data)

    if not all_embeddings:
        raise RuntimeError("No embeddings generated; all batches failed.")

    embeddings_np = np.array(all_embeddings, dtype="float32")
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    print(f"Created FAISS index with {index.ntotal} vectors.")

    return index, chunks
