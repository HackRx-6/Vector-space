import faiss
import numpy as np
import asyncio
import json
import httpx
from typing import List, Dict, Any
from dotenv import load_dotenv
from config import config
from get_request_tool import get_request_tool
import re
import os

from text_chunker import ASYNC_CLIENTS

load_dotenv()

SUBSCRIPTION_KEY = os.environ["SUBSCRIPTION_KEY"]
API_URL = "https://register.hackrx.in/llm/openai"

FAILED_ANSWER_PHRASES = [
    "The answer is not available in the document.",
    "The answer is not available in the document"
]

TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "make_get_request",
            "description": "Make HTTP GET requests to APIs and return JSON responses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    "params": {"type": "object", "additionalProperties": {"type": "string"}},
                    "auth_token": {"type": "string"}
                },
                "required": ["url"]
            }
        }
    }
]

MULTI_TOOL_SYSTEM_PROMPT = """You are an expert QA system. Your task is to answer user queries by extracting the most relevant, concise, and precise information from a provided chunk of a document or from tool call responses. Answer user queries based on the context provided and any data received from get api function. Do not rely on prior knowledge or assumptions. Follow the instructions given in document one after the other for tool calls. If you make tool calls, base your answer primarily on the response data received. If the answer to the question is not clearly present in the given chunks or tool responses, respond with "The answer is not available in the document."

Available Tools:
- make_get_request: Make HTTP GET requests to fetch real-time data from APIs or web services. You can call this tool MULTIPLE TIMES if needed.

Instructions:
- Give the most concise answer as you can.
- Carefully examine the given context to give right answers.
- Strictly answer based on context only after Get Request.
- Verify your answer twice before answering.

Output Format:
- A single, concise sentence or two directly answering the question.
- OR, if the answer is not in the context, the exact phrase "The answer is not available in the document."
- Use only the information from the provided chunks AND any API responses.
- Be as concise and precise as possible."""


# ------------------- TOOL HANDLER -------------------
async def handle_tool_calls(tool_calls) -> List[Dict[str, Any]]:
    tool_results = []
    
    for i, tool_call in enumerate(tool_calls, 1):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        tool_id = tool_call.get("id", f"call_{i}")
        
        if tool_name == "make_get_request":
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                result = await get_request_tool.execute_get_request(
                    url=tool_args.get("url", ""),
                    headers=tool_args.get("headers"),
                    params=tool_args.get("params"),
                    auth_token=tool_args.get("auth_token")
                )
                
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps(result, indent=2)
                })
            except Exception as e:
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps({
                        "success": False,
                        "error": f"Tool execution failed: {str(e)}",
                        "error_type": "tool_error"
                    })
                })
        else:
            tool_results.append({
                "tool_call_id": tool_id,
                "content": json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "error_type": "unknown_tool"
                })
            })
    
    return tool_results


# ------------------- CALL HACKRX LLM -------------------
async def call_hackrx_llm(messages: List[Dict[str, str]], model: str = "gpt-5") -> Dict[str, Any]:
    payload = {"model": model, "messages": messages}
    headers = {"Content-Type": "application/json", "x-subscription-key": SUBSCRIPTION_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


# ------------------- MULTI-TOOL EXECUTION -------------------
async def execute_with_multi_tool_support(
    messages: List[Dict[str, Any]], 
    max_tool_rounds: int = 5
) -> str:
    current_messages = messages.copy()
    tool_round = 0

    while tool_round < max_tool_rounds:
        response_data = await call_hackrx_llm(current_messages, model=config.qa_model)
        choices = response_data.get("choices", [])
        if not choices:
            return "LLM returned no choices"

        message = choices[0].get("message", {})
        current_answer = message.get("content", "").strip()
        assistant_message = {"role": "assistant", "content": current_answer}

        # Process tool calls if any
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tool_results = await handle_tool_calls([
                {
                    "id": tc.get("id"),
                    "name": tc.get("function", {}).get("name"),
                    "arguments": tc.get("function", {}).get("arguments", {})
                }
                for tc in tool_calls
            ])

            for result in tool_results:
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"]
                })

            current_messages.append(assistant_message)
            tool_round += 1
        else:
            return current_answer

    return current_answer


# ------------------- GET ANSWER FOR QUESTION -------------------
async def get_answer_for_question(question: str, index: faiss.Index, chunks: List[str]) -> str:
    last_answer = ""
    try:
        # Generate embedding
        embedding_response = await ASYNC_CLIENTS.embeddings.create(
            input=question,
            model=config.embedding_model
        )
        embedding = np.array(embedding_response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        return f"Embedding Error: {e}"

    for attempt in range(config.MAX_RETRIES):
        try:
            if not hasattr(index, "search"):
                raise TypeError(f"Provided index is not a FAISS index: {type(index)}")

            search_result = await asyncio.to_thread(index.search, embedding, (attempt + 1) * config.top_k + 1)
            distances, all_indices = search_result
            start_index = attempt * config.top_k
            end_index = start_index + config.top_k
            current_indices = all_indices[0][start_index:end_index]

            if len(current_indices) == 0:
                break

            selected_chunks = [chunks[i] for i in current_indices if 0 <= i < len(chunks)]
            context = "---\n".join(selected_chunks)

            user_prompt = (
                f"{question}\n\n{context}" if attempt == 0 else
                f"Retry Attempt: {attempt}. Additional chunks provided.\nQuestion: {question}\nContext:\n{context}"
            )

            initial_messages = [
                {"role": "system", "content": MULTI_TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            current_answer = await execute_with_multi_tool_support(
                initial_messages, max_tool_rounds=3
            )
            last_answer = current_answer
            if current_answer not in FAILED_ANSWER_PHRASES:
                return current_answer

        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY_SECONDS)
            else:
                return f"LLM Error: {e}"

    return last_answer


# ------------------- GET ANSWER WITHOUT EMBEDDING -------------------
async def get_answer_for_question_without_embedding(question: str, chunks: List[str]) -> str:
    context = "\n---\n".join(chunks)
    user_prompt = f"{question}\n{context}"

    initial_messages = [
        {"role": "system", "content": MULTI_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    current_answer = await execute_with_multi_tool_support(
        initial_messages, max_tool_rounds=3
    )
    return current_answer


# ------------------- PARALLEL EXECUTION -------------------
async def generate_answers_in_parallel(questions: List[str], index: faiss.Index, chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_question(q, index, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def generate_answers_in_without_embedding(questions: List[str], chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_question_without_embedding(q, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)
