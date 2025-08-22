import faiss
import numpy as np
import asyncio
import json
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
from config import config
from webcrawler import run_browser_task
import re
load_dotenv()


ASYNC_CLIENT = AsyncOpenAI()
client = OpenAI()

FAILED_ANSWER_PHRASES = [
    "The answer is not available in the document.",
    "The answer is not available in the document"
]

# Enhanced tool configuration - extracted as constant
TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "make_get_request",  # This matches the tool handler
            "description": "Crawl the web and return HTML content based on the given steps or URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL or instructions for the web crawler to follow"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

async def handle_tool_calls(tool_calls) -> List[Dict[str, Any]]:
    tool_results = []

    print(f"DEBUG: Processing {len(tool_calls)} tool calls")

    for i, tool_call in enumerate(tool_calls, 1):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        tool_id = tool_call.get("id", f"call_{i}")

        print(f"DEBUG: Executing tool {i}/{len(tool_calls)}: {tool_name}")

        if tool_name == "make_get_request":
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                url = tool_args.get("url", "")
                if not url:
                    raise ValueError("No URL provided for make_get_request")

                print(f"DEBUG: Running crawler for URL/task: {url}")
                
                # FIX: Await directly
                result = await run_browser_task(task=url)

                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps(result, indent=2)
                })

                print(f"DEBUG: Tool {i} completed successfully")

            except Exception as e:
                print(f"DEBUG: Tool {i} failed with error: {e}")
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps({
                        "success": False,
                        "error": f"Tool execution failed: {str(e)}",
                        "error_type": "tool_error"
                    })
                })

    return tool_results

async def execute_with_multi_tool_support(
    messages: list,
    max_tool_rounds: int = 5
) -> str:
    current_messages = messages.copy()
    tool_round = 0

    while tool_round < max_tool_rounds:
        print(f"DEBUG: Tool round {tool_round + 1}/{max_tool_rounds}")

        # Use AsyncOpenAI directly
        response = await ASYNC_CLIENT.chat.completions.create(
            model=config.qa_model,
            messages=current_messages,
            temperature=1.0,
            tools=TOOLS_CONFIG
        )

        message = response.choices[0].message
        current_answer = message.content.strip() if message.content else ""

        print(f"DEBUG: LLM response: {current_answer[:200]}...")

        # Add assistant's response to conversation
        assistant_message = {"role": "assistant", "content": current_answer}

        # Handle tool calls
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]

            current_messages.append(assistant_message)

            print(f"DEBUG: LLM wants to make {len(message.tool_calls)} tool calls")

            # Execute all tool calls async
            tool_results = await handle_tool_calls([
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
                for tool_call in message.tool_calls
            ])

            # Add tool results to conversation
            for result in tool_results:
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"]
                })

            tool_round += 1

        else:
            # No tool calls, finish
            print(f"DEBUG: No tool calls requested, finishing after {tool_round} rounds")
            return current_answer

    print(f"WARNING: Reached maximum tool rounds ({max_tool_rounds})")
    return current_answer

# UPDATED SYSTEM PROMPT - Emphasizes multi-tool capability
MULTI_TOOL_SYSTEM_PROMPT = """You are an expert QA system. Your task is to answer user queries by extracting the most relevant, concise, and precise information from a provided chunk of a document or from tool call responses. Answer user queries based on the context provided and any data received from get api function. Do not rely on prior knowledge or assumptions. Follow the instructions given in document one after the other for tool calls. If you make tool calls, base your answer primarily on the response data received. If the answer to the question is not clearly present in the given chunks or tool responses, respond with "The answer is not available in the document."

Available Tools:
- make_get_request: Make HTTP GET requests to fetch html content of the url. you use can multiple times.
Instructions:
- Extract information if tool used and take appropiate steps and actions.
- Give the most concise answer as you can.
- Carefully examine the given context to give right answers.
- Strictly answer based on context only after Get Request.
- Verify your answer twice before answering.

Output Format:
- A single, concise sentence or two directly answering the question.
- OR, if the answer is not in the context, the exact phrase "The answer is not available in the document."
- Use only the information from the provided chunks AND any API responses.
- Be as concise and precise as possible."""


async def get_answer_for_question(question: str, index: faiss.Index, chunks: List[str]) -> str:
    try:
        # Step 1 — Generate embedding
        embedding_response = await ASYNC_CLIENT.embeddings.create(
            input=question,
            model=config.embedding_model,
        )
        embedding = np.array(
            embedding_response.data[0].embedding,
            dtype=np.float32
        ).reshape(1, -1)  # Ensure correct shape for FAISS
    except Exception as e:
        print(f"DEBUG: ERROR during embedding API call: {e}")
        return f"Embedding Error: {e}"

    last_answer = ""
    for attempt in range(config.MAX_RETRIES):
        try:
            # Step 2 — Ensure FAISS object is valid
            if not hasattr(index, "search"):
                raise TypeError(f"Provided index object is not a FAISS index: {type(index)}")

            # Step 3 — Perform search safely
            search_result = await asyncio.to_thread(index.search, embedding, (attempt + 1) * config.top_k + 1)

            if not isinstance(search_result, tuple) or len(search_result) != 2:
                raise ValueError(f"FAISS search returned unexpected result: {type(search_result)} with length {len(search_result) if hasattr(search_result, '_len_') else 'N/A'}")

            distances, all_indices = search_result

            start_index = attempt * config.top_k
            end_index = start_index + config.top_k
            current_indices = all_indices[0][start_index:end_index]

            if len(current_indices) == 0:
                break

            selected_chunks = [chunks[i] for i in current_indices if 0 <= i < len(chunks)]
            context = "---\n".join(selected_chunks)

            if attempt > 0:
                print(f"INFO: Retrying with additional chunks ({config.top_k} more)")

            # Step 4 — Prepare prompt
            user_prompt = (
                f"{question}\n\n{context}" if attempt == 0 else
                f"Retry Attempt: {attempt}. Additional chunks provided.\n"
                f"Question: {question}\nContext:\n{context}"
            )

            initial_messages = [
                {"role": "system", "content": MULTI_TOOL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            # Step 5 — Execute with tools
            current_answer = await execute_with_multi_tool_support(
                initial_messages,
                max_tool_rounds=3
            )

            print(f"DEBUG: Final answer after tool execution: {current_answer}")
            print("-" * 60)

            last_answer = current_answer
            if current_answer not in FAILED_ANSWER_PHRASES:
                return current_answer

        except Exception as e:
            print(f"DEBUG: ERROR during chat completion API call (Attempt {attempt + 1}): {repr(e)}")
            if attempt < config.MAX_RETRIES - 1:
                await asyncio.sleep(config.RETRY_DELAY_SECONDS)
            else:
                return f"LLM Error: {e}"

    return last_answer

async def get_answer_for_question_without_embedding(question: str, chunks: List[str]) -> str:
    try:
        context = "\n---\n".join(chunks)

        user_prompt = f"{question}\n{context}"
        
        initial_messages = [
            {"role": "system", "content": MULTI_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Execute with multi-tool support
        current_answer = await execute_with_multi_tool_support(
            initial_messages,
            max_tool_rounds=3
        )

        print(f"DEBUG: Final answer after tool execution: {current_answer}")
        print("-" * 60)
        
        if current_answer not in FAILED_ANSWER_PHRASES:
            return current_answer
        else:
            return current_answer
            
    except Exception as e:
        print(f"DEBUG: ERROR during chat completion API call: {e}")
        return f"LLM Error: {e}"



# Parallel execution functions
async def generate_answers_in_parallel(questions: List[str], index: faiss.Index, chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_question(q, index, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def generate_answers_in_without_embedding(questions: List[str], chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_question_without_embedding(q, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)







