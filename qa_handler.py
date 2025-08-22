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


# Enhanced tool configuration - updated to take prompt instead of URL
TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "make_get_request",  # This matches the tool handler
            "description": "Execute browser automation tasks based on the given prompt and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed prompt with specific steps for the browser to follow and what to extract"
                    }
                },
                "required": ["prompt"]
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

                prompt = tool_args.get("prompt", "")
                if not prompt:
                    raise ValueError("No prompt provided for make_get_request")

                print(f"DEBUG: Running browser task with prompt: {prompt}")
                
                # Execute browser task
                result = await run_browser_task(task=prompt)

                # Append the actual string result
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": result
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
    max_tool_rounds: int = 2
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
            print(tool_results)
            for result in tool_results:
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result
                })


            tool_round += 1


        else:
            # No tool calls, finish
            print(f"DEBUG: No tool calls requested, finishing after {tool_round} rounds")
            return current_answer


    print(f"WARNING: Reached maximum tool rounds ({max_tool_rounds})")
    return current_answer


# UPDATED SYSTEM PROMPT - Emphasizes multi-tool capability with prompt-based browser automation
MULTI_TOOL_SYSTEM_PROMPT = """You are an agent that has access to browser automation tools. Use the tool by providing it with a detailed prompt like which link to procced with and then step by step that includes exact steps to follow and what to extract, then give the answer accordingly. Be specific about what actions the browser should take and what information should be returned. the retyne type from tool can be any thing extract detain fromm there"""



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
