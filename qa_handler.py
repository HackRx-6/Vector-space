import faiss
import numpy as np
import asyncio
import json
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Any
from dotenv import load_dotenv
from config import config
from webcrawler import run_browser_task
from get_request_tool import get_request_tool
import re
import os
import requests
import base64
from pathlib import Path
load_dotenv()

ASYNC_CLIENT_EMBEDDING = AsyncOpenAI()
client = OpenAI()

ASYNC_CLIENT = AsyncOpenAI(
    base_url="https://register.hackrx.in/llm/openai",
    api_key="sk-spgw-api01-dd5fc1dcbd8612f84063a902a79bb3c5"
)

client = OpenAI(
    base_url="https://register.hackrx.in/llm/openai",
    api_key="sk-spgw-api01-dd5fc1dcbd8612f84063a902a79bb3c5"
)


FAILED_ANSWER_PHRASES = [
    "The answer is not available in the document.",
    "The answer is not available in the document"
]


# Enhanced tool configuration - now includes file saving and git push
TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "web_crawller",
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

GET_TOOL_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "make_get_request",
            "description": "Make HTTP GET requests to APIs and return JSON responses. Use when you need real-time data or external information not available in the document context. You can call this multiple times for different APIs or endpoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The complete URL to make the GET request to"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional headers to include in the request",
                        "additionalProperties": {"type": "string"}
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional query parameters to append to the URL",
                        "additionalProperties": {"type": "string"}
                    },
                    "auth_token": {
                        "type": "string",
                        "description": "Optional Bearer token for authentication"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

TOOLS_CONFIG_QUERY = [
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "Save content to a local file",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to save to file"
                    },
                    "filename": {
                        "type": "string", 
                        "description": "Name of the file to save"
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to save file in (default: saved_files)"
                    }
                },
                "required": ["content", "filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "push_to_github",
            "description": "Push file content to GitHub repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "File content to push"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Name of the file"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message (optional, defaults to 'Update file')"
                    },
                },
                "required": ["content", "filename"]
            }
        }
    }
]

async def save_file_tool(content: str, filename: str, directory: str = "saved_files") -> dict:
    try:
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        file_path = save_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"File saved: {filename}",
            "file_path": str(file_path)
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file: {str(e)}"}

async def git_push_tool(content: str, filename: str, 
                       repo_owner: str = None, repo_name: str = None, 
                       commit_message: str = "Update file", github_token: str = None) -> dict:
    try:
        token = github_token or os.getenv('GITHUB_TOKEN')
        owner = repo_owner or os.getenv('DEFAULT_REPO_OWNER')
        repo = repo_name or os.getenv('DEFAULT_REPO_NAME')
        
        if not token or not owner or not repo:
            return {
                "status": "error", 
                "message": "Missing required: GitHub token, repo owner, or repo name"
            }        
        
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filename}"
        
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        existing = requests.get(api_url, headers=headers)
        data = {"message": commit_message, "content": encoded_content}
        
        if existing.status_code == 200:
            data["sha"] = existing.json()["sha"]
            action = "updated"
        else:
            action = "created"
        
        response = requests.put(api_url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            return {
                "status": "success", 
                "message": f"File {action}: {filename}",
                "url": response.json().get("content", {}).get("html_url", "")
            }
        else:
            return {"status": "error", "message": f"GitHub API error: {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Push failed: {str(e)}"}


async def handle_tool_calls(tool_calls) -> List[Dict[str, Any]]:
    tool_results = []
    print(f"DEBUG: Processing {len(tool_calls)} tool calls")

    for i, tool_call in enumerate(tool_calls, 1):
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("arguments", {})
        tool_id = tool_call.get("id", f"call_{i}")

        print(f"DEBUG: Executing tool {i}/{len(tool_calls)}: {tool_name}")

        if tool_name == "web_crawller":
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                prompt = tool_args.get("prompt", "")
                if not prompt:
                    raise ValueError("No prompt provided for web_crawller")

                print(f"DEBUG: Running browser task with prompt: {prompt}")
                result = await run_browser_task(task=prompt)

                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": str(result)
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

        elif tool_name == "save_file":
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                result = await save_file_tool(
                    content=tool_args.get("content", ""),
                    filename=tool_args.get("filename", "untitled.txt"),
                    directory=tool_args.get("directory", "saved_files")
                )
                
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps(result)
                })
                
                print(f"DEBUG: File save tool {i} completed")
                
            except Exception as e:
                print(f"DEBUG: File save tool {i} failed with error: {e}")
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps({
                        "success": False,
                        "error": f"File save failed: {str(e)}",
                        "error_type": "tool_error"
                    })
                })

        elif tool_name == "push_to_github":
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                result = await git_push_tool(
                    content=tool_args.get("content", ""),
                    filename=tool_args.get("filename", ""),
                    repo_owner=tool_args.get("repo_owner"),           # ✅ None if not provided
                    repo_name=tool_args.get("repo_name"),             # ✅ None if not provided
                    commit_message=tool_args.get("commit_message", "Update file"),
                    github_token=tool_args.get("github_token")
                )
                
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps(result)
                })
        
                print(f"DEBUG: GitHub push tool {i} completed")
        
            except Exception as e:
                print(f"DEBUG: GitHub push tool {i} failed with error: {e}")
                tool_results.append({
                    "tool_call_id": tool_id,
                    "content": json.dumps({
                        "success": False,
                        "error": f"GitHub push failed: {str(e)}",
                        "error_type": "tool_error"
                    })
                })
        elif tool_name == "make_get_request":
            try:
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                
                print(f"DEBUG: Making GET request to: {tool_args.get('url', 'NO_URL')}")
                
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
                
                print(f"DEBUG: Tool {i} completed successfully, result: {json.dumps(result, indent=2)}")
                
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
                    "content": str(result)
                })


            tool_round += 1


        else:
            # No tool calls, finish
            print(f"DEBUG: No tool calls requested, finishing after {tool_round} rounds")
            return current_answer


    print(f"WARNING: Reached maximum tool rounds ({max_tool_rounds})")
    return current_answer

async def execute_with_multi_tool_support_document(
    messages: List[Dict[str, Any]], 
    max_tool_rounds: int = 5
) -> str:
   
    current_messages = messages.copy()
    tool_round = 0
    
    while tool_round < max_tool_rounds:
        print(f"DEBUG: Tool round {tool_round + 1}/{max_tool_rounds}")
        
        # Make API call with tools available
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.qa_model,
            messages=current_messages,
            temperature=1.0,
            tools=GET_TOOL_CONFIG  # ALWAYS include tools
        )
        
        message = response.choices[0].message
        current_answer = message.content.strip() if message.content else ""
        
        print(f"DEBUG: LLM response: {current_answer[:200]}...")
        
        # Add assistant's response to conversation
        assistant_message = {
            "role": "assistant", 
            "content": current_answer
        }
        
        # If there are tool calls, add them to the message
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
            
            # Execute all tool calls
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
            
            # Continue the loop - LLM can now make more tool calls if needed
            
        else:
            # No tool calls - we're done
            print(f"DEBUG: No tool calls requested, finishing after {tool_round} rounds")
            return current_answer
    
    # Reached max tool rounds
    print(f"WARNING: Reached maximum tool rounds ({max_tool_rounds})")
    return current_answer

async def execute_with_multi_tool_query(
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
            tools=TOOLS_CONFIG_QUERY
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
                    "content": str(result)
                })


            tool_round += 1


        else:
            # No tool calls, finish
            print(f"DEBUG: No tool calls requested, finishing after {tool_round} rounds")
            return current_answer


    print(f"WARNING: Reached maximum tool rounds ({max_tool_rounds})")
    return current_answer


# UPDATED SYSTEM PROMPT - Emphasizes multi-tool capability with prompt-based browser automation
DOCUMENT_SYSTEM_PROMPT = """You are an expert QA system. Your task is to answer user queries by extracting the most relevant, concise, and precise information from a provided chunk of a document or from tool call responses. Answer user queries based on the context provided and any data received from get api function. Do not rely on prior knowledge or assumptions. Follow the instructions given in document one after the other for tool calls. If you make tool calls, base your answer primarily on the response data received. If the answer to the question is not clearly present in the given chunks or tool responses, respond with "The answer is not available in the document."

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


MULTI_TOOL_SYSTEM_PROMPT = """You are an agent that has access to browser automation tools. Use the tool by providing it with a detailed prompt like which link to procced with and then step by step that includes exact steps to follow and what to extract, then give the answer accordingly. Be specific about what actions the browser should take and what information should be returned. the retyne type from tool can be any thing extract detain fromm there. 
For file operations: web_crawller is the tool to search and crawl web"""

QUERY_SYSTEM_PROMPT = """You are an agent that obeys user instructions, that has access to automation tools to procced with.
Available Tools: You can save files locally using save_file tool and git_push_tool to push files using save_file tool to GitHub repositories without extra details.
Tool Instruction: Repo name and owner is already set make sure to push the code"""

async def get_answer_for_question(question: str, index: faiss.Index, chunks: List[str]) -> str:
    try:
        # Step 1 — Generate embedding
        embedding_response = await ASYNC_CLIENT_EMBEDDING.embeddings.create(
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
                raise ValueError(f"FAISS search returned unexpected result: {type(search_result)} with length {len(search_result) if hasattr(search_result, 'len') else 'N/A'}")


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
                {"role": "system", "content": DOCUMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]


            # Step 5 — Execute with tools
            current_answer = await execute_with_multi_tool_support_document(
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


async def get_answer_for_url(question: str, chunks: List[str]) -> str:
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
    
async def get_answer_for_question_query(question: str, chunks: List[str]) -> str:
    try:
        context = "\n---\n".join(chunks)


        user_prompt = f"{question}\n{context}"
        
        initial_messages = [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Execute with multi-tool support
        current_answer = await execute_with_multi_tool_query(
            initial_messages,
            max_tool_rounds=1
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

async def get_answer_for_question_without_embedding(question: str, chunks: List[str]) -> str:
    try:
        context = "\n---\n".join(chunks)

        user_prompt = f"{question}\n{context}"
        
        initial_messages = [
            {"role": "system", "content": DOCUMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Execute with multi-tool support
        current_answer = await execute_with_multi_tool_support_document(
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

async def generate_answers_for_url(questions: List[str], chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_url(q, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def generate_answers_in_without_embedding(questions: List[str], chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_question_without_embedding(q, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def generate_answers_in_query(questions: List[str], chunks: List[str]) -> List[str]:
    tasks = [get_answer_for_question_query(q, chunks) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)
    