import base64
from io import BytesIO
import requests
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
# Function to download and encode image as base64
def encode_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return base64.b64encode(image_data.read()).decode('utf-8')
    else:
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

# System prompt
SYSTEM_PROMPT = """You are an expert Q&A system. Your task is to answer user queries by extracting the most relevant, concise, and precise information from a provided chunk of a document (the "similarity chunk").
Answer user queries based only on the context provided in the current chunk. Do not rely on prior knowledge or assumptions. If the answer to the question is not clearly present in the given chunk, respond with: "The answer is not available in the document."
### Inputs
For every input, you will receive:
- The user's question.
- A specific chunk from the document most relevant to the question (the "similarity chunk").
Additional chunks may be provided later. However, for each question, respond only based on the currently provided chunk.
### Output Format
- A single, concise sentence or two directly answering the question, ending with a brief traceability clause in brackets.
- OR, if the answer is not in the context, the exact phrase: "The answer is not available in the document."
### Important Reminders
- Use only the information from the current similarity chunk.
- If the answer is not in the chunk, respond only with: "The answer is not available in the document."
- Do not include reasoning or explanation—just the answer and traceability.
- Be as concise and precise as possible.
- Always include a traceability clause at the end of the answer, unless the answer is unavailable.
"""

# Ask one question with the image and get the answer
async def ask_question(client, base64_image, question):
    inp = [
        {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
        {"type": "input_text", "text": question}
    ]
    conversation = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": inp}
    ]

    response = await client.responses.create(
        model="gpt-4.1-mini",
        input=conversation,
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )

    return response.output_text.strip()

# Async function to handle all questions in parallel
async def get_answers_async(questions, image_url):
    base64_image = encode_image_from_url(image_url)
    client = AsyncOpenAI(api_key=OPEN_AI_KEY)

    tasks = [ask_question(client, base64_image, q) for q in questions]
    answers = await asyncio.gather(*tasks)
    return answers

# Ask one question with the image and get the answer
async def ask_question_small_pdf(client, base64_pdf, question):
    inp = [
        {
            "type": "input_file",
            "file": {
                "data": base64_pdf,
                "mime_type": "application/pdf",
                "filename": "document.pdf"
                }
                },
                {"type": "input_text", "text": question}
    ]
    conversation = [
        {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": inp}
    ]

    response = await client.responses.create(
        model="gpt-4.1-mini",
        input=conversation,
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )

    return response.output_text.strip()