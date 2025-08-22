# agent_runner.py
import asyncio
from browser_use import BrowserSession, Agent
from browser_use import Agent
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os

# Create OpenAI client with custom base URL
client = OpenAI(
    base_url="https://register.hackrx.in/llm/openai",
    api_key="sk-spgw-api01-dd5fc1dcbd8612f84063a902a79bb3c5"
)

async def run_agent_with_prompt(prompt: str):
    browser_session = BrowserSession(
        headless=True,
    )
    
    # Configure ChatOpenAI to use the custom endpoint
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        base_url="https://register.hackrx.in/llm/openai",
        api_key="sk-spgw-api01-dd5fc1dcbd8612f84063a902a79bb3c5"
    )
    
    agent = Agent(
        task=prompt,
        llm=llm,
        max_failures=3,
        browser_session=browser_session,
    )
    result = await agent.run()
    return result
