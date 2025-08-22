# import asyncio
# import os
# import sys
# from dotenv import load_dotenv

# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# from browser_use import Agent, BrowserSession
# from langchain_openai import ChatOpenAI



# # Load environment variables if using .env file
# load_dotenv(override=True)

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# async def run_browser_task(task: str) -> str:
#     browser_session = BrowserSession(headless=True)

#     agent = Agent(
#         task=task,
#         llm=ChatOpenAI(model="gpt-4.1-mini"),
#         max_failures=5,
#         browser_session=browser_session,
#     )

#     try:
#         response = await agent.run()
#         return response
#     except Exception as e:
#         return f"Browser task failed: {e}"


# def run_task(task: str) -> str:
#     return asyncio.run(run_browser_task(task))


# # Example usage
# if __name__ == "__main__":
#     task = """ 
#     go to amazon and searh for iphone and get cheapest phone.
#     """

#     result = run_task(task)
#     print("Agent Response:", result)




# webcrawler.py
import asyncio
import os
import sys
from dotenv import load_dotenv
import nest_asyncio

# Windows: set ProactorEventLoop for subprocess support
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

nest_asyncio.apply()

from browser_use import Agent, BrowserSession
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


async def run_browser_task(task: str) -> str:
    """
    Fully async version of browser task.
    Safe to call from other async functions (like your multi-tool handler).
    """
    try:
        async with BrowserSession(headless=True) as browser_session:
            agent = Agent(
                task=task,
                llm=ChatOpenAI(model="gpt-4.1-mini"),
                max_failures=5,
                browser_session=browser_session,
            )
            # Debug logging
            print("[Info] Starting browser agent task...")
            response = await agent.run()
            print("[Info] Browser agent task finished.")
            return response
    except Exception as e:
        return f"[Error] Browser task failed: {e}"
