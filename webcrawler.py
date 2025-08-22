import sys
import asyncio
import nest_asyncio
import aiohttp
import os

os.environ["BROWSER_USE_MEMORY_PATH"] = "/home/user/app/storage/.mem0"

# Apply nest_asyncio patch
nest_asyncio.apply()

# Set event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def run_browser_task(task: str) -> str:
    """
    Send task to external endpoint and receive response
    """
    endpoint_url = "http://10.64.166.200:7860/webhook"
    
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": task}
        async with session.post(endpoint_url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                # Correct key returned by your Flask webhook
                return result.get("output", "")
            else:
                raise Exception(f"External service failed with status: {response.status}")
