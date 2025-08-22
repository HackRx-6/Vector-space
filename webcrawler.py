import sys
import asyncio
import aiohttp
import os


async def run_browser_task(task: str) -> str:
    endpoint_url = "http://13.71.109.246:7860/webhook"
    
    async with aiohttp.ClientSession() as session:
        payload = {"prompt": task}
        async with session.post(endpoint_url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                # Correct key returned by your Flask webhook
                return result.get("output", "")
            else:
                raise Exception(f"External service failed with status: {response.status}")