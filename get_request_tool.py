import httpx
import json
from typing import Dict, Any, Optional

class GetRequestTool:
    
    def __init__(self):
        self.tool_name = "get_request"
        self.tool_description = "Makes a GET request to a specified URL and returns the response, including full HTML content."
        self.tool_parameters = [
            {"name": "url", "type": "string", "description": "The URL to make the GET request to."}
        ]

    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self.execute_get_request(url=kwargs.get("url"))

    @staticmethod
    async def execute_get_request(
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        # Prepare request headers
        request_headers = headers or {}
        if auth_token:
            request_headers["Authorization"] = f"Bearer {auth_token}"

        if "User-Agent" not in request_headers:
            request_headers["User-Agent"] = "QA-Handler-Bot/1.0"

        timeout = httpx.Timeout(30.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(
                    url,
                    headers=request_headers,
                    params=params,
                    follow_redirects=True
                )

                content_type = response.headers.get("content-type", "").lower()

                if "application/json" in content_type:
                    try:
                        data = response.json()
                        return {
                            "success": True,
                            "status_code": response.status_code,
                            "url": str(response.url),
                            "data": data,
                            "content_type": content_type
                        }
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "status_code": response.status_code,
                            "url": str(response.url),
                            "data": {"text_content": response.text},  # full text now
                            "content_type": content_type
                        }
                elif "text/" in content_type:
                    # Return full text content (HTML or plain text)
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "url": str(response.url),
                        "data": {"text_content": response.text},  # full HTML/text
                        "content_type": content_type
                    }
                else:
                    # Binary content
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "url": str(response.url),
                        "data": {
                            "message": f"Binary content received ({content_type})",
                            "size_bytes": len(response.content)
                        },
                        "content_type": content_type
                    }
            except httpx.TimeoutException:
                return {"success": False, "error": "Request timeout (30 seconds)", "error_type": "timeout"}
            except httpx.HTTPStatusError as e:
                return {"success": False, "status_code": e.response.status_code, "error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}", "error_type": "http_error"}
            except Exception as e:
                return {"success": False, "error": str(e), "error_type": "unknown"}

# Global instance for use in QA handler
get_request_tool = GetRequestTool()
