import httpx
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        keep_alive: str = "10m",
        default_num_ctx: int = 8192,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.default_num_ctx = default_num_ctx

    async def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        num_ctx: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx or self.default_num_ctx,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
