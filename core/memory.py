import json
import math
import httpx
from pathlib import Path

class AgentMemory:
    def __init__(self, storage_path: str = "/home/sid/RAID/_AI/universal_agent/skills/registry.json", base_url: str = "http://localhost:11434", embed_model: str = "nomic-embed-text:latest"):
        self.path = Path(storage_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.embed_model = embed_model
        self.registry = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return {"successful_tasks": []}
                    data = json.loads(content)
                    if "successful_tasks" not in data:
                        data["successful_tasks"] = []
                    return data
            except json.JSONDecodeError:
                return {"successful_tasks": []}
        return {"successful_tasks": []}

    async def _get_embedding(self, text: str) -> list:
        """Получает векторное представление (эмбеддинг) текста через локальную Ollama."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embed_model,
            "prompt": text
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json().get("embedding", [])
        except Exception:
            return []

    def _cosine_similarity(self, v1: list, v2: list) -> float:
        """Вычисляет косинусное сходство между двумя векторами."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(b * b for b in v2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    async def save_success(self, task_key: str, solution: str):
        """Сохраняет задачу, результат и ее эмбеддинг для семантического поиска."""
        embedding = await self._get_embedding(task_key)
        self.registry["successful_tasks"].append({
            "task": task_key,
            "solution": solution,
            "embedding": embedding
        })
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    async def find_similar(self, query: str, threshold: float = 0.75) -> str:
        """Ищет наиболее семантически похожую успешную задачу в памяти."""
        tasks = self.registry.get("successful_tasks", [])
        if not tasks:
            return ""

        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            return ""

        best_match = ""
        highest_score = 0.0

        for item in tasks:
            item_embedding = item.get("embedding", [])
            if item_embedding:
                score = self._cosine_similarity(query_embedding, item_embedding)
                if score > highest_score:
                    highest_score = score
                    best_match = item["solution"]

        if highest_score >= threshold:
            print(f"[🧠 Memory] Найдено семантически похожее решение в памяти (сходство: {highest_score:.2f})")
            return best_match

        return ""