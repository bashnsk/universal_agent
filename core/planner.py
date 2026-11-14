import json
from agents.base import OllamaClient


class TaskPlanner:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    async def decompose(self, main_task: str, context: str = "") -> list:
        system = 'Ты Senior Tech Lead. Разбей главную задачу на независимые атомарные подзадачи. Верни результат СТРОГО в формате JSON-массива строк: ["подзадача 1", "подзадача 2"]'
        prompt = f"Главная задача: {main_task}\nКонтекст прошлых попыток: {context}"

        response = await self.client.generate(
            self.model, prompt, system_prompt=system, temperature=0.3
        )
        try:
            # Очистка от markdown-оберток если есть
            cleaned = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception:
            # Fallback декомпозиция
            return [main_task]
