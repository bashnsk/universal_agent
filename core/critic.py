import json
import re
from agents.base import OllamaClient


class CriticAgent:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    async def evaluate(self, main_task: str, accumulated_results: str) -> dict:
        system = 'Ты строгий Code Reviewer и Архитектор. Оцени результат выполнения задачи. Верни ответ СТРОГО в формате JSON: {"status": "SUCCESS" или "RETRY", "feedback": "детальный разбор ошибок или подтверждение успеха"}'
        prompt = f"Главная задача: {main_task}\n\nПолученные результаты:\n{accumulated_results}"

        response = await self.client.generate(
            self.model, prompt, system_prompt=system, temperature=0.1
        )

        # 1. Попытка прямого парсинга (если модель строго послушалась)
        try:
            cleaned = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception:
            pass

        # 2. Устойчивый поиск JSON блока через Regex, если модель добавила лишний текст
        try:
            match = re.search(r"\{.*?\}", response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        # 3. Fallback если совсем не удалось найти JSON
        return {
            "status": "RETRY",
            "feedback": f"Критик ответил в свободной форме. Текст ответа: {response[:200]}...",
        }
