import json
import re
from typing import Optional
from agents.base import OllamaClient


class CriticAgent:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def _has_obvious_placeholder_or_excuse(self, text: str) -> str:
        lowered = text.lower()
        excuse_markers = [
            "your_api_key",
            "your_client_id",
            "your_access_token",
            "не существует в dns",
            "недоступен из-за проблем с dns",
            "рекомендуется использовать официальные api",
            "зарегистрируйтесь и получите",
            "обратитесь в техническую поддержку",
            "свяжитесь с поддержкой",
        ]
        for marker in excuse_markers:
            if marker in lowered:
                return f"Обнаружена отписка или заглушка вместо реальных данных ('{marker}')."
        return ""

    async def evaluate(
        self,
        main_task: str,
        accumulated_results: str,
        model_override: Optional[str] = None,
    ) -> dict:
        target_model = model_override or self.model

        if not accumulated_results or len(accumulated_results.strip()) < 20:
            return {
                "status": "RETRY",
                "feedback": "Ответ пустой или слишком короткий. Необходимо выполнить задачу и предоставить фактический результат.",
                "dead_end": "Пустой результат генерации",
                "next_strategy": "Использовать инструменты (<bash>) для сбора реальных данных по задаче.",
            }

        heuristic_fail = self._has_obvious_placeholder_or_excuse(accumulated_results)
        if heuristic_fail:
            return {
                "status": "RETRY",
                "feedback": f"Жёсткий отказ: {heuristic_fail} Запрещено предлагать пользователю инструкции вместо готового результата.",
                "dead_end": heuristic_fail,
                "next_strategy": "Найти открытый рабочий источник или выполнить прямой парсинг/вычисление без заглушек.",
            }

        system = (
            "Ты строгий независимый Аудитор и Верификатор фактов.\n"
            "Твоя главная задача — жёстко отклонять (RETRY) любые ответы, в которых нет реального решения задачи.\n\n"
            "ПРАВИЛА ПРИЁМКИ:\n"
            "1. Статус SUCCESS ставится ТОЛЬКО если ответ содержит конкретные фактические данные, "
            "подтверждённые сущности, рабочий код или точный результат, прямо отвечающий на запрос пользователя.\n"
            "2. Ставь RETRY, если ответ является 'отпиской': содержит жалобы на недоступность ресурса, "
            "советы пользователю сделать работу самостоятельно, инструкции 'получите API-ключ', "
            "плейсхолдеры (YOUR_KEY, example) или общие рассуждения без фактов.\n"
            "3. При статусе RETRY чётко сформулируй, какой путь оказался тупиковым (dead_end) и "
            "какое принципиально другое направление поиска нужно применить (next_strategy).\n\n"
            "Верни ответ СТРОГО в формате JSON без пояснений:\n"
            '{"status": "SUCCESS" или "RETRY", "feedback": "разбор", "dead_end": "тупиковый путь или пусто", "next_strategy": "альтернативный путь или пусто"}'
        )
        prompt = f"Главная задача пользователя:\n{main_task}\n\nПредставленный ответ на проверку:\n{accumulated_results}"

        response = await self.client.generate(
            target_model, prompt, system_prompt=system, temperature=0.1
        )

        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        for candidate in (clean_response, response):
            try:
                cleaned = candidate.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "status" in parsed:
                    return parsed
            except Exception:
                pass

            try:
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict) and "status" in parsed:
                        return parsed
            except Exception:
                pass

        return {
            "status": "RETRY",
            "feedback": f"Критик не подтвердил корректность ответа. Фрагмент: {clean_response[:250]}",
            "dead_end": "Невалидный формат верификации",
            "next_strategy": "Перепроверить данные через инструменты и сформировать точный фактологический ответ.",
        }
