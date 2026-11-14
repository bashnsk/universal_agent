import json
import re
from pathlib import Path
from typing import Optional
from agents.base import OllamaClient

class TaskPlanner:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        skills_registry_path: str = "/home/sid/RAID/_AI/universal_agent/skills/registry.json",
    ):
        self.client = client
        self.model = model
        self.skills_registry_path = Path(skills_registry_path)

    def _resolve_registry_path(self) -> Optional[Path]:
        if self.skills_registry_path.exists():
            return self.skills_registry_path
        fallback = Path("skills/registry.json")
        if fallback.exists():
            return fallback
        return None

    def _extract_commands_preview(self, registry_path: Path, entry: dict) -> str:
        cmds = entry.get("commands")
        if not cmds and entry.get("file"):
            skill_file = registry_path.parent / str(entry["file"])
            if skill_file.exists():
                try:
                    skill_data = json.loads(skill_file.read_text(encoding="utf-8"))
                    cmds = skill_data.get("commands")
                except Exception:
                    cmds = None

        if isinstance(cmds, list) and cmds:
            cleaned_cmds = [str(c).strip().replace("\n", " ")[:80] for c in cmds[:3] if str(c).strip()]
            if cleaned_cmds:
                return f" | команды: {'; '.join(cleaned_cmds)}"
        return ""

    def _load_skills_summary(self, limit: int = 15) -> str:
        registry_path = self._resolve_registry_path()
        if not registry_path:
            return ""

        try:
            raw_data = registry_path.read_text(encoding="utf-8").strip()
            if not raw_data:
                return ""
            data = json.loads(raw_data)
        except Exception:
            return ""

        if isinstance(data, dict):
            entries = list(data.values())
        elif isinstance(data, list):
            entries = data
        else:
            return ""

        items = []
        for entry in entries[-limit:]:
            if not isinstance(entry, dict):
                continue

            skill_id = entry.get("id") or entry.get("file") or "skill"
            pattern = (
                entry.get("task_pattern")
                or entry.get("description")
                or entry.get("task")
                or entry.get("summary")
                or ""
            )
            pattern_str = str(pattern).strip().replace("\n", " ")[:180]
            cmds_preview = self._extract_commands_preview(registry_path, entry)

            if pattern_str:
                items.append(f"- [{skill_id}] Задача: {pattern_str}{cmds_preview}")
            else:
                items.append(f"- [{skill_id}]{cmds_preview}")

        return "\n".join(items)

    def _parse_subtasks(self, response: str, fallback_task: str) -> list[str]:
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        candidate = match.group(0) if match else cleaned

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                subtasks = [str(item).strip() for item in parsed if str(item).strip()]
                if subtasks:
                    return subtasks
        except Exception:
            pass

        return [fallback_task]

    async def decompose(self, main_task: str, context: str = "") -> list[str]:
        skills_summary = self._load_skills_summary()

        skills_instruction = ""
        if skills_summary:
            skills_instruction = (
                "\n\nВ системе уже сохранены проверенные навыки (рецепты решения прошлых задач):\n"
                f"{skills_summary}\n"
                "Если какой-либо из этих навыков подходит для решения части новой задачи, явно укажи в формулировке "
                "соответствующей подзадачи, какой навык и команды стоит переиспользовать."
            )

        system = (
            "Ты Senior Tech Lead. Разбей главную задачу на независимые атомарные подзадачи, "
            "которые можно выполнять параллельно."
            f"{skills_instruction}\n\n"
            'Верни результат СТРОГО в формате JSON-массива строк без пояснений: ["подзадача 1", "подзадача 2"]'
        )

        prompt_parts = [f"Главная задача: {main_task}"]
        if context and context.strip():
            prompt_parts.append(f"Контекст прошлых попыток и замечания критика:\n{context.strip()}")
        prompt = "\n\n".join(prompt_parts)

        response = await self.client.generate(
            self.model, prompt, system_prompt=system, temperature=0.3
        )
        return self._parse_subtasks(response, main_task)
