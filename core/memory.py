import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx


class AgentMemory:
    def __init__(
        self,
        storage_path: str = "/home/sid/RAID/_AI/universal_agent/storage/memory/vector_memory.json",
        skills_dir: str = "/home/sid/RAID/_AI/universal_agent/skills",
        base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text:latest",
    ):
        passed_path = Path(storage_path)
        if passed_path.name == "registry.json":
            self.memory_path = Path("/home/sid/RAID/_AI/universal_agent/storage/memory/vector_memory.json")
            self.skills_dir = passed_path.parent
        else:
            self.memory_path = passed_path
            self.skills_dir = Path(skills_dir)

        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.skills_dir / "registry.json"

        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.records: List[Dict[str, Any]] = self._load_memory()
        self.skills_registry: List[Dict[str, Any]] = self._load_registry()

    def _load_memory(self) -> List[Dict[str, Any]]:
        if self.memory_path.exists():
            try:
                content = self.memory_path.read_text(encoding="utf-8").strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def _save_memory_file(self) -> None:
        self.memory_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_registry(self) -> List[Dict[str, Any]]:
        if self.registry_path.exists():
            try:
                content = self.registry_path.read_text(encoding="utf-8").strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def _save_registry_file(self) -> None:
        self.registry_path.write_text(
            json.dumps(self.skills_registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.embed_model,
            "prompt": text[:4000],
            "keep_alive": "10s",
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json().get("embedding")
        except Exception:
            return None

    async def search(self, query: str, top_k: int = 3, min_score: float = 0.65) -> str:
        if not self.records:
            return ""

        query_vec = await self._get_embedding(query)
        scored: List[tuple[float, Dict[str, Any]]] = []

        for item in self.records:
            score = 0.0
            if query_vec and item.get("embedding"):
                score = self._cosine_similarity(query_vec, item["embedding"])
            else:
                q_words = set(re.findall(r"\w+", query.lower()))
                t_words = set(re.findall(r"\w+", item.get("task", "").lower()))
                if q_words and t_words:
                    score = len(q_words & t_words) / len(q_words | t_words)

            if score >= min_score:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = scored[:top_k]
        if not top_items:
            return ""

        blocks = []
        for score, rec in top_items:
            is_success = rec.get("is_success", True)
            if not is_success:
                bad_cmds = rec.get("failed_commands") or []
                cmds_str = f"\nЗабракованные команды/URL: {', '.join(bad_cmds[:5])}" if bad_cmds else ""
                blocks.append(
                    f"[Сходство {score:.2f}] ⛔ АНТИ-ПАТТЕРН (ЭТИМ ПУТЁМ ИДТИ НЕ НАДО!):\n"
                    f"Задача: {rec['task']}\n"
                    f"Причина провала: {rec.get('failure_reason', 'Не получены фактические данные')}{cmds_str}\n"
                    f"ИНСТРУКЦИЯ: Категорически запрещено повторять этот подход. Ищи принципиально другой источник или домен."
                )
            else:
                skill_hint = ""
                if rec.get("skill_file"):
                    skill_hint = f"\nСохранённый навык: {rec['skill_file']}"
                blocks.append(
                    f"[Сходство {score:.2f}] ✅ ПРОВЕРЕННОЕ УСПЕШНОЕ РЕШЕНИЕ:\n"
                    f"Задача: {rec['task']}\n"
                    f"Результат:\n{rec['solution'][:1500]}{skill_hint}"
                )
        return "\n---\n".join(blocks)

    async def save(
        self,
        task: str,
        solution: str,
        executed_commands: Optional[List[str]] = None,
        is_success: bool = True,
        failure_reason: str = "",
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        embedding = await self._get_embedding(task)
        unique_cmds = list(dict.fromkeys(c.strip() for c in (executed_commands or []) if c.strip()))

        skill_filename = ""
        if is_success and unique_cmds:
            skill_filename = f"skill_{timestamp}.json"
            skill_data = {
                "id": f"skill_{timestamp}",
                "created_at": datetime.now().isoformat(),
                "task_pattern": task,
                "commands": unique_cmds,
                "summary": solution[:1000],
            }
            (self.skills_dir / skill_filename).write_text(
                json.dumps(skill_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.skills_registry.append(
                {
                    "id": skill_data["id"],
                    "file": skill_filename,
                    "task_pattern": task,
                    "commands": unique_cmds[:5],
                    "created_at": skill_data["created_at"],
                }
            )
            self._save_registry_file()

        record: Dict[str, Any] = {
            "timestamp": timestamp,
            "task": task,
            "is_success": is_success,
            "solution": solution if is_success else "",
            "failure_reason": "" if is_success else (failure_reason or "Отклонено критиком: нет фактических данных"),
            "failed_commands": [] if is_success else unique_cmds[:8],
            "skill_file": skill_filename,
            "embedding": embedding or [],
        }
        self.records.append(record)
        self._save_memory_file()
