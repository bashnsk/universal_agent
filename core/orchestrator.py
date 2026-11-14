import asyncio
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from agents.base import OllamaClient
from core.planner import Planner
from core.critic import Critic

@dataclass
class SubtaskJob:
    subtask_id: int
    description: str
    role_name: str
    system_prompt: str
    temperature: float

class BatchOrchestrator:
    def __init__(self, config: Dict[str, Any], log_callback=None):
        self.cfg = config
        self.client = OllamaClient(
            base_url=config["ollama"]["base_url"],
            timeout=config["ollama"]["timeout"],
            keep_alive=config["ollama"]["keep_alive"]
        )
        self.planner = Planner(self.client, config["models"]["planner"])
        self.critic = Critic(self.client, config["models"]["critic"])
        self.semaphore = asyncio.Semaphore(config["ollama"]["max_concurrent_requests"])
        self.log = log_callback or print

    async def _run_single_job(self, job: SubtaskJob, context_hint: str) -> Dict[str, Any]:
        """Выполнение одной атомарной сессии в рамках лимита параллельных слотов VRAM."""
        async with self.semaphore:
            self.log(f"[Queue -> GPU] Подзадача #{job.subtask_id} | Роль: {job.role_name}")
            prompt = (
                f"Контекст из памяти/прошлых шагов:\n{context_hint}\n\n"
                f"Твоя узкая подзадача:\n{job.description}\n\n"
                f"Выдай точное, готовое к применению решение."
            )
            response = await self.client.generate(
                model=self.cfg["models"]["executor"],
                prompt=prompt,
                system=job.system_prompt,
                options={
                    "temperature": job.temperature,
                    "num_ctx": self.cfg["execution"]["context_window"]
                }
            )
            return {
                "subtask_id": job.subtask_id,
                "description": job.description,
                "role": job.role_name,
                "output": response
            }

    async def execute_pipeline(self, main_task: str, memory_context: str = "") -> str:
        feedback = ""
        max_iters = self.cfg["execution"]["max_iterations"]

        for iteration in range(1, max_iters + 1):
            self.log(f"=== Итерация {iteration}/{max_iters} | Фаза 1: Планирование ({self.cfg['models']['planner']}) ===")
            subtasks = await self.planner.decompose(main_task, memory_context, feedback)

            # Формируем единую очередь (все подзадачи × все роли) для одной модели
            job_queue: List[SubtaskJob] = []
            for idx, sub in enumerate(subtasks, start=1):
                for role in self.cfg["execution"]["roles"]:
                    job_queue.append(SubtaskJob(
                        subtask_id=idx,
                        description=sub,
                        role_name=role["name"],
                        system_prompt=role["system_prompt"],
                        temperature=role["temperature"]
                    ))

            self.log(f"=== Фаза 2: Очередь из {len(job_queue)} сессий на {self.cfg['models']['executor']} ===")
            tasks = [self._run_single_job(job, f"{memory_context}\n{feedback}") for job in job_queue]
            raw_results = await asyncio.gather(*tasks)

            # Аккумуляция и арбитраж лучших ответов на той же модели (без перезагрузки VRAM)
            self.log(f"=== Фаза 3: Синтез результатов на {self.cfg['models']['executor']} ===")
            synthesized_solution = await self._synthesize_results(main_task, raw_results)

            # Переключение модели в VRAM строго 1 раз за итерацию для финального аудита
            self.log(f"=== Фаза 4: Проверка критиком ({self.cfg['models']['critic']}) ===")
            verdict = await self.critic.evaluate(main_task, synthesized_solution)

            if verdict["status"] == "SUCCESS":
                self.log("[SUCCESS] Критик утвердил решение.")
                return synthesized_solution

            feedback = f"Замечания критика с итерации {iteration}: {verdict['critique']}"
            self.log(f"[RETRY] Требуется доработка: {verdict['critique']}")

        return synthesized_solution

    async def _synthesize_results(self, main_task: str, results: List[Dict[str, Any]]) -> str:
        """Сборка единого решения из мультиролевых ответов на текущей загруженной модели."""
        grouped = "\n---\n".join(
            f"Подзадача #{r['subtask_id']} ({r['description']}) [Роль: {r['role']}]:\n{r['output']}"
            for r in results
        )
        prompt = (
            f"Главная задача: {main_task}\n\n"
            f"Ниже приведены решения подзадач от разных специалистов. "
            f"Сравни их, отбрось ошибочные варианты, возьми лучшие технические решения и собери единый целостный результат:\n\n"
            f"{grouped}"
        )
        return await self.client.generate(
            model=self.cfg["models"]["executor"],
            prompt=prompt,
            options={"temperature": 0.1, "num_ctx": self.cfg["execution"]["context_window"]}
        )
