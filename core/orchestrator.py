import asyncio
import re
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from agents.base import OllamaClient
from core.planner import TaskPlanner
from core.critic import CriticAgent
from core.memory import AgentMemory
from tools.shell import ShellTool


@dataclass
class SubtaskJob:
    subtask_id: int
    description: str
    role_name: str
    system_prompt: str
    temperature: float


class BatchOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self._cancelled = False
        self._active_tasks: List[asyncio.Task] = []
        self.executed_commands: List[str] = []

        ollama_cfg = config.get("ollama", {})
        exec_cfg = config.get("execution", {})
        paths_cfg = config.get("paths", {})

        self.base_url = ollama_cfg.get("base_url", "http://localhost:11434")
        self.executor_model = ollama_cfg.get("default_model", "qwen2.5-coder:14b")
        self.critic_model = ollama_cfg.get("critic_model", "deepseek-r1:14b")
        self.embed_model = ollama_cfg.get("embed_model", "nomic-embed-text:latest")
        self.timeout = float(ollama_cfg.get("timeout", 300.0))
        self.keep_alive = ollama_cfg.get("keep_alive", "10m")
        self.max_concurrent = int(ollama_cfg.get("max_concurrent_requests", 4))

        self.max_iterations = int(exec_cfg.get("max_iterations", 5))
        self.multi_role_count = int(exec_cfg.get("multi_role_count", 3))
        self.context_window = int(exec_cfg.get("context_window", 8192))
        self.max_workspace_sessions = int(exec_cfg.get("max_workspace_sessions", 15))

        self.workspace_base = Path(
            paths_cfg.get(
                "workspace_root",
                "/home/sid/RAID/_AI/universal_agent/storage/workspace",
            )
        )
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self._rotate_workspaces(keep_last=self.max_workspace_sessions)

        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.workspace_dir = self.workspace_base / session_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        default_roles = [
            {
                "name": "Senior_Developer",
                "temperature": 0.2,
                "system_prompt": "Ты ведущий разработчик. Пиши чистый, рабочий код и команды без заглушек.",
            },
            {
                "name": "System_Architect",
                "temperature": 0.3,
                "system_prompt": "Ты системный архитектор Linux/DevOps. Проверяй реальные домены, HTML-страницы и источники данных через curl/python3.",
            },
            {
                "name": "EdgeCase_Tester",
                "temperature": 0.4,
                "system_prompt": "Ты инженер по проверке фактов и тестированию. Никогда не выдумывай URL или данные, проверяй их через инструменты.",
            },
        ]
        all_roles = exec_cfg.get("roles", default_roles)
        self.active_roles = all_roles[: max(1, self.multi_role_count)]

        self.client = OllamaClient(
            base_url=self.base_url,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            default_num_ctx=self.context_window,
        )
        self.planner = TaskPlanner(self.client, self.executor_model)
        self.critic = CriticAgent(self.client, self.critic_model)

        memory_path = paths_cfg.get(
            "memory_path",
            "/home/sid/RAID/_AI/universal_agent/storage/memory/vector_memory.json",
        )
        skills_dir = paths_cfg.get(
            "skills_dir",
            "/home/sid/RAID/_AI/universal_agent/skills",
        )
        self.memory = AgentMemory(
            storage_path=memory_path,
            skills_dir=skills_dir,
            base_url=self.base_url,
            embed_model=self.embed_model,
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def _rotate_workspaces(self, keep_last: int = 15) -> None:
        if keep_last < 1 or not self.workspace_base.exists():
            return
        sessions = sorted(
            [
                p for p in self.workspace_base.iterdir()
                if p.is_dir() and p.name.startswith("session_")
            ],
            key=lambda p: p.name,
        )
        excess = len(sessions) - keep_last
        if excess > 0:
            for old_dir in sessions[:excess]:
                try:
                    shutil.rmtree(old_dir, ignore_errors=True)
                except Exception:
                    pass
            print(f"[Workspace] Ротация: удалено старых сессий: {excess}")

    def cancel(self):
        self._cancelled = True
        for t in self._active_tasks:
            if not t.done():
                t.cancel()

    def _check_cancelled(self):
        if self._cancelled:
            raise asyncio.CancelledError("Задача отменена пользователем.")
    async def _run_single_job(self, job: SubtaskJob, context_hint: str) -> Dict[str, Any]:
        async with self.semaphore:
            self._check_cancelled()
            print(f"[Queue -> GPU] Подзадача #{job.subtask_id} | Роль: {job.role_name}")

            tool_instruction = (
                "У тебя есть доступ к bash-песочнице Linux (с выходом в интернет, curl, python3).\n"
                "ЖЁСТКИЕ ПРАВИЛА РАБОТЫ С ИНСТРУМЕНТАМИ:\n"
                "1. Никогда не выдумывай несуществующие API-адреса или домены (например, не угадывай /v2/hits).\n"
                "2. Если нужно получить данные из интернета, сначала проверь реальный официальный сайт через curl -sL с User-Agent "
                "или найди точный URL через HTML-поиск на Python, затем распарси реальную страницу.\n"
                "3. Если команда вернула ошибку DNS, 403 или 404 — НЕ сдавайся и НЕ пиши пользователю инструкцию по диагностике сети. "
                "Вместо этого в следующем <bash>...</bash> проверь другой домен (например, .ru вместо .com) или другой способ парсинга.\n"
                "4. Для вызова команды выведи строго тег <bash>команда</bash> и больше ничего. "
                "Когда реальные факты получены — дай чёткий итоговый ответ."
            )

            prompt = (
                f"{tool_instruction}\n\n"
                f"Контекст из памяти и прошлых итераций:\n{context_hint}\n\n"
                f"Твоя узкая подзадача:\n{job.description}"
            )

            response = await self.client.generate(
                model=self.executor_model,
                prompt=prompt,
                system_prompt=job.system_prompt,
                temperature=job.temperature,
                num_ctx=self.context_window,
            )

            for step in range(1, 5):
                bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
                if not bash_match:
                    break

                self._check_cancelled()
                cmd = bash_match.group(1).strip()
                print(f"  [Shell #{job.subtask_id}:{job.role_name}] Шаг {step}: {cmd[:90]}...")
                shell_out = await ShellTool.run(cmd, workspace_dir=str(self.workspace_dir))
                if "[EXIT CODE]: 0" in shell_out and "[STDOUT]:" in shell_out:
                    self.executed_commands.append(cmd)

                followup_prompt = (
                    f"Ты запросил выполнение команды:\n{cmd}\n\n"
                    f"Результат выполнения из песочницы:\n{shell_out}\n\n"
                    f"Проанализируй вывод. Если данных ещё недостаточно или возникла ошибка (DNS/404/пустой вывод), "
                    f"выполни следующую команду через <bash>...</bash> с исправленным URL или парсером. "
                    f"Если нужные данные получены — дай полный финальный ответ по подзадаче: {job.description}"
                )
                response = await self.client.generate(
                    model=self.executor_model,
                    prompt=followup_prompt,
                    system_prompt=job.system_prompt,
                    temperature=job.temperature,
                    num_ctx=self.context_window,
                )

            return {
                "subtask_id": job.subtask_id,
                "description": job.description,
                "role": job.role_name,
                "output": response,
            }

    async def _synthesize_results(self, main_task: str, results: List[Dict[str, Any]]) -> str:
        self._check_cancelled()
        grouped = "\n---\n".join(
            f"Подзадача #{r['subtask_id']} ({r['description']}) [Роль: {r['role']}]:\n{r['output']}"
            for r in results
        )
        prompt = (
            f"Главная задача: {main_task}\n\n"
            f"Ниже приведены решения подзадач от разных ролей (включая реальные данные из терминала и сети). "
            f"Отбрось предположения, ошибки DNS и отписки. Используй только подтверждённые факты "
            f"и сформируй точный, полезный итоговый ответ на запрос пользователя:\n\n"
            f"{grouped}"
        )
        return await self.client.generate(
            model=self.executor_model,
            prompt=prompt,
            system_prompt="Ты технический лидер. Собери единый, точный и полный финальный ответ по существу задачи.",
            temperature=0.1,
            num_ctx=self.context_window,
        )

    async def run(self, task_text: str) -> str:
        feedback = ""
        synthesized_solution = ""
        self.executed_commands = []
        print(f"[Workspace] Изолированная папка сессии: {self.workspace_dir}")

        memory_context = ""
        try:
            past = await self.memory.search(task_text)
            if past:
                print("[Memory] Найдены релевантные решения из прошлого опыта!")
                memory_context = f"Релевантный прошлый опыт:\n{past}\n"
        except Exception as e:
            print(f"[Memory WARN] Поиск в памяти пропущен: {e}")

        for iteration in range(1, self.max_iterations + 1):
            self._check_cancelled()
            print(f"\n=== Итерация {iteration}/{self.max_iterations} | Фаза 1: Планирование ({self.executor_model}) ===")
            full_context = f"{memory_context}\n{feedback}".strip()
            subtasks = await self.planner.decompose(task_text, context=full_context)
            print(f"[+] Сформировано подзадач: {len(subtasks)}")

            job_queue: List[SubtaskJob] = []
            for idx, sub in enumerate(subtasks, start=1):
                for role in self.active_roles:
                    job_queue.append(
                        SubtaskJob(
                            subtask_id=idx,
                            description=sub,
                            role_name=role["name"],
                            system_prompt=role["system_prompt"],
                            temperature=role["temperature"],
                        )
                    )

            print(f"=== Фаза 2: Очередь из {len(job_queue)} сессий на модели {self.executor_model} ===")
            self._active_tasks = [
                asyncio.create_task(self._run_single_job(job, full_context))
                for job in job_queue
            ]
            raw_results = await asyncio.gather(*self._active_tasks)
            self._active_tasks.clear()

            print(f"=== Фаза 3: Синтез результатов на модели {self.executor_model} ===")
            synthesized_solution = await self._synthesize_results(task_text, raw_results)

            print(f"=== Фаза 4: Проверка критиком ({self.critic_model}) ===")
            verdict = await self.critic.evaluate(task_text, synthesized_solution)

            status = str(verdict.get("status", "RETRY")).upper()
            critique_feedback = verdict.get("feedback", "Требуется доработка.")

            if status == "SUCCESS":
                print(f"[SUCCESS] Критик утвердил решение: {critique_feedback}")
                try:
                    await self.memory.save(
                        task_text,
                        synthesized_solution,
                        executed_commands=self.executed_commands,
                    )
                    print("[Memory] Опыт и рабочие команды сохранены в память и реестр навыков.")
                except Exception as e:
                    print(f"[Memory WARN] Не удалось сохранить опыт: {e}")
                return synthesized_solution

            feedback = f"Замечания критика (итерация {iteration}): {critique_feedback}"
            print(f"[RETRY] Требуется доработка: {critique_feedback}")

        print("[WARN] Достигнут лимит итераций. Возвращается лучшее собранное решение.")
        return synthesized_solution


Orchestrator = BatchOrchestrator
