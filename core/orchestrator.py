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
        self.all_attempted_commands: List[str] = []

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
                "system_prompt": "Ты ведущий разработчик. Пиши рабочий код, парсеры и команды, извлекающие реальные данные без заглушек.",
            },
            {
                "name": "System_Architect",
                "temperature": 0.3,
                "system_prompt": "Ты системный архитектор и исследователь. Если один источник или метод не работает, находишь альтернативный рабочий источник.",
            },
            {
                "name": "EdgeCase_Tester",
                "temperature": 0.4,
                "system_prompt": "Ты инженер верификации данных. Проверяешь достоверность каждого факта через инструменты и отсекаешь догадки.",
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
                "У тебя есть доступ к bash-песочнице Linux (curl, python3, стандартные утилиты).\n"
                "УНИВЕРСАЛЬНЫЕ ПРАВИЛА ПОИСКА РЕШЕНИЙ:\n"
                "1. Запрещено выдумывать несуществующие URL, API-эндпоинты и данные.\n"
                "2. Запрещено перекладывать задачу на пользователя (не пиши 'зарегистрируйтесь и получите ключ' или 'проверьте сами').\n"
                "3. Если выбранный источник или метод вернул ошибку или пустоту — немедленно меняй стратегию в следующем <bash>...</bash>: "
                "ищи открытые веб-страницы, парси HTML/RSS через python3, проверяй альтернативные источники и зеркала.\n"
                "4. Если в контексте указаны АНТИ-ПАТТЕРНЫ (тупиковые пути) — категорически запрещено их повторять.\n"
                "5. Для запуска команды выведи строго <bash>команда</bash> без лишнего текста. "
                "Когда реальные факты получены — дай точный, структурированный ответ."
            )

            prompt = (
                f"{tool_instruction}\n\n"
                f"Контекст из памяти, тупиковых путей и прошлых итераций:\n{context_hint}\n\n"
                f"Твоя подзадача:\n{job.description}"
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
                self.all_attempted_commands.append(cmd)
                print(f"  [Shell #{job.subtask_id}:{job.role_name}] Шаг {step}: {cmd[:90]}...")
                shell_out = await ShellTool.run(cmd, workspace_dir=str(self.workspace_dir))
                if "[EXIT CODE]: 0" in shell_out and "[STDOUT]:" in shell_out:
                    self.executed_commands.append(cmd)

                followup_prompt = (
                    f"Выполнена команда:\n{cmd}\n\n"
                    f"Вывод песочницы:\n{shell_out}\n\n"
                    f"Оцени результат: если реальные фактические данные ещё не добыты или возникла ошибка, "
                    f"попробуй принципиально другой подход через <bash>...</bash>. "
                    f"Если факты получены — сформируй итоговый ответ по подзадаче: {job.description}"
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
            f"Ниже приведены результаты работы агентов:\n{grouped}\n\n"
            f"ЖЁСТКОЕ ТРЕБОВАНИЕ: Отбрось все отписки, предположения, ошибки доступа и советы 'как сделать вручную'. "
            f"Включи в финальный ответ ТОЛЬКО подтверждённые фактические данные, напрямую решающие задачу."
        )
        return await self.client.generate(
            model=self.executor_model,
            prompt=prompt,
            system_prompt="Ты технический лидер. Собираешь только доказанные факты и рабочие решения.",
            temperature=0.1,
            num_ctx=self.context_window,
        )

    async def run(self, task_text: str) -> str:
        dead_ends: List[str] = []
        last_feedback = ""
        synthesized_solution = ""
        self.executed_commands = []
        self.all_attempted_commands = []
        print(f"[Workspace] Изолированная папка сессии: {self.workspace_dir}")

        memory_context = ""
        try:
            past = await self.memory.search(task_text)
            if past:
                print("[Memory] Загружен релевантный опыт и анти-паттерны из памяти.")
                memory_context = f"Записи из памяти (успешные рецепты и тупики):\n{past}\n"
        except Exception as e:
            print(f"[Memory WARN] Поиск в памяти пропущен: {e}")

        for iteration in range(1, self.max_iterations + 1):
            self._check_cancelled()
            print(f"\n=== Итерация {iteration}/{self.max_iterations} | Фаза 1: Планирование ({self.executor_model}) ===")

            dead_ends_block = ""
            if dead_ends:
                formatted_dead_ends = "\n".join(f"- ЭТИМ ПУТЁМ ИДТИ НЕ НАДО: {d}" for d in dead_ends)
                dead_ends_block = (
                    "\n⛔ ТУПИКОВЫЕ ПУТИ ТЕКУЩЕЙ ЗАДАЧИ (ПОВТОРЯТЬ ЗАПРЕЩЕНО):\n"
                    f"{formatted_dead_ends}\n"
                    "Построй новый план, использующий принципиально иные источники или методы!\n"
                )

            full_context = f"{memory_context}\n{dead_ends_block}\n{last_feedback}".strip()
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

            print(f"=== Фаза 4: Жёсткий аудит критиком ({self.critic_model}) ===")
            verdict = await self.critic.evaluate(task_text, synthesized_solution)

            status = str(verdict.get("status", "RETRY")).upper()
            critique_feedback = str(verdict.get("feedback", "Не получены фактические данные.")).strip()
            dead_end = str(verdict.get("dead_end", "")).strip()
            next_strategy = str(verdict.get("next_strategy", "")).strip()

            if status == "SUCCESS":
                print(f"[SUCCESS] Критик подтвердил наличие фактических данных: {critique_feedback}")
                try:
                    await self.memory.save(
                        task_text,
                        synthesized_solution,
                        executed_commands=self.executed_commands,
                        is_success=True,
                    )
                    print("[Memory] Успешное решение и проверенный навык сохранены.")
                except Exception as e:
                    print(f"[Memory WARN] Не удалось сохранить опыт: {e}")
                return synthesized_solution

            recorded_dead_end = dead_end or critique_feedback
            if recorded_dead_end not in dead_ends:
                dead_ends.append(recorded_dead_end)

            last_feedback = (
                f"Замечание критика (итерация {iteration}): {critique_feedback}\n"
                f"Рекомендуемый альтернативный вектор: {next_strategy}"
            ).strip()
            print(f"[RETRY] Критик отклонил ответ: {critique_feedback}")
            if next_strategy:
                print(f"[SEARCH ALT] Новый вектор поиска: {next_strategy}")

        # Если все итерации исчерпаны, но задача НЕ решена:
        # 1. Не сохраняем провал в навыки (skills/).
        # 2. Фиксируем в векторной памяти анти-паттерн, чтобы будущие запуски обходили эти тупики.
        final_failure_reason = "; ".join(dead_ends[-3:]) if dead_ends else "Задача не решена за отведённые итерации."
        print(f"[FAILED] Задача не выполнена. Фиксируем тупиковый путь в памяти как анти-паттерн.")
        try:
            await self.memory.save(
                task_text,
                "",
                executed_commands=self.all_attempted_commands,
                is_success=False,
                failure_reason=final_failure_reason,
            )
        except Exception as e:
            print(f"[Memory WARN] Не удалось зафиксировать анти-паттерн: {e}")

        return (
            f"## Статус: Задача не выполнена за {self.max_iterations} итераций\n\n"
            f"Система не приняла неподтверждённые ответы и зафиксировала тупиковые пути в памяти, "
            f"чтобы не повторять их в будущем:\n"
            + "\n".join(f"- ⛔ {d}" for d in dead_ends)
        )


Orchestrator = BatchOrchestrator
