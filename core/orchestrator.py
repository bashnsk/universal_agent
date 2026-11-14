import asyncio
from typing import List, Dict, Any
from agents.base import OllamaClient
from core.blackboard import SessionBlackboard
from core.critic import CriticAgent
from core.memory import AgentMemory
from core.planner import TaskPlanner
from core.worker import SubtaskJob, SubtaskWorker
from core.workspace import WorkspaceManager


class BatchOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self._cancelled = False
        self._active_tasks: List[asyncio.Task] = []

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

        workspace_root = paths_cfg.get(
            "workspace_root",
            "/home/sid/RAID/_AI/universal_agent/storage/workspace",
        )
        self.workspace = WorkspaceManager(
            root_dir=workspace_root,
            keep_last=int(exec_cfg.get("max_workspace_sessions", 15)),
        )
        self.blackboard = SessionBlackboard()

        default_roles = [
            {
                "name": "Senior_Developer",
                "temperature": 0.2,
                "system_prompt": "Ты ведущий разработчик. Извлекаешь реальные структурированные данные без заглушек.",
            },
            {
                "name": "System_Architect",
                "temperature": 0.3,
                "system_prompt": "Ты системный исследователь. Находишь точные внутренние разделы сайтов и рабочие источники.",
            },
            {
                "name": "EdgeCase_Tester",
                "temperature": 0.4,
                "system_prompt": "Ты инженер верификации. Отсекаешь страницы 404, пустые списки и неподтверждённые данные.",
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
        self.memory = AgentMemory(
            storage_path=paths_cfg.get(
                "memory_path",
                "/home/sid/RAID/_AI/universal_agent/storage/memory/vector_memory.json",
            ),
            skills_dir=paths_cfg.get(
                "skills_dir",
                "/home/sid/RAID/_AI/universal_agent/skills",
            ),
            base_url=self.base_url,
            embed_model=self.embed_model,
        )
        self.worker = SubtaskWorker(
            client=self.client,
            model=self.executor_model,
            context_window=self.context_window,
            workspace_dir=str(self.workspace.session_dir),
            blackboard=self.blackboard,
            semaphore=asyncio.Semaphore(self.max_concurrent),
            cancel_checker=self._check_cancelled,
        )

    def cancel(self) -> None:
        self._cancelled = True
        for t in self._active_tasks:
            if not t.done():
                t.cancel()

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError("Задача отменена пользователем.")

    async def _synthesize_results(self, main_task: str, results: List[Dict[str, Any]]) -> str:
        self._check_cancelled()
        grouped = "\n---\n".join(
            f"Подзадача #{r['subtask_id']} ({r['description']}) [Роль: {r['role']}]:\n{r['output']}"
            for r in results
        )
        blackboard_facts = ""
        if self.blackboard.extracted_facts:
            blackboard_facts = (
                "\n\n=== СЫРЫЕ ИЗВЛЕЧЁННЫЕ ДАННЫЕ С РЕАЛЬНЫХ СТРАНИЦ ===\n"
                + "\n---\n".join(self.blackboard.extracted_facts[-3:])[:3500]
            )

        prompt = (
            f"Главная задача пользователя: {main_task}\n\n"
            f"Отчёты агентов:\n{grouped}"
            f"{blackboard_facts}\n\n"
            f"ЖЁСТКОЕ ТРЕБОВАНИЕ: Отбрось все неудачные попытки, нерелевантные сайты и советы 'сделайте сами'. "
            f"Используя реальные данные из отчётов и извлечённых страниц, сформируй точный фактический ответ."
        )
        return await self.client.generate(
            model=self.executor_model,
            prompt=prompt,
            system_prompt="Ты технический лидер. Собираешь только доказанные факты и рабочие решения.",
            temperature=0.1,
            num_ctx=self.context_window,
        )
    async def _run_iteration(self, task_text: str, label: str, memory_context: str, last_feedback: str) -> str:
        self._check_cancelled()
        print(f"\n=== {label} | Фаза 1: Планирование ({self.executor_model}) ===")

        full_context = self.blackboard.build_prompt_context(memory_context, last_feedback)
        subtasks = await self.planner.decompose(task_text, context=full_context)
        print(f"[+] Сформировано подзадач: {len(subtasks)}")

        job_queue = [
            SubtaskJob(
                subtask_id=idx,
                description=sub,
                role_name=role["name"],
                system_prompt=role["system_prompt"],
                temperature=role["temperature"],
            )
            for idx, sub in enumerate(subtasks, start=1)
            for role in self.active_roles
        ]

        print(f"=== Фаза 2: Очередь из {len(job_queue)} сессий на модели {self.executor_model} ===")
        self._active_tasks = [
            asyncio.create_task(self.worker.execute(job, full_context))
            for job in job_queue
        ]
        raw_results = await asyncio.gather(*self._active_tasks)
        self._active_tasks.clear()

        print(f"=== Фаза 3: Синтез результатов на модели {self.executor_model} ===")
        return await self._synthesize_results(task_text, raw_results)

    async def run(self, task_text: str) -> str:
        last_feedback = ""
        synthesized_solution = ""
        fast_verified = False

        self.blackboard.clear()
        print(f"[Workspace] Изолированная папка сессии: {self.workspace.session_dir}")

        memory_context = ""
        try:
            past = await self.memory.search(task_text)
            if past:
                print("[Memory] Загружены проверенные решения похожих задач.")
                memory_context = past
        except Exception as e:
            print(f"[Memory WARN] Поиск в памяти пропущен: {e}")

        for iteration in range(1, self.max_iterations + 1):
            synthesized_solution = await self._run_iteration(
                task_text=task_text,
                label=f"Итерация {iteration}/{self.max_iterations}",
                memory_context=memory_context,
                last_feedback=last_feedback,
            )

            print(f"=== Фаза 4.1: Быстрый аудит без смены VRAM ({self.executor_model}) ===")
            verdict = await self.critic.evaluate(
                task_text,
                synthesized_solution,
                model_override=self.executor_model,
            )

            if str(verdict.get("status", "RETRY")).upper() == "SUCCESS":
                print(f"[FAST-PASS] Предварительная проверка пройдена: {verdict.get('feedback', '')}")
                fast_verified = True
                break

            last_feedback = self.blackboard.apply_critic_rejection(verdict, iteration)
            print(f"[RETRY] Критик отклонил ответ: {verdict.get('feedback', '')}")

        if fast_verified and self.critic_model and self.critic_model != self.executor_model:
            self._check_cancelled()
            print(f"\n=== Фаза 4.2: Финальный глубокий аудит ({self.critic_model}) ===")
            final_verdict = await self.critic.evaluate(
                task_text,
                synthesized_solution,
                model_override=self.critic_model,
            )

            if str(final_verdict.get("status", "RETRY")).upper() != "SUCCESS":
                final_feedback = str(final_verdict.get("feedback", "")).strip()
                print(f"[FINAL-CRITIC REJECT] Глубокий аудит выявил недочёт: {final_feedback}")
                last_feedback = self.blackboard.apply_critic_rejection(final_verdict, self.max_iterations + 1)

                synthesized_solution = await self._run_iteration(
                    task_text=task_text,
                    label="Финальная доработка после глубокого аудита",
                    memory_context=memory_context,
                    last_feedback=last_feedback,
                )
                recheck = await self.critic.evaluate(
                    task_text,
                    synthesized_solution,
                    model_override=self.executor_model,
                )
                if str(recheck.get("status", "RETRY")).upper() != "SUCCESS":
                    fast_verified = False
            else:
                print(f"[SUCCESS] Финальный аудитор утвердил решение: {final_verdict.get('feedback', '')}")

        if fast_verified:
            try:
                await self.memory.save(
                    task_text,
                    synthesized_solution,
                    executed_commands=self.blackboard.working_commands,
                    is_success=True,
                )
                print("[Memory] Победившее решение и рабочие инструменты сохранены в базу навыков.")
            except Exception as e:
                print(f"[Memory WARN] Не удалось сохранить опыт: {e}")

            self.blackboard.clear()
            return synthesized_solution

        print("[FAILED] Задача не решена. Временный буфер сессии очищен, постоянная память не засорена.")
        failed_summary = "\n".join(f"- ⛔ {item}" for item in self.blackboard.failed_attempts[-8:])
        self.blackboard.clear()

        return (
            f"## Статус: Задача не выполнена за {self.max_iterations} итераций\n\n"
            f"Система отклонила неподтверждённые ответы и не сохранила неудачный запуск в базу навыков.\n\n"
            f"### Проверенные в сессии тупиковые пути:\n{failed_summary or '- Нет данных'}"
        )


Orchestrator = BatchOrchestrator
