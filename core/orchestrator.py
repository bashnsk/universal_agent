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

        # Оперативная память текущей сессии (очищается после завершения задачи)
        self.session_working_commands: List[str] = []
        self.session_failed_attempts: List[str] = []

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
                "system_prompt": "Ты ведущий разработчик. Пиши рабочий код, парсеры и скрипты, извлекающие реальные данные без заглушек.",
            },
            {
                "name": "System_Architect",
                "temperature": 0.3,
                "system_prompt": "Ты системный архитектор и исследователь. Находишь реальные источники данных и рабочие инструменты вместо угадывания.",
            },
            {
                "name": "EdgeCase_Tester",
                "temperature": 0.4,
                "system_prompt": "Ты инженер верификации. Проверяешь фактический вывод команд и отбрасываешь пустые или ошибочные ответы.",
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

    def _record_session_failure(self, cmd: str, reason: str) -> None:
        short_cmd = cmd.strip().replace("\n", " ")[:120]
        short_reason = reason.strip().replace("\n", " ")[:140]
        entry = f"Команда `{short_cmd}` -> ОШИБКА: {short_reason}"
        if entry not in self.session_failed_attempts:
            self.session_failed_attempts.append(entry)

    def _is_command_output_useful(self, shell_out: str) -> tuple[bool, str]:
        if "[EXIT CODE]: 0" not in shell_out:
            return False, "ненулевой код возврата или ошибка выполнения"
        if "[STDOUT]:" not in shell_out:
            return False, "пустой вывод (нет данных)"
        stdout_part = shell_out.split("[STDOUT]:", 1)[1].strip()
        if len(stdout_part) < 15:
            return False, "слишком короткий или пустой ответ"
        return True, ""

    async def _run_single_job(self, job: SubtaskJob, context_hint: str) -> Dict[str, Any]:
        async with self.semaphore:
            self._check_cancelled()
            print(f"[Queue -> GPU] Подзадача #{job.subtask_id} | Роль: {job.role_name}")

            tool_instruction = (
                "У тебя есть доступ к bash-песочнице Linux и готовым Python-инструментам:\n"
                "- Поиск ссылок в интернете: python3 /home/sid/RAID/_AI/universal_agent/skills/web_search.py \"запрос\" --limit 5\n"
                "- Парсинг одной страницы (BeautifulSoup): python3 /home/sid/RAID/_AI/universal_agent/skills/web_parser.py \"https://реальный_url\"\n\n"
                "УНИВЕРСАЛЬНЫЕ ПРАВИЛА:\n"
                "1. Для поиска в сети ВСЕГДА используй web_search.py, а для чтения страницы — передай ОДИН конкретный реальный URL в web_parser.py. "
                "ЗАПРЕЩЕНО делать прямой curl к ://google.com и использовать шаблоны вроде <URL1>.\n"
                "2. Никогда не повторяй команды из блока 'НЕУДАЧНЫЕ ПОПЫТКИ ТЕКУЩЕЙ ЗАДАЧИ'.\n"
                "3. Для запуска команды выведи строго <bash>команда</bash> без лишнего текста. "
                "Когда реальные факты получены — дай чёткий итоговый ответ."
            )

            prompt = (
                f"{tool_instruction}\n\n"
                f"{context_hint}\n\n"
                f"Твоя подзадача:\n{job.description}"
            )

            response = await self.client.generate(
                model=self.executor_model,
                prompt=prompt,
                system_prompt=job.system_prompt,
                temperature=job.temperature,
                num_ctx=self.context_window,
            )

            for step in range(1, 3):
                bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
                if not bash_match:
                    break

                self._check_cancelled()
                cmd = bash_match.group(1).strip()
                print(f"  [Shell #{job.subtask_id}:{job.role_name}] Шаг {step}: {cmd[:90]}...")
                shell_out = await ShellTool.run(
                    cmd,
                    workspace_dir=str(self.workspace_dir),
                    timeout=20.0,
                )

                is_ok, fail_reason = self._is_command_output_useful(shell_out)
                if is_ok:
                    if cmd not in self.session_working_commands:
                        self.session_working_commands.append(cmd)
                else:
                    self._record_session_failure(cmd, fail_reason)

                followup_prompt = (
                    f"Выполнена команда:\n{cmd}\n\n"
                    f"Вывод песочницы:\n{shell_out}\n\n"
                    f"Если получены ссылки через web_search.py — передай ОДИН лучший реальный URL в web_parser.py через <bash>...</bash>. "
                    f"Если факты уже получены — дай точный итоговый ответ по подзадаче: {job.description}"
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

    def _build_session_context(self, memory_context: str, last_feedback: str) -> str:
        sections = []
        if memory_context.strip():
            sections.append(f"=== ПРОВЕРЕННЫЙ ОПЫТ ИЗ БАЗЫ НАВЫКОВ ===\n{memory_context.strip()}")

        if self.session_working_commands:
            working_list = "\n".join(f"- {c[:140]}" for c in self.session_working_commands[-8:])
            sections.append(
                "=== КОМАНДЫ, УЖЕ ДАВШИЕ НЕНУЛЕВОЙ ВЫВОД В ЭТОЙ СЕССИИ ===\n"
                f"{working_list}"
            )

        if self.session_failed_attempts:
            failed_list = "\n".join(f"- ⛔ {f}" for f in self.session_failed_attempts[-15:])
            sections.append(
                "=== НЕУДАЧНЫЕ ПОПЫТКИ ТЕКУЩЕЙ ЗАДАЧИ (ПОВТОРЯТЬ ЗАПРЕЩЕНО!) ===\n"
                f"{failed_list}\n"
                "ОБЯЗАТЕЛЬНО выбери другой источник, другой домен или другой инструмент!"
            )

        if last_feedback.strip():
            sections.append(f"=== ВЕРДИКТ КРИТИКА ПО ПРОШЛОЙ ИТЕРАЦИИ ===\n{last_feedback.strip()}")

        return "\n\n".join(sections)

    async def _synthesize_results(self, main_task: str, results: List[Dict[str, Any]]) -> str:
        self._check_cancelled()
        grouped = "\n---\n".join(
            f"Подзадача #{r['subtask_id']} ({r['description']}) [Роль: {r['role']}]:\n{r['output']}"
            for r in results
        )
        prompt = (
            f"Главная задача пользователя: {main_task}\n\n"
            f"Ниже приведены отчёты агентов:\n{grouped}\n\n"
            f"ЖЁСТКОЕ ТРЕБОВАНИЕ: Отбрось все неудачные попытки, ошибки доступа, догадки и советы 'сделайте сами'. "
            f"Собери в финальный ответ ТОЛЬКО реальные фактические данные, напрямую решающие главную задачу."
        )
        return await self.client.generate(
            model=self.executor_model,
            prompt=prompt,
            system_prompt="Ты технический лидер. Собираешь только доказанные факты и рабочие решения.",
            temperature=0.1,
            num_ctx=self.context_window,
        )

    async def _execute_iteration_pipeline(
        self,
        task_text: str,
        iteration_label: str,
        memory_context: str,
        last_feedback: str,
    ) -> str:
        self._check_cancelled()
        print(f"\n=== {iteration_label} | Фаза 1: Планирование ({self.executor_model}) ===")

        full_context = self._build_session_context(memory_context, last_feedback)
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
        return await self._synthesize_results(task_text, raw_results)
    async def run(self, task_text: str) -> str:
        last_feedback = ""
        synthesized_solution = ""
        fast_verified = False

        # Сброс временной памяти текущей задачи
        self.session_working_commands = []
        self.session_failed_attempts = []
        print(f"[Workspace] Изолированная папка сессии: {self.workspace_dir}")

        memory_context = ""
        try:
            past = await self.memory.search(task_text)
            if past:
                print("[Memory] Загружены проверенные решения похожих задач.")
                memory_context = past
        except Exception as e:
            print(f"[Memory WARN] Поиск в памяти пропущен: {e}")

        # ЭТАП 1: Поиск решения без переключения VRAM с накоплением временного журнала ошибок
        for iteration in range(1, self.max_iterations + 1):
            synthesized_solution = await self._execute_iteration_pipeline(
                task_text=task_text,
                iteration_label=f"Итерация {iteration}/{self.max_iterations}",
                memory_context=memory_context,
                last_feedback=last_feedback,
            )

            print(f"=== Фаза 4.1: Быстрый аудит без смены VRAM ({self.executor_model}) ===")
            verdict = await self.critic.evaluate(
                task_text,
                synthesized_solution,
                model_override=self.executor_model,
            )

            status = str(verdict.get("status", "RETRY")).upper()
            critique_feedback = str(verdict.get("feedback", "Не получены фактические данные.")).strip()
            dead_end = str(verdict.get("dead_end", "")).strip()
            next_strategy = str(verdict.get("next_strategy", "")).strip()

            if status == "SUCCESS":
                print(f"[FAST-PASS] Предварительная проверка пройдена: {critique_feedback}")
                fast_verified = True
                break

            if dead_end:
                self._record_session_failure(f"Подход итерации {iteration}", dead_end)

            last_feedback = (
                f"Причина отклонения (итерация {iteration}): {critique_feedback}\n"
                f"Тупиковый путь: {dead_end or 'не указан'}\n"
                f"Что сделать по-другому: {next_strategy or 'найти другой источник/инструмент'}"
            ).strip()
            print(f"[RETRY] Критик отклонил ответ: {critique_feedback}")
            if next_strategy:
                print(f"[SEARCH ALT] Новый вектор поиска: {next_strategy}")

        # ЭТАП 2: Финальный глубокий аудит (1 загрузка critic_model в самом конце)
        if fast_verified:
            self._check_cancelled()
            if self.critic_model and self.critic_model != self.executor_model:
                print(f"\n=== Фаза 4.2: Финальный глубокий аудит ({self.critic_model}) ===")
                final_verdict = await self.critic.evaluate(
                    task_text,
                    synthesized_solution,
                    model_override=self.critic_model,
                )
                final_status = str(final_verdict.get("status", "RETRY")).upper()
                final_feedback = str(final_verdict.get("feedback", "")).strip()

                if final_status != "SUCCESS":
                    print(f"[FINAL-CRITIC REJECT] Глубокий аудит выявил недочёт: {final_feedback}")
                    dead_end = str(final_verdict.get("dead_end", "")).strip() or final_feedback
                    self._record_session_failure("Финальный аудит", dead_end)

                    synthesized_solution = await self._execute_iteration_pipeline(
                        task_text=task_text,
                        iteration_label="Финальная доработка после глубокого аудита",
                        memory_context=memory_context,
                        last_feedback=f"Замечания финального аудитора: {final_feedback}",
                    )
                    recheck = await self.critic.evaluate(
                        task_text,
                        synthesized_solution,
                        model_override=self.executor_model,
                    )
                    if str(recheck.get("status", "RETRY")).upper() != "SUCCESS":
                        fast_verified = False
                else:
                    print(f"[SUCCESS] Финальный аудитор утвердил решение: {final_feedback}")

        # В постоянную память и навыки сохраняем ТОЛЬКО подтверждённый успех
        if fast_verified:
            try:
                await self.memory.save(
                    task_text,
                    synthesized_solution,
                    executed_commands=self.session_working_commands,
                    is_success=True,
                )
                print("[Memory] Победившее решение и рабочие инструменты сохранены в базу навыков.")
            except Exception as e:
                print(f"[Memory WARN] Не удалось сохранить опыт: {e}")

            # Очищаем временный журнал ошибок сессии
            self.session_failed_attempts.clear()
            return synthesized_solution

        print("[FAILED] Задача не решена. Временный журнал ошибок очищен, постоянная память не засорена.")
        failed_summary = "\n".join(f"- ⛔ {item}" for item in self.session_failed_attempts[-8:])
        self.session_failed_attempts.clear()

        return (
            f"## Статус: Задача не выполнена за {self.max_iterations} итераций\n\n"
            f"Система отклонила неподтверждённые ответы и не сохранила неудачный запуск в базу навыков.\n\n"
            f"### Проверенные в сессии тупиковые пути:\n{failed_summary or '- Нет данных'}"
        )


Orchestrator = BatchOrchestrator
