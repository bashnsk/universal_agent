import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Any, Callable
from agents.base import OllamaClient
from core.blackboard import SessionBlackboard
from tools.shell import ShellTool


@dataclass
class SubtaskJob:
    subtask_id: int
    description: str
    role_name: str
    system_prompt: str
    temperature: float


class SubtaskWorker:
    """Исполнитель одной сессии (подзадача × роль) с bash-песочницей и кэшем команд сессии."""

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        context_window: int,
        workspace_dir: str,
        blackboard: SessionBlackboard,
        semaphore: asyncio.Semaphore,
        cancel_checker: Callable[[], None],
    ):
        self.client = client
        self.model = model
        self.context_window = context_window
        self.workspace_dir = workspace_dir
        self.blackboard = blackboard
        self.semaphore = semaphore
        self._check_cancelled = cancel_checker

    @staticmethod
    def _normalize_command(raw_cmd: str) -> str:
        cmd = raw_cmd.strip().strip("`")
        if re.match(r"^['\"]?https?://\S+['\"]?$", cmd):
            clean_url = cmd.strip("'\"")
            return f'python3 /home/sid/RAID/_AI/universal_agent/skills/web_parser.py "{clean_url}"'
        return cmd

    @staticmethod
    def _is_placeholder_command(cmd: str) -> bool:
        lowered = cmd.lower()
        return bool(
            re.search(r'web_search\.py\s+["\']запрос["\']', lowered)
            or "example.com" in lowered
            or "/api/new-" in lowered
        )

    async def execute(self, job: SubtaskJob, context_hint: str) -> Dict[str, Any]:
        async with self.semaphore:
            self._check_cancelled()
            print(f"[Queue -> GPU] Подзадача #{job.subtask_id} | Роль: {job.role_name}")

            tool_instruction = (
                "У тебя есть доступ к bash-песочнице Linux и готовым Python-инструментам:\n"
                "- Поиск ссылок в интернете: python3 /home/sid/RAID/_AI/universal_agent/skills/web_search.py \"<ключевые_слова_по_задаче>\" --limit 5\n"
                "- Парсинг одной страницы (BeautifulSoup + SPA JSON): python3 /home/sid/RAID/_AI/universal_agent/skills/web_parser.py \"<реальный_https_url>\"\n\n"
                "УНИВЕРСАЛЬНЫЕ ПРАВИЛА:\n"
                "1. Если в блоке 'УЖЕ ИЗВЛЕЧЁННЫЕ СО СТРАНИЦ ДАННЫЕ' уже есть нужная информация — сразу используй её для ответа без повторного парсинга!\n"
                "2. Если в блоке 'УЖЕ НАЙДЕННЫЕ В ЭТОЙ СЕССИИ URL' есть релевантный сайт — парси его через web_parser.py.\n"
                "3. Запрещено выдумывать несуществующие /api/... пути, обращаться к example.com или к вычеркнутым доменам.\n"
                "4. Для запуска команды выведи строго <bash>команда</bash> без лишнего текста."
            )

            prompt = (
                f"{tool_instruction}\n\n"
                f"{context_hint}\n\n"
                f"Твоя подзадача:\n{job.description}"
            )

            response = await self.client.generate(
                model=self.model,
                prompt=prompt,
                system_prompt=job.system_prompt,
                temperature=job.temperature,
                num_ctx=self.context_window,
            )

            for step in range(1, 4):
                bash_match = re.search(r"<bash>(.*?)</bash>", response, re.DOTALL)
                if not bash_match:
                    break

                self._check_cancelled()
                cmd = self._normalize_command(bash_match.group(1))
                blocked_hit = self.blackboard.find_blocked_domain_in_command(cmd)

                if self._is_placeholder_command(cmd):
                    shell_out = "[-] Ошибка: Использован вымышленный URL или шаблон. Используй web_search.py для поиска реальных ссылок."
                elif blocked_hit:
                    shell_out = f"[-] Ошибка: Домен '{blocked_hit}' вычеркнут как нерелевантный. Используй другой источник."
                elif cmd in self.blackboard.command_cache:
                    shell_out = self.blackboard.command_cache[cmd]
                else:
                    print(f"  [Shell #{job.subtask_id}:{job.role_name}] Шаг {step}: {cmd[:95]}...")
                    shell_out = await ShellTool.run(
                        cmd,
                        workspace_dir=self.workspace_dir,
                        timeout=20.0,
                    )
                    self.blackboard.command_cache[cmd] = shell_out
                    is_ok, fail_reason = self.blackboard.harvest_output(cmd, shell_out)
                    if not is_ok:
                        self.blackboard.record_failure(cmd, fail_reason)

                followup_prompt = (
                    f"Выполнена команда:\n{cmd}\n\n"
                    f"Вывод песочницы:\n{shell_out}\n\n"
                    f"Если получены релевантные ссылки — распарси нужную страницу через web_parser.py. "
                    f"Если факты уже извлечены — дай точный итоговый ответ по подзадаче: {job.description}"
                )
                response = await self.client.generate(
                    model=self.model,
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
