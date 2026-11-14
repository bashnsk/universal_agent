import asyncio
import re
from agents.base import OllamaClient
from core.planner import TaskPlanner
from core.critic import CriticAgent
from core.memory import AgentMemory
from tools.shell import ShellTool

class Orchestrator:
    def __init__(self, config: dict):
        self.client = OllamaClient(base_url=config["ollama"]["base_url"])
        self.model = config["ollama"]["default_model"]
        self.critic_model = config["ollama"]["critic_model"]
        self.max_iterations = config["execution"]["max_iterations"]
        self.multi_role_count = config["execution"]["multi_role_count"]

        self.planner = TaskPlanner(self.client, self.model)
        self.critic = CriticAgent(self.client, self.critic_model)
        self.memory = AgentMemory()
        self.is_cancelled = False  # Флаг отмены

    def cancel(self):
        self.is_cancelled = True

    async def execute_subtask_with_roles(self, subtask: str) -> str:
        if self.is_cancelled:
            raise asyncio.CancelledError("Задача была отменена пользователем.")

        system_base = (
            "Ты универсальный ИИ-агент с доступом к Debian Linux и поисковому модулю DuckDuckGo. "
            "Для поиска актуальной информации в сети используй быстрый однострочник на Python через duckduckgo_search, например:\n"
            "```bash\n"
            "python3 -c \"from duckduckgo_search import DDGS; print([r['title'] + ' - ' + r['body'] for r in DDGS().text('Европа Плюс топ хитов', max_results=5)])\"\n"
            "```\n"
            "Если нужен парсинг страницы, используй Python urllib с заголовком User-Agent: Mozilla/5.0. "
            "Все bash-команды пиши внутри блоков ```bash ... ```."
        )

        roles = [
            f"{system_base} Ты экспертный исследователь и аналитик данных.",
            f"{system_base} Ты системный инженер, умеющий работать с консольными утилитами.",
            f"{system_base} Ты практик, ориентированный на точный результат."
        ]
        
        tasks = []
        for i in range(min(self.multi_role_count, len(roles))):
            tasks.append(
                self.client.generate(self.model, f"Выполни подзадачу: {subtask}", system_prompt=roles[i])
            )
        
        results = await asyncio.gather(*tasks)
        
        if self.is_cancelled:
            raise asyncio.CancelledError("Задача была отменена пользователем.")

        processed_results = []
        for res in results:
            bash_matches = re.findall(r'```bash(.*?)```', res, re.DOTALL)
            if bash_matches:
                cmd = bash_matches[0].strip()
                print(f"[🛠️ Tool Execution] Выполнение команды от агента: {cmd}")
                tool_output = await ShellTool.run(cmd)
                print(f"[🛠️ Tool Output]: {tool_output[:300]}...")
                
                final_prompt = f"Подзадача: {subtask}\n\nКоманда:\n{cmd}\n\nРезультат выполнения из консоли:\n{tool_output}\n\nСформируй финальный ответ на основе этих данных."
                res = await self.client.generate(self.model, final_prompt, system_prompt=roles[0])
            
            processed_results.append(res)

        synth_prompt = f"Синтезируй лучшие части из следующих решений для подзадачи '{subtask}':\n\n"
        for idx, res in enumerate(processed_results):
            synth_prompt += f"--- Вариант {idx+1} ---\n{res}\n\n"
        
        final_subtask_result = await self.client.generate(
            self.model, 
            synth_prompt, 
            system_prompt="Ты интегратор. Объедини лучшие решения в одно идеальное финальное решение подзадачи."
        )
        return final_subtask_result

    async def run(self, main_task: str):
        print(f"[+] Старт задачи: {main_task}")
        
        cached_solution = await self.memory.find_similar(main_task)
        context = ""
        if cached_solution:
            print("[+] Контекст извлечен из памяти успешных задач прошлого опыта.")
            context = f"Найден похожий успешный опыт из прошлого:\n{cached_solution}\n\n"

        for iteration in range(1, self.max_iterations + 1):
            if self.is_cancelled:
                raise asyncio.CancelledError("Задача была отменена пользователем.")

            print(f"\n--- Итерация {iteration}/{self.max_iterations} ---")
            
            subtasks = await self.planner.decompose(main_task, context)
            print(f"[+] Декомпозиция на подзадачи: {subtasks}")

            subtask_results = []
            for st in subtasks:
                if self.is_cancelled:
                    raise asyncio.CancelledError("Задача была отменена пользователем.")
                print(f" -> Выполнение подзадачи (в {self.multi_role_count} ролях): {st}")
                res = await self.execute_subtask_with_roles(st)
                subtask_results.append(f"Подзадача: {st}\nРезультат:\n{res}")

            accumulated = "\n\n".join(subtask_results)
            
            if self.is_cancelled:
                raise asyncio.CancelledError("Задача была отменена пользователем.")

            print("[+] Запуск проверки критиком...")
            evaluation = await self.critic.evaluate(main_task, accumulated)
            
            print(f"[+] Вердикт критика: {evaluation.get('status')}")
            print(f"[+] Отзыв: {evaluation.get('feedback')}")

            if evaluation.get("status") == "SUCCESS":
                print("[+] Задача успешно выполнена и прошла верификацию!")
                await self.memory.save_success(main_task, accumulated)
                return accumulated
            else:
                context = f"Предыдущая попытка не прошла проверку. Замечания критика: {evaluation.get('feedback')}"

        print("[-] Достигнут лимит итераций. Задача не завершена идеально.")
        return accumulated
