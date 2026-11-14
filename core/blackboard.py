import re
from typing import List, Dict
from urllib.parse import urlparse


class SessionBlackboard:
    """Оперативная доска прогресса, находок и тупиковых путей текущей задачи."""

    def __init__(self):
        self.working_commands: List[str] = []
        self.failed_attempts: List[str] = []
        self.discovered_urls: List[str] = []
        self.extracted_facts: List[str] = []
        self.blocked_domains: List[str] = []
        self.command_cache: Dict[str, str] = {}

    def clear(self) -> None:
        self.working_commands.clear()
        self.failed_attempts.clear()
        self.discovered_urls.clear()
        self.extracted_facts.clear()
        self.blocked_domains.clear()
        self.command_cache.clear()

    @staticmethod
    def extract_domain(url_or_domain: str) -> str:
        raw = url_or_domain.strip().lower()
        if not raw:
            return ""
        if "://" not in raw:
            raw = "https://" + raw
        host = urlparse(raw).netloc.split(":")
        return host[4:] if host.startswith("www.") else host

    def block_domain(self, domain_or_url: str) -> None:
        domain = self.extract_domain(domain_or_url)
        if not domain or len(domain) < 4:
            return
        if domain not in self.blocked_domains:
            self.blocked_domains.append(domain)
            print(f"[Blackboard] Домен '{domain}' признан нерелевантным и вычеркнут из сессии.")

        self.discovered_urls = [
            u for u in self.discovered_urls
            if domain not in self.extract_domain(u)
        ]
        self.extracted_facts = [
            fact for fact in self.extracted_facts
            if domain not in fact.lower()
        ]

    def find_blocked_domain_in_command(self, cmd: str) -> str:
        lowered = cmd.lower()
        for domain in self.blocked_domains:
            if domain in lowered:
                return domain
        return ""

    def record_failure(self, cmd: str, reason: str) -> None:
        short_cmd = cmd.strip().replace("\n", " ")[:120]
        short_reason = reason.strip().replace("\n", " ")[:140]
        entry = f"Команда `{short_cmd}` -> ОШИБКА: {short_reason}"
        if entry not in self.failed_attempts:
            self.failed_attempts.append(entry)

    def harvest_output(self, cmd: str, shell_out: str) -> tuple[bool, str]:
        if "[EXIT CODE]: 0" not in shell_out or "[STDOUT]:" not in shell_out:
            return False, "ошибка выполнения или пустой вывод"

        stdout_part = shell_out.split("[STDOUT]:", 1).strip()
        if len(stdout_part) < 25:
            return False, "слишком короткий ответ"

        lowered = stdout_part.lower()
        if any(m in lowered for m in ("страница не найдена", "404 not found", "результатов не найдено")):
            return False, "страница 404 или пустой поиск"

        if cmd not in self.working_commands:
            self.working_commands.append(cmd)

        for u in re.findall(r"https?://[^\s\"'<>)\]]+", stdout_part):
            clean_u = u.rstrip(".,;:")
            host = self.extract_domain(clean_u)
            if any(b in host for b in self.blocked_domains):
                continue
            if clean_u not in self.discovered_urls and len(self.discovered_urls) < 35:
                self.discovered_urls.append(clean_u)

        if "web_parser.py" in cmd and ("=== СТРУКТУРИРОВАННЫЕ ДАННЫЕ" in stdout_part or "=== ТЕКСТОВЫЕ ДАННЫЕ ===" in stdout_part):
            snippet = stdout_part[:1800]
            if snippet not in self.extracted_facts and len(self.extracted_facts) < 6:
                self.extracted_facts.append(snippet)

        return True, ""

    def apply_critic_rejection(self, verdict: dict, iteration: int) -> str:
        irrelevant = verdict.get("irrelevant_domains") or []
        if isinstance(irrelevant, list):
            for dom in irrelevant:
                if isinstance(dom, str) and dom.strip():
                    self.block_domain(dom)

        feedback = str(verdict.get("feedback", "Не получены фактические данные.")).strip()
        dead_end = str(verdict.get("dead_end", "")).strip()
        next_strategy = str(verdict.get("next_strategy", "")).strip()

        scan_text = f"{dead_end} {feedback}"
        if any(w in scan_text.lower() for w in ("не связан", "не относится", "нерелевант", "другой теме", "не про")):
            for dom in re.findall(r"(?:https?://)?(?:www\.)?([a-z0-9.-]+\.[a-z]{2,})", scan_text.lower()):
                self.block_domain(dom)

        if dead_end:
            self.record_failure(f"Подход итерации {iteration}", dead_end)

        return (
            f"Причина отклонения (итерация {iteration}): {feedback}\n"
            f"Тупиковый путь: {dead_end or 'не указан'}\n"
            f"Что сделать по-другому: {next_strategy or 'использовать другой релевантный источник'}"
        ).strip()

    def build_prompt_context(self, memory_context: str, last_feedback: str) -> str:
        sections = []
        if memory_context.strip():
            sections.append(f"=== ПРОВЕРЕННЫЙ ОПЫТ ИЗ БАЗЫ НАВЫКОВ ===\n{memory_context.strip()}")

        if self.blocked_domains:
            sections.append(
                "=== 🚫 ВЫЧЕРКНУТЫЕ НЕРЕЛЕВАНТНЫЕ ДОМЕНЫ (ИСПОЛЬЗОВАТЬ ЗАПРЕЩЕНО!) ===\n"
                f"{', '.join(self.blocked_domains)}\n"
                "Эти сайты не соответствуют теме задачи. Исключи их из запросов и ищи другие источники!"
            )

        if self.discovered_urls:
            urls_list = "\n".join(f"- {u}" for u in self.discovered_urls[-25:])
            sections.append(f"=== УЖЕ НАЙДЕННЫЕ В ЭТОЙ СЕССИИ РЕАЛЬНЫЕ URL ===\n{urls_list}")

        if self.extracted_facts:
            facts_blob = "\n---\n".join(self.extracted_facts[-3:])
            sections.append(f"=== УЖЕ ИЗВЛЕЧЁННЫЕ СО СТРАНИЦ ДАННЫЕ ТЕКУЩЕЙ СЕССИИ ===\n{facts_blob[:3500]}")

        if self.failed_attempts:
            failed_list = "\n".join(f"- ⛔ {f}" for f in self.failed_attempts[-15:])
            sections.append(f"=== НЕУДАЧНЫЕ ПОПЫТКИ ТЕКУЩЕЙ ЗАДАЧИ (ПОВТОРЯТЬ ЗАПРЕЩЕНО!) ===\n{failed_list}")

        if last_feedback.strip():
            sections.append(f"=== ВЕРДИКТ КРИТИКА ПО ПРОШЛОЙ ИТЕРАЦИИ ===\n{last_feedback.strip()}")

        return "\n\n".join(sections)
