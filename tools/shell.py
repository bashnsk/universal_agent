import asyncio
import os
import re
from pathlib import Path
from typing import Optional


class ShellTool:
    FORBIDDEN_COMMANDS = {
        "top", "htop", "vim", "vi", "nano", "less", "more", "man",
        "shutdown", "reboot", "poweroff", "init", "mkfs", "dd", "fdisk"
    }

    FORBIDDEN_PATTERNS = [
        r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/",
        r":\(\)\{\s*:\|\:&\s*\};:",
        r">\s*/dev/sd",
    ]

    DEFAULT_WORKSPACE = Path("/home/sid/RAID/_AI/universal_agent/storage/workspace")

    @classmethod
    def _sanitize_command(cls, command: str) -> str:
        cmd = command.strip()
        cmd = re.sub(r"\bapt-get\s+install\b(?!.*-y)", "apt-get install -y", cmd)
        cmd = re.sub(r"\bapt\s+install\b(?!.*-y)", "apt install -y", cmd)
        cmd = re.sub(r"\bpip\s+install\b(?!.*--no-input)", "pip install --no-input", cmd)
        return cmd

    @classmethod
    async def run(
        cls,
        command: str,
        workspace_dir: Optional[str] = None,
        timeout: float = 45.0,
        max_output_chars: int = 8000,
    ) -> str:
        cmd_trimmed = command.strip()
        if not cmd_trimmed:
            return "[-] Ошибка: Пустая команда."

        first_word = os.path.basename(cmd_trimmed.split()[0])
        if first_word in cls.FORBIDDEN_COMMANDS:
            return f"[-] Ошибка безопасности: Команда '{first_word}' заблокирована."

        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, cmd_trimmed):
                return "[-] Ошибка безопасности: Обнаружен опасный паттерн."

        work_path = Path(workspace_dir) if workspace_dir else cls.DEFAULT_WORKSPACE
        work_path.mkdir(parents=True, exist_ok=True)

        safe_cmd = cls._sanitize_command(cmd_trimmed)
        env = os.environ.copy()
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["PYTHONUNBUFFERED"] = "1"
        venv_bin = "/home/sid/RAID/_AI/universal_agent/venv/bin"
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

        try:
            process = await asyncio.create_subprocess_shell(
                safe_cmd,
                cwd=str(work_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return f"[-] Ошибка: Превышен таймаут ({int(timeout)} сек)."

            out_str = stdout.decode("utf-8", errors="ignore").strip()
            err_str = stderr.decode("utf-8", errors="ignore").strip()

            result_parts = [f"[EXIT CODE]: {process.returncode}"]
            if out_str:
                result_parts.append(f"[STDOUT]:\n{out_str}")
            if err_str:
                result_parts.append(f"[STDERR]:\n{err_str}")

            full_output = "\n".join(result_parts)
            if len(full_output) > max_output_chars:
                full_output = (
                    full_output[:max_output_chars]
                    + f"\n... [ОБРЕЗАНО ДО {max_output_chars} СИМВОЛОВ] ..."
                )

            return full_output

        except Exception as e:
            return f"[-] Ошибка выполнения команды: {str(e)}"

    execute = run
