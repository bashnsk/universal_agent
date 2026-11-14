import asyncio
import shlex

class ShellTool:
    # Запрещенные утилиты (только как отдельное имя команды)
    FORBIDDEN_COMMANDS = {"top", "htop", "rm -rf /", "shutdown", "reboot", "poweroff"}

    @staticmethod
    async def run(command: str) -> str:
        cmd_trimmed = command.strip()
        
        # Точная проверка вызова опасных системных утилит
        first_word = cmd_trimmed.split()[0] if cmd_trimmed.split() else ""
        if first_word in ShellTool.FORBIDDEN_COMMANDS or cmd_trimmed in ShellTool.FORBIDDEN_COMMANDS:
            return f"[-] Ошибка безопасности: Команда '{first_word}' заблокирована песочницей."

        if "rm -rf /" in cmd_trimmed:
            return "[-] Ошибка безопасности: Команда опасна и заблокирована."

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                return "[-] Ошибка: Превышен таймаут выполнения команды (30 сек)."

            output = ""
            if stdout:
                output += f"[STDOUT]:\n{stdout.decode('utf-8', errors='ignore')}\n"
            if stderr:
                output += f"[STDERR]:\n{stderr.decode('utf-8', errors='ignore')}\n"
            
            return output if output else "[Команда выполнена успешно без вывода]"

        except Exception as e:
            return f"[-] Ошибка выполнения команды: {str(e)}"