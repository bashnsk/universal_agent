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
