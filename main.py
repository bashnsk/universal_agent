import asyncio
import sys
import yaml
from pathlib import Path
from core.orchestrator import Orchestrator


def load_config():
    config_path = Path("/home/sid/RAID/_AI/universal_agent/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Дефолтный конфиг под вашу систему
    return {
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "qwen2.5:7b-instruct",
            "critic_model": "llama3.1:8b-instruct",
            "timeout": 120,
        },
        "execution": {"max_iterations": 4, "multi_role_count": 3},
    }


async def main():
    if len(sys.argv) < 2:
        print('Использование: python main.py "Ваша задача для выполнения"')
        sys.exit(1)

    task = sys.argv[1]
    config = load_config()

    orchestrator = Orchestrator(config)
    result = await orchestrator.run(task)

    output_dir = Path("/home/sid/RAID/_AI/universal_agent/storage/tasks")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / "last_result.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\n[+] Результат сохранен в {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
