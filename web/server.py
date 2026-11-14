import asyncio
from datetime import datetime
from pathlib import Path
import json
import yaml
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from core.orchestrator import Orchestrator

app = FastAPI(title="Universal AI Agent WebUI")

task_status = {"running": False, "log": "Система готова к работе.\n"}
active_orchestrator = None  # Ссылка на запущенный оркестратор для отмены

TASKS_DIR = Path("/home/sid/RAID/_AI/universal_agent/storage/tasks")
TASKS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = Path("/home/sid/RAID/_AI/universal_agent/config.yaml")
HTML_PATH = Path("/home/sid/RAID/_AI/universal_agent/web/index.html")

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "qwen2.5-coder:14b",
            "critic_model": "llama3.1:latest",
            "timeout": 120
        },
        "execution": {
            "max_iterations": 4,
            "multi_role_count": 3
        }
    }

def save_config(config_data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, allow_unicode=True)

class TaskRequest(BaseModel):
    task: str

class ConfigRequest(BaseModel):
    default_model: str
    critic_model: str
    max_iterations: int
    multi_role_count: int

async def run_agent_task(task_text: str):
    global task_status, active_orchestrator
    task_status["log"] = f"Старт задачи: {task_text}\n"
    
    class LoggerCapture:
        def write(self, message):
            task_status["log"] += message
        def flush(self):
            pass

    import sys
    original_stdout = sys.stdout
    sys.stdout = LoggerCapture()

    try:
        config = load_config()
        active_orchestrator = Orchestrator(config)
        result = await active_orchestrator.run(task_text)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_file = TASKS_DIR / f"task_{timestamp}.md"
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# Задача: {task_text}\n\n## Результат:\n\n{result}")
            
        print(f"\n[+] Задача успешно завершена! Артефакт сохранен: {out_file.name}\n")
    except asyncio.CancelledError:
        print("\n[!] Выполнение задачи было прервано пользователем.\n")
    except Exception as e:
        print(f"\n[-] Критическая ошибка выполнения: {str(e)}\n")
    finally:
        sys.stdout = original_stdout
        task_status["running"] = False
        active_orchestrator = None

@app.post("/api/run")
async def run_task(req: TaskRequest, background_tasks: BackgroundTasks):
    global task_status
    if task_status["running"]:
        return {"status": "already_running"}
    
    task_status["running"] = True
    task_status["log"] = f"Инициализация агента для задачи: {req.task}\n"
    
    background_tasks.add_task(run_agent_task, req.task)
    return {"status": "started"}

@app.post("/api/stop")
async def stop_task():
    global active_orchestrator, task_status
    if active_orchestrator:
        active_orchestrator.cancel()
        task_status["log"] += "\n[!] Получен сигнал остановки от пользователя...\n"
        return {"status": "stopping"}
    return {"status": "not_running"}

@app.get("/api/stream-logs")
async def stream_logs():
    async def event_generator():
        while True:
            current_state = {
                "running": task_status["running"],
                "log": task_status["log"]
            }
            yield f"data: {json.dumps(current_state, ensure_ascii=False)}\n\n"
            
            if not task_status["running"]:
                await asyncio.sleep(0.5)
                yield f"data: {json.dumps(current_state, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/config")
async def get_config():
    return load_config()

@app.post("/api/config")
async def update_config(req: ConfigRequest):
    config = load_config()
    config["ollama"]["default_model"] = req.default_model
    config["ollama"]["critic_model"] = req.critic_model
    config["execution"]["max_iterations"] = req.max_iterations
    config["execution"]["multi_role_count"] = req.multi_role_count
    save_config(config)
    return {"status": "success"}

@app.get("/api/tasks")
async def get_tasks():
    files = sorted(TASKS_DIR.glob("task_*.md"), reverse=True)
    return [f.name for f in files]

@app.get("/api/task/{filename}")
async def get_task_content(filename: str):
    file_path = TASKS_DIR / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    return {"content": "Файл не найден."}

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    if HTML_PATH.exists():
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI файл не найден</h1>"