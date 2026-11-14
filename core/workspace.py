import shutil
from datetime import datetime
from pathlib import Path


class WorkspaceManager:
    def __init__(self, root_dir: str, keep_last: int = 15):
        self.base_dir = Path(root_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rotate(keep_last=keep_last)

        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def rotate(self, keep_last: int = 15) -> None:
        if keep_last < 1 or not self.base_dir.exists():
            return

        sessions = sorted(
            [
                p for p in self.base_dir.iterdir()
                if p.is_dir() and p.name.startswith("session_")
            ],
            key=lambda p: p.name,
        )
        excess = len(sessions) - keep_last
        if excess > 0:
            for old_dir in sessions[:excess]:
                shutil.rmtree(old_dir, ignore_errors=True)
            print(f"[Workspace] Ротация: удалено старых сессий: {excess}")
