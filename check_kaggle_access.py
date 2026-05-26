from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        raise SystemExit(
            subprocess.run([str(venv_python), str(Path(__file__).resolve())], cwd=project_root).returncode
        )

    load_dotenv(dotenv_path=project_root / ".env")

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    competition = os.getenv("KAGGLE_COMPETITION", "m5-forecasting-accuracy")

    print("Kaggle Diagnostics")
    print("=" * 40)
    print(f"Project root           : {project_root}")
    print(f"KAGGLE_USERNAME loaded : {bool(username)}")
    print(f"KAGGLE_KEY loaded      : {bool(key)}")
    print(f"KAGGLE_COMPETITION     : {competition}")
    print("=" * 40)

    if not username or not key:
        print("Result: credentials are missing from .env")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:
        print(f"Result: Kaggle package is not available: {exc}")
        return

    api = KaggleApi()
    try:
        api.authenticate()
        print("Authentication: OK")
    except Exception as exc:
        print(f"Authentication: FAILED -> {exc}")
        return

    try:
        api.competition_download_files(competition, path=str(project_root / "data" / "raw" / "m5"), force=False, quiet=True)
        print("Competition access: OK")
        print("You can run: py .\\run_project.py")
    except Exception as exc:
        message = str(exc)
        if "401" in message or "Unauthorized" in message:
            print("Competition access: DENIED")
            print("Next step: open the M5 competition page in your Kaggle account and accept/join the competition rules.")
            print("Then rerun this file and, after it passes, run: py .\\run_project.py")
        else:
            print(f"Competition access: FAILED -> {exc}")


if __name__ == "__main__":
    main()
