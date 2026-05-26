from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class Step:
    number: int
    name: str
    description: str
    command: List[str]


def preferred_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def build_steps(python_executable: str) -> tuple[List[Step], Step, Step]:
    pipeline_steps = [
        Step(1, "data_ingestion", "Download or detect M5 files and create sampled processed inputs", [python_executable, "-m", "src.data_ingestion"]),
        Step(2, "preprocessing", "Convert wide daily sales to weekly planning demand", [python_executable, "-m", "src.preprocessing"]),
        Step(3, "hierarchy_builder", "Build hierarchy tables and hierarchical demand views", [python_executable, "-m", "src.hierarchy_builder"]),
        Step(4, "feature_engineering", "Create lag and rolling demand features", [python_executable, "-m", "src.feature_engineering"]),
        Step(5, "forecasting", "Train and compare forecasting models", [python_executable, "-m", "src.forecasting"]),
        Step(6, "reconciliation", "Reconcile forecasts across planning levels", [python_executable, "-m", "src.reconciliation"]),
        Step(7, "network_generator", "Create synthetic plants, DCs, lanes, and SKU master data", [python_executable, "-m", "src.network_generator"]),
        Step(8, "optimization", "Optimize production, shipment, inventory, and service decisions", [python_executable, "-m", "src.optimization"]),
        Step(9, "simulation", "Stress-test the plan using SimPy simulation", [python_executable, "-m", "src.simulation"]),
        Step(10, "scenario_analysis", "Compare cost and service under scenario shocks", [python_executable, "-m", "src.scenario_analysis"]),
    ]
    test_step = Step(11, "tests", "Run the unit test suite", [python_executable, "-m", "pytest"])
    dashboard_step = Step(12, "dashboard", "Launch the Streamlit dashboard", [python_executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    return pipeline_steps, test_step, dashboard_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SOPNet pipeline steps in order with beginner-friendly progress output."
    )
    parser.add_argument("--from-step", type=int, default=1, help="Start from a specific step number.")
    parser.add_argument("--to-step", type=int, default=10, help="Stop at a specific step number.")
    parser.add_argument("--include-tests", action="store_true", help="Run pytest after the pipeline.")
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Do not launch Streamlit after the pipeline completes.",
    )
    parser.add_argument("--list-steps", action="store_true", help="Show available steps and exit.")
    return parser.parse_args()


def print_header(step_python: str) -> None:
    print("\nSOPNet Runner")
    print("=" * 64)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Launcher     : {sys.executable}")
    print(f"Step Python  : {step_python}")
    print("=" * 64)


def print_steps(steps: Iterable[Step]) -> None:
    for step in steps:
        print(f"{step.number}. {step.name} - {step.description}")


def select_pipeline_steps(from_step: int, to_step: int, pipeline_steps: List[Step]) -> List[Step]:
    if from_step < 1 or to_step > 10 or from_step > to_step:
        raise ValueError("Use a valid range such as --from-step 1 --to-step 10.")
    return [step for step in pipeline_steps if from_step <= step.number <= to_step]


def run_step(step: Step) -> None:
    print(f"\n[{step.number}] Running {step.name}")
    print(f"    {step.description}")
    print(f"    Command: {' '.join(step.command)}")
    subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
    print(f"[{step.number}] Completed {step.name}")


def main() -> None:
    args = parse_args()
    step_python = preferred_python()
    pipeline_steps, test_step, dashboard_step = build_steps(step_python)
    print_header(step_python)

    if args.list_steps:
        print("Available steps:")
        print_steps(pipeline_steps + [test_step, dashboard_step])
        return

    try:
        selected_steps = select_pipeline_steps(args.from_step, args.to_step, pipeline_steps)
        print("Pipeline steps to run:")
        print_steps(selected_steps)

        for step in selected_steps:
            run_step(step)

        if args.include_tests:
            run_step(test_step)

        if not args.skip_dashboard:
            print("\nLaunching dashboard. Press Ctrl+C in this terminal window to stop Streamlit.\n")
            run_step(dashboard_step)

        print("\nSOPNet pipeline finished successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"\nStep failed with exit code {exc.returncode}.")
        print("Fix the error shown above, then resume with a command like:")
        print(f"  {sys.executable} run_project.py --from-step {max(args.from_step, 1)}")
        raise SystemExit(exc.returncode) from exc
    except ValueError as exc:
        print(f"\nConfiguration error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
