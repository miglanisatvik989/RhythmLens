"""Legacy alias: GPU diagnostics now live in configured doctor/preflight."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/doctor_environment.py"),
        run_name="__main__",
    )
