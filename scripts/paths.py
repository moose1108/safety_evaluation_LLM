"""Repo-root relative paths for pipeline scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = ROOT / "benchmarks"
RESPONSES = ROOT / "responses"
JUDGMENTS = ROOT / "judgments"
ANALYSIS = ROOT / "analysis"
FAILURES = ANALYSIS / "failures"
SCORES = ANALYSIS / "scores"
ARCHIVE = ROOT / "archive"
