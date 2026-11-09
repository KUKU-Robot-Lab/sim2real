"""런 기록(종류 S) 경로 — 한 곳에서만 정한다(logs/README.md '새 코드를 쓸 때').

    logs/runs/<YYMMDD>/<종류>/<HHMMSS>_<label>/     런마다 폴더
    logs/runs/<YYMMDD>/<종류>/<파일>                  하루 · 런마다 파일 하나

날짜는 런을 시작한 날(현지 시각)이다. 자정을 넘겨도 한 런의 파일은 시작한 날 폴더에 남는다.
경로는 repo 루트 기준 절대 경로다(cwd 와 무관).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # scripts/run_paths.py
LOGS = REPO / "logs"
RUNS = LOGS / "runs"
DAY_FMT = "%y%m%d"
TIME_FMT = "%H%M%S"


def day_dir(kind: str, day: datetime | str | None = None) -> Path:
    """logs/runs/<YYMMDD>/<kind> — 만들지는 않는다. day 는 datetime 이나 'YYMMDD'."""
    if not isinstance(day, str):
        day = (day or datetime.now()).strftime(DAY_FMT)
    return RUNS / day / kind


def unique_dir(path: Path, *, mkdir: bool = True) -> Path:
    """path 가 이미 있으면 _2, _3 … 을 붙인다. mkdir 이면 만들면서 잡는다(동시 실행에도 덮지 않음)."""
    cand, k = path, 1
    while True:
        if mkdir:
            try:
                cand.mkdir(parents=True)
                return cand
            except FileExistsError:
                pass
        elif not cand.exists():
            return cand
        k += 1
        cand = path.with_name(f"{path.name}_{k}")


def run_dir(kind: str, label: str, *, now: datetime | None = None, mkdir: bool = True) -> Path:
    """logs/runs/<YYMMDD>/<kind>/<HHMMSS>_<label>, 같은 초에 겹치면 _2, _3."""
    now = now or datetime.now()
    name = f"{now.strftime(TIME_FMT)}_{label}" if label else now.strftime(TIME_FMT)
    return unique_dir(day_dir(kind, now) / name, mkdir=mkdir)


def run_time(path: Path) -> datetime | None:
    """logs/runs/<YYMMDD>/<kind>/<HHMMSS>_… 의 시작 시각. 형식이 아니면 None."""
    path = Path(path)
    try:
        return datetime.strptime(path.parent.parent.name + path.name[:6], DAY_FMT + TIME_FMT)
    except ValueError:
        return None


def find_runs(kind: str, pattern: str = "*") -> list[Path]:
    """모든 날의 logs/runs/*/<kind>/<pattern>, (날짜, 이름) 순."""
    return sorted(RUNS.glob(f"[0-9]*/{kind}/{pattern}"), key=lambda p: (p.parent.parent.name, p.name))


def latest(kind: str, pattern: str = "*") -> Path | None:
    """find_runs 의 마지막(가장 최근) 하나. 없으면 None."""
    runs = find_runs(kind, pattern)
    return runs[-1] if runs else None
