"""run_paths — 런 기록 경로 logs/runs/<YYMMDD>/<종류>/<HHMMSS>_<label>(logs/README.md)."""
from __future__ import annotations

from datetime import datetime

import pytest

import run_paths

NOW = datetime(2026, 11, 9, 10, 15, 0)


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_paths, "RUNS", tmp_path)
    return tmp_path


def test_day_dir_puts_the_date_first(runs):
    assert run_paths.day_dir("cpu", NOW) == runs / "261109" / "cpu"
    assert run_paths.day_dir("cpu", "261108") == runs / "261108" / "cpu"
    assert not (runs / "261109").exists()                       # 만들지 않는다


def test_run_dir_names_by_time_and_suffixes_a_collision(runs):
    a = run_paths.run_dir("episode", "cocktail_1c_e4000", now=NOW)
    b = run_paths.run_dir("episode", "cocktail_1c_e4000", now=NOW)
    c = run_paths.run_dir("episode", "cocktail_1c_e4000", now=NOW)
    assert a == runs / "261109" / "episode" / "101500_cocktail_1c_e4000" and a.is_dir()
    assert (b.name, c.name) == ("101500_cocktail_1c_e4000_2", "101500_cocktail_1c_e4000_3")
    d = run_paths.run_dir("episode", "x", now=NOW, mkdir=False)
    assert not d.exists() and d.name == "101500_x"
    assert run_paths.run_time(a) == NOW


def test_find_runs_spans_days_and_latest_is_the_newest(runs):
    old = run_paths.run_dir("tilt", "pour_v12", now=datetime(2026, 11, 8, 23, 59, 0))
    new = run_paths.run_dir("tilt", "pour_v12", now=datetime(2026, 11, 9, 0, 1, 0))
    run_paths.run_dir("tilt", "setting_v12", now=NOW)
    run_paths.run_dir("episode", "pour_v12", now=NOW)           # 다른 종류는 섞이지 않는다
    assert run_paths.find_runs("tilt", "*_pour_*") == [old, new]
    assert run_paths.latest("tilt", "*_pour_*") == new
    assert run_paths.latest("tilt").name == "101500_setting_v12"
    assert run_paths.latest("mission") is None
