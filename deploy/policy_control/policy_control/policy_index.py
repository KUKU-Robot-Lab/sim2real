"""`deploy/policies/INDEX.md` — 정책 설명 · 계열별 학습 조건 차이 · 개별 실행법 (rclpy·torch 무의존).

10.04 사용자: "오늘은 실기 테스트, 나중에 비교 분석을 위해 학습 정책들을 개별 실행할 수도 있어서 차이점이 정리되어야".
값은 손으로 옮겨 적지 않는다 — 각 정책 폴더의 `params/env.yaml`(학습 런 덤프) · `policy.yaml`(summary · eval · note · deploy)
· 계약 · 미션 yaml 에서 읽는다. `deploy/policy_control/tools/policies.py --write-index` 가 부른다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import yaml

from policy_control import policy_registry as R

Row = Mapping[str, str]


class _Loader(yaml.SafeLoader):
    """학습 덤프의 !!python/tuple · slice 를 읽는다(값 비교에 필요한 만큼만)."""


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, n: tuple(ld.construct_sequence(n)))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda ld, suffix, n: None)


def load_env(e: R.Entry) -> dict:
    rel = next((r for r in R._params_of(e.path, e.checkpoint) if r.endswith("env.yaml")), "params/env.yaml")
    try:
        return yaml.load((e.path / rel).read_text(), Loader=_Loader) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _contract(e: R.Entry) -> dict:
    try:
        return json.loads((e.path / e.contract).read_text()) if e.contract else {}
    except (OSError, ValueError):
        return {}


# ── 칸: (env, card, contract) → 글자 ────────────────────────────────────────
def _reward(env: Mapping, card: Mapping, _c: Mapping) -> str:
    parts = Path(str(env.get("reward_code_path") or "")).parts
    return "/".join(parts[-3:-1]) if len(parts) >= 3 else str(card.get("reward") or "–").split(" ")[0]


def _object(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    name = str(env.get("object_name") or env.get("cup_object") or "shaker")
    if name == "shaker":
        return f"shaker×{env.get('cup_scale', 1.0)} (흰 출력물 aglt_cup_s065, Ø57)"
    return f"{name} (노란 원통 Ø60×170)" if name.startswith("cyl60") else name


def _delay(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    arm, hand = env.get("arm_cmd_delay_steps"), env.get("hand_cmd_delay_steps")
    if arm is None and hand is None:
        return "없음 (키 전 런)"
    span = lambda v: "0" if not v or max(v) == 0 else f"{v[0]}–{v[1]}"  # noqa: E731
    return "없음" if span(arm) == span(hand) == "0" else f"팔 {span(arm)} · 손 {span(hand)} 스텝"


def _fpp(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    on = [n for k, n in (("fpp_enable", "지각"), ("fpp_attach_enable", "부착")) if env.get(k)]
    return " + ".join(on) or "없음"


def _tol(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    s, f = env.get("tol_start"), env.get("tol_floor")
    return "–" if f is None else (f"{f}" if s in (None, f) else f"{s}→{f}")


def _start(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    if env.get("start_bank_path") and float(env.get("start_bank_frac", 1.0)) >= 1.0:
        return f"인계 뱅크 (hold {env.get('hold_steps', 0)})"
    adr = int(env.get("adr_start_increment") or 0)
    return f"홈 · hold {env.get('hold_steps', '?')}" + (f" (ADR {adr} 부터)" if adr else "")


def _goal(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    holders = env.get("target_holders")
    return f"홀더 {' · '.join(str(h) for h in holders)}" if holders else "컵 + (0, 0, 0.14)"


def _arm_law(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    mode = str(env.get("arm_joint_mode", "increment"))
    return f"{mode} · amax {env['arm_abs_amax']}" if env.get("arm_abs_amax") else mode


def _hand_law(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    out = str(env.get("hand_direct_range", "limits")) + (" · 접촉 멈춤" if env.get("hand_direct_contact_freeze") else "")
    margin = float(env.get("hand_close_margin_rad") or 0.0)
    return out + (f" · 닫힘 상한 +{margin}" if margin else "")


def _deploy(field: str, zero: str) -> Callable[[Mapping, Mapping, Mapping], str]:
    def f(_env: Mapping, card: Mapping, _c: Mapping) -> str:
        v = (card.get("deploy") or {}).get(field)
        return "–" if v is None else (zero if float(v) == 0 else str(v))
    return f


def _hand_order(_env: Mapping, _card: Mapping, c: Mapping) -> str:
    src = str(c.get("hand_obs_order_source", ""))
    return "–" if not c else ("실측" if src.startswith("measured") else "가정")


def _slew(env: Mapping, _card: Mapping, _c: Mapping) -> str:
    return str(env["palm_cmd_max_step"]) if env.get("palm_cmd_max_step") else "없음 (slew 전)"


# ── 계열 ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Family:
    key: str
    title: str
    match: tuple[str, ...]           # task 에 들어 있는 글자(모두)
    about: str
    real: str                        # 실기에서 개별 실행하는 길
    cols: tuple[tuple[str, Callable], ...]


FAMILIES = (
    Family("rh_aglt", "RH56F1 한 팔 파지 · 이송 (rh_aglt)", ("open-rh_", "_aglt"),
           "먼 출발 → 컵 쥐기 → 들기 → 목표(리셋 때 컵 + 0.14 m)로 이송. 관측 96 · 행동 13(팔 관절 증분 7 + 손 6) · 60 Hz · LSTM. "
           "계약은 체크포인트 · 컵 치수 말고 모두 같다 — 차이는 아래 학습 조건과 가중치다.",
           "콘솔 → 로봇 `openarm_rh56f1` → 오른/왼 자리에서 고른다(미션 단계 `policy_aglt_<팔>`, 노드 `rh_aglt_node`). "
           "★실물 컵과 FP++ 물체가 정책의 `학습 물체`와 같아야 한다 — 실기 미션은 `REAL_CUP = cyl60`"
           "(`scripts/ops/make_rh56f1_missions.py`)이라 shaker 정책을 돌리려면 컵 · 미션을 바꿔야 한다.",
           (("보상", _reward), ("학습 물체", _object), ("명령 지연", _delay), ("FP++ (학습)", _fpp), ("공차", _tol),
            ("시작", _start))),
    Family("rh_place", "RH56F1 한 팔 컵 홀더 놓기 (rh_place)", ("open-rh_", "_place"),
           "rh_aglt 가 cyl60 을 쥐고 (0.25, ∓0.12, +0.12)에 멈춘 상태를 인계받아 컵 홀더 자리에 내려놓는다. 관측 · 행동 차원과 "
           "디코더는 rh_aglt 와 같고 목표(홀더 자리) · 시작(인계 뱅크, hold 0) · 놓은 뒤 45 스텝 sim 스크립트가 다르다.",
           "미션 단계 `policy_place_<팔>`(노드 `rh_place_node`, 계약 rh_place_contract.json) — aglt 가 컵을 쥐고 인계 자리"
           "(`aglt_goal.py --handoff`)에서 stop 한 뒤. 목표 홀더는 미션 `PLACE_HOLDER`(기본 1). 콘솔 자리는 아직 없다"
           "(팔마다 한 자리 = aglt) — 다른 놓기 정책은 미션 산출물 `place_<팔>` 을 바꿔 쓴다.",
           (("보상", _reward), ("학습 물체", _object), ("명령 지연", _delay), ("FP++ (학습)", _fpp), ("시작", _start),
            ("목표", _goal))),
    Family("pour_fj", "RH56F1 양팔 붓기 (pour_fj)", ("open-rh_", "_pour_fj"),
           "두 팔이 각자 컵을 쥔 채 소스 → 리시버로 붓는다. 관측 165 · 행동 26(팔당 7 + 손 6). 행동 법칙이 런마다 바뀌어 계약이 "
           "그 런의 env.yaml 에서 법칙을 읽는다.",
           "콘솔 → `openarm_rh56f1` → 양팔 자리(미션 단계 `policy_pourfj`). 실기 미션에서는 두 컵 구분 전이라 막혀 있다.",
           (("보상", _reward), ("팔 법칙", _arm_law), ("손 법칙", _hand_law), ("컵", _object), ("시작", _start))),
    Family("dg5f_joint", "DG-5F-M short 한 팔 컵 집기 (joint)", ("open-short_",),
           "DG-5F-M short 손 · 관절 증분 팔. 관측 133 · 행동 26(팔 7 증분 + 손 19 절대). 계약 joint_contract.json.",
           "콘솔 → 로봇 `openarm_dg5f_m_short` → 오른/왼 자리(joint 계약이 있는 것만). 도착 판정 · 에피소드 길이는 카드 deploy 가 정한다.",
           (("보상", _reward), ("목표 이어주기", lambda e, c, k: str(e.get("goal_delta_distance", "–"))),
            ("배포 도착 판정", _deploy("success_tol_m", "끔")), ("에피소드", _deploy("max_episode_s", "끝없음")),
            ("손 관측 순서", _hand_order))),
    Family("pour_fab", "DG-5F short 양팔 붓기 (pour_fab, fabric)", ("open-short_b_pour_fab",),
           "팔당 손바닥 6D 증분(fabric) + grip3 · 관측 223 · 행동 18 · MLP.",
           "fake 미션만(`config/mission_pour_fake.yaml` · `mission_pour_i24_fake.yaml`). 실기는 ckpt_gate · pour_guard 를 넘긴 뒤.",
           (("보상", _reward), ("팜 slew", _slew))),
)
OTHER = Family("other", "기타", (), "", "", (("보상", _reward),))


def family_of(task: str) -> Family:
    """task 이름에 맞는 계열 중 가장 구체적인 것(맞춘 글자가 가장 긴 것)."""
    hits = [f for f in FAMILIES if all(m in task for m in f.match)]
    return max(hits, key=lambda f: sum(map(len, f.match))) if hits else OTHER


def epoch_of(checkpoint: str) -> str:
    m = re.search(r"_ep_?(\d+)|_e(\d+)", checkpoint)
    return f"ep{m.group(1)}" if m and m.group(1) else (f"e{m.group(2)}" if m else "best")


def play_task(task: str) -> str:
    return task.replace("-lstm", "-play-lstm") if task.endswith("-lstm") else task + "-play"


def columns(e: R.Entry) -> dict[str, str]:
    env, card, c = load_env(e), e.card or {}, _contract(e)
    return {label: fn(env, card, c) for label, fn in family_of(str(card.get("task", ""))).cols}


def mission_defaults(repo: Path) -> dict[str, tuple[str, ...]]:
    """config/mission_*.yaml 이 가리키는 정책 → 미션 이름들(콘솔에서 안 고르면 쓰는 것)."""
    out: dict[str, set[str]] = {}
    for p in sorted((repo / "config").glob("mission_*.yaml")):
        # id = 등록부 상대 경로(10.06 <손>/<과제>/<팔>_<태그>) — 계약 파일 이름 앞까지가 id 다
        for pid in re.findall(r"deploy/policies/((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+)/[A-Za-z0-9_.-]+\.json",
                              p.read_text(encoding="utf-8")):
            out.setdefault(pid, set()).add(p.stem.removeprefix("mission_"))
    return {k: tuple(sorted(v)) for k, v in out.items()}


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip() or "–"


def _family_section(fam: Family, entries: list[R.Entry], missions: Mapping[str, tuple[str, ...]]) -> list[str]:
    out = [f"## {fam.title}", "", fam.about, "", f"**실기 개별 실행** — {fam.real}", ""]
    out += ["| id | 쪽 | status | 체크포인트 | 설명 | sim 평가 | 미션 기본 |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        card = e.card or {}
        ck = e.checkpoint or str(card.get("checkpoint") or "")
        out.append(f"| `{e.id}` | {card.get('side', '')} | {e.status}{'' if e.ok else ' ✗'} | {epoch_of(ck)} "
                   f"| {_cell(card.get('summary', ''))} | {_cell(card.get('eval', ''))} "
                   f"| {', '.join(missions.get(e.id, ())) or '–'} |")
    labels = [label for label, _ in fam.cols]
    out += ["", "학습 조건(각 폴더 `params/env.yaml` 에서 읽음):", "",
            "| id | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for e in entries:
        row = columns(e)
        out.append(f"| `{e.id}` | " + " | ".join(_cell(row[label]) for label in labels) + " |")
    return out + [""]


def _play_section(entries: Iterable[R.Entry]) -> list[str]:
    out = ["## sim 에서 개별 실행(비교 분석)", "",
           "hdgp 에서 play 로 돌린다. 체크포인트 옆 `params/` 를 play 가 복원하므로 학습 때 env(물체 · FP++ · 지연)가 그대로 선다. "
           "학습이 도는 GPU 에는 올리지 않는다(nvidia-smi 확인). 결정론 평가는 그 정책을 학습한 호스트에서 한다.", "",
           "```bash", "cd ~/rl_ws/hdgp"]
    for e in entries:
        ck = e.checkpoint or str((e.card or {}).get("checkpoint") or "")
        task = str((e.card or {}).get("task", ""))
        if not ck or not task or not (e.path / "params" / "env.yaml").is_file():
            out.append(f"# {e.id}: 한 벌(런별 params) — params/<체크포인트>/ 를 런 폴더 모양으로 옮겨서 돌린다")
            continue
        out.append(f"python scripts/reinforcement_learning/rl_games/play.py --task {play_task(task)} --headless "
                   f"--num_envs 64 --seed 42 --checkpoint ~/rl_ws/sim2real/deploy/policies/{e.id}/nn/{ck}")
    return out + ["```", ""]


def _tree_section(entries: list[R.Entry]) -> list[str]:
    """폴더 = 로봇 손 / 정책 과제 / 팔_태그(10.06 사용자) — 한눈에 보는 목록."""
    out = ["## 폴더 (손 / 과제 / 팔_태그)", "", "```"]
    last: tuple[str, ...] = ()
    for e in entries:
        parts = tuple(e.id.split("/"))
        for depth in range(len(parts) - 1):
            if parts[:depth + 1] != last[:depth + 1]:
                out.append("  " * depth + parts[depth] + "/")
        status = "" if e.status == "candidate" else f"  ({e.status})"
        out.append("  " * (len(parts) - 1) + parts[-1] + status)
        last = parts
    return out + ["```", ""]


def render(entries: Iterable[R.Entry], *, repo: Path) -> str:
    entries = list(entries)
    missions = mission_defaults(repo)
    out = ["# policies — 쓸 정책 목록 · 차이 · 개별 실행", "",
           "`python3 deploy/policy_control/tools/policies.py --write-index` 가 만든다. 손으로 고치지 않는다 — 설명은 각 "
           "`policy.yaml` 의 `summary` · `eval` · `note`, 학습 조건은 `params/env.yaml` 에서 온다.", "",
           "- **미션 기본** = 콘솔에서 정책을 고르지 않으면 그 미션이 쓰는 정책(`config/mission_*.yaml`).",
           "- **sim 평가** 는 카드에 적힌 결정론 평가 요약이다. 공차 · 조건이 정책마다 달라 숫자끼리 바로 비교하지 않는다.",
           "- 체크포인트 가중치는 git 에 없다(`nn/` .gitignore) — 다른 호스트에서는 `check_host.py` 가 받는 명령을 알려 준다.", ""]
    out += _tree_section(entries)
    for fam in (*FAMILIES, OTHER):
        members = [e for e in entries if family_of(str((e.card or {}).get("task", ""))) is fam]
        if members:
            out += _family_section(fam, members, missions)
    out += _play_section(entries)
    out += ["## status", ""] + [f"- `{k}` — {v}" for k, v in R.STATUSES.items()]
    bad = [(e, (*e.issues, *R.layout_issues(e))) for e in entries]
    bad = [(e, problems) for e, problems in bad if problems]
    if bad:
        out += ["", "## 점검 문제", ""] + [f"- `{e.id}` — {' / '.join(problems)}" for e, problems in bad]
    notes = [(e.id, str((e.card or {}).get("note") or "").strip()) for e in entries]
    out += ["", "## note (카드 원문)", ""] + [f"- `{i}` — {n}" for i, n in notes if n]
    return "\n".join(out) + "\n"
