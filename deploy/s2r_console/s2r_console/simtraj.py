"""★10.19 사용자 "정책이 live 로 하는 것 하나랑 sim 에서 나온 궤적 play 하는 것, 둘 다 선택할 수 있게": 잡기 카드의 sim 궤적.

잡기 카드(경로 A aglt 정책)에 버튼이 둘 더 붙는다 — 정책 live 실행(rg3_<카드>)은 그대로다.
  simplan_<카드>  카드 물체의 FP++ live 컵 자리로 server sim 을 돌려 계획을 만든다(tools/sim_traj.py plan) — 로봇은 안 움직인다.
  simrun_<카드>   그 카드의 마지막 계획을 pd 로 재생한다(tools/sim_traj.py run --execute) — 정책 노드 없이 실기 동작.
계획은 logs/sim_traj/card_<카드>/<시각_팔_물체>/plan.npz. 카드의 정책 · 팔 · 물체와 meta 가 같고 문제가 없는 가장 새 계획만 쓴다.
순수 함수만(콘솔 상태 · ROS 없음).
"""
from __future__ import annotations

import json
import math
import shlex
from pathlib import Path

import numpy as np

TOOL = 'deploy/policy_control/tools/sim_traj.py'
RECORD = 'deploy/policy_control/tools/rh56f1_record.sh'
ROOT = 'logs/sim_traj'
CUP_TOL_M = 0.01            # = sim_traj.CUP_TOL_M(재생 검사와 같은 값 — 시험이 맞춘다)
PLAN_TIMEOUT_S = 1500       # = sim_traj plan --timeout 기본

_meta_cache: dict[Path, tuple[float, dict | None]] = {}


def supported(card: dict) -> bool:
    """sim 궤적 버튼을 붙이는 카드 — 경로 A 잡기 카드(aglt 정책). 접근 카드(경로 C)는 sim 에서 들기까지 안 간다."""
    pid = str(card.get('policy_id', ''))
    return card.get('task') == 'grasping' and bool(card.get('execution_supported')) and '/aglt/' in pid


def card_root(repo: Path, card: dict) -> Path:
    return Path(repo) / ROOT / f"card_{card['id']}"


def read_meta(path: Path) -> dict | None:
    """plan.npz 의 meta(JSON) — mtime 으로 캐시(화면이 자주 묻는다). 못 읽으면 None."""
    path = Path(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _meta_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
    except (OSError, ValueError, KeyError):
        meta = None
    _meta_cache[path] = (mtime, meta)
    return meta


def latest_plan(repo: Path, card: dict, obj: dict | None) -> dict | None:
    """이 카드로 쓸 수 있는 가장 새 계획 {'path', 'meta'} — 정책 · 팔 · 물체가 카드와 같고 계획 문제가 없는 것."""
    if obj is None:
        return None
    root = card_root(repo, card)
    plans = sorted(root.glob('*/plan.npz'), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []
    for path in plans:
        meta = read_meta(path)
        if (meta and meta.get('policy') == card['policy_id'] and meta.get('side') == card['side']
                and meta.get('object') == obj.get('policy_object') and not meta.get('problems')):
            return {'path': path, 'meta': meta}
    return None


def summary(plan: dict | None) -> dict | None:
    """화면 · 경고용 요약 — 승인 근거에 들어가므로 계획 파일에서만 뽑는다(지금 컵 같은 실시간 값은 넣지 않는다)."""
    if plan is None:
        return None
    m = plan['meta']
    return {'path': str(plan['path']), 'made': m.get('made'), 'cup_xy': [round(float(v), 3) for v in m.get('cup_xy', [])],
            'n_ok': m.get('n_ok'), 'n': m.get('n'), 'env': m.get('env'), 'object': m.get('object'),
            'video': next((str(p) for p in sorted(Path(plan['path']).parent.glob('*.mp4'))), None)}


def cup_gap(plan: dict, position) -> float | None:
    """FP++ 컵(로봇 base x, y) ↔ 계획 컵 거리(m)."""
    if plan is None or position is None:
        return None
    try:
        x, y = float(position[0]), float(position[1])
        px, py = (float(v) for v in plan['meta']['cup_xy'])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return math.hypot(x - px, y - py)


def cup_topic(obj: dict) -> str:
    return f"/objects/{obj['id']}/live_pose"


def plan_argv(python: str, repo: Path, card: dict, obj: dict) -> tuple[str, ...]:
    return (python, str(Path(repo) / TOOL), 'plan', '--policy', card['policy_id'], '--object', obj['policy_object'],
            '--cup-topic', cup_topic(obj), '--out-root', str(card_root(repo, card)))


def run_argv(python: str, repo: Path, plan: dict, obj: dict) -> tuple[str, ...]:
    return (python, str(Path(repo) / TOOL), 'run', str(plan['path']), '--cup-topic', cup_topic(obj), '--execute')


def record_name(side: str) -> str:
    return f'simtraj_{side}'


def record_start_argv(repo: Path, side: str, obj: dict) -> tuple[str, ...]:
    """rg3 에피소드와 같은 기록기(팔 · 손 bag 따로) — 재생 목표 · pd 상태 · 컵을 같이 남긴다."""
    extra = ' '.join(('/policy_control/joint_target', f'/policy_control/status/pd_{side}', f'/policy_control/{side}/episode',
                      f"/objects/{obj['id']}/pose", cup_topic(obj), f'/policy_control/pd_{side}/applied'))
    return ('bash', '-lc', f"HAND_SIDES={shlex.quote(side)} EXTRA={shlex.quote(extra)} "
                           f"bash {shlex.quote(str(Path(repo) / RECORD))} start {record_name(side)}")


def record_stop_argv(repo: Path, side: str) -> tuple[str, ...]:
    return ('bash', str(Path(repo) / RECORD), 'stop', record_name(side))


def refusal(log_tail: str) -> str:
    """sim_traj 출력에서 거부 · 끝 사유 한 줄(없으면 마지막 줄)."""
    lines = [ln.strip() for ln in str(log_tail).splitlines() if ln.strip()]
    for ln in reversed(lines):
        if '거부' in ln or '끝 —' in ln or '문제' in ln:
            return ln[-300:]
    return lines[-1][-300:] if lines else ''
