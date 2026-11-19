"""실기 경계 사진 꺼내기: 러너 기록(episode_*.jsonl) + 카메라 bag → 경계별 jpg 와 bounds.jsonl. ROS 는 쓰지 않는다.

실행(저장소 루트): python3 -m vlm_supervisor.bench.real_frames <runs_root> --cam <cam_root> --spec <에피소드 yaml> --out <새 폴더>
읽는 것: <runs_root>/<날짜>/episode/*_<이름>/episode_*.jsonl, <cam_root>/<날짜>/bags/*/camera/*.db3 (이름 = yaml 파일 이름).
경계 = 러너가 다음 승인을 기다리는 지점. yaml sequence 의 맨 위 노드 하나와 뒤따르는 auto 노드(lanes 는 안 노드 포함)가 한 묶음이고
묶음의 마지막 종료(exit·failure) 뒤가 경계다. sim 시험대(build_set)와 같은 기준이다.
사진 = 마지막 종료 + AFTER_S(러너의 놓기 판정 대기와 같다)에 가장 가까운 컬러 프레임. 다음 노드가 먼저 들어오면 진입 직전.
bounds.jsonl 의 사진 경로는 그 파일이 있는 폴더 기준이다(폴더째 다른 PC 로 옮겨도 된다). 라벨·판정 결과는 여기서 읽지 않는다.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import json
import sqlite3
import zoneinfo
from pathlib import Path

import yaml

from . import bagread

TOPIC = '/camera/camera/color/image_raw/compressed'
AFTER_S = 1.0    # 종료 뒤 사진까지(episode_ros 의 PLACE_JUDGE_WAIT_S 와 같은 값)
MARGIN_S = 0.1   # 다음 노드 진입보다 이만큼 앞 프레임을 쓴다
TOL_S = 0.3      # 목표 시각과 프레임 시각의 허용 차(영상 5 Hz)
FIN = ('exit', 'failure')  # 노드 종료 행. 실패한 놓기는 exit 없이 failure 로만 남는다


def groups(spec: dict) -> dict:
    """노드 id → (묶음 번호, 묶음 첫 노드 id). auto 노드는 앞 묶음에 붙고 lanes 안 노드는 그 lanes 묶음이다."""
    out, g, head = {}, -1, None
    for node in spec['sequence']:
        if node.get('type') == 'terminal':
            continue
        if g < 0 or not node.get('auto'):
            g, head = g + 1, node['id']
        lanes = [n for lane in node.get('lanes') or [] for n in (lane if isinstance(lane, list) else lane.get('nodes') or [])]
        out.update({n['id']: (g, head) for n in [node, *lanes]})
    return out


def episode_bounds(rows: list, grp: dict, to_s) -> list:
    """한 에피소드 기록의 경계 목록(head · node · nodes · t_fin · t_photo · gap_s · end). 시각은 epoch 초.

    node = 묶음에서 마지막으로 끝난 노드. gap_s = 종료에서 다음 진입까지(끝이면 None). 0 에 가까우면 로봇이 서지 않고 이어 간 경계다.
    같은 노드가 묶음 안에서 다시 들어오면(재시도) 앞 시도를 따로 닫는다. end = 그 묶음이 yaml 의 마지막 묶음이다.
    """
    out, cur, last_g = [], None, max(g for g, _ in grp.values())

    def close(cur: dict, nxt):
        gap = None if nxt is None else nxt - cur['t_fin']
        t = cur['t_fin'] + (AFTER_S if gap is None else max(0.0, min(AFTER_S, gap - MARGIN_S)))
        out.append({'head': cur['head'], 'node': cur['nodes'][-1]['id'], 'nodes': cur['nodes'],
                    't_fin': cur['t_fin'], 't_photo': t, 'gap_s': gap, 'end': cur['g'] == last_g})

    for r in rows:
        node, ev = r.get('state_id'), r.get('event')
        if node not in grp:
            continue
        g, head = grp[node]
        if ev == 'enter':
            ts = to_s(r['timestamp'])
            if cur and cur['nodes'] and (g != cur['g'] or node in {n['id'] for n in cur['nodes']}):
                close(cur, ts)
                cur = None
            if cur is None or g != cur['g']:
                cur = {'g': g, 'head': head, 'nodes': [], 't_fin': None, 'ent': {}}
            cur['ent'][node] = ts
        elif cur and g == cur['g'] and (ev in FIN or r.get('signals')):
            last = cur['nodes'][-1] if cur['nodes'] and cur['nodes'][-1]['id'] == node else None
            if ev in FIN and r.get('executor_type') != 'lanes' and last is None:
                last = {'id': node, 'type': r.get('executor_type'), 'policy': r.get('policy_name'), 'event': ev,
                        'result': r.get('result'), 'reason': r.get('termination_reason'),
                        't_enter': cur['ent'].get(node), 't_exit': to_s(r['timestamp']), 'signals': None}
                cur['nodes'].append(last)
            if ev in FIN:
                cur['t_fin'] = to_s(r['timestamp'])
            if last is not None and last['signals'] is None and r.get('signals'):
                last['signals'] = r['signals']  # 실패한 놓기의 신호는 failure 뒤 행에 실릴 수 있다
    if cur and cur['nodes']:
        close(cur, None)
    return out


def cam_index(day_dir: Path) -> list:
    """그 날짜의 카메라 bag 마다 (첫 시각 s, 끝 시각 s, [(ns, db)]). 못 읽는 bag(기록·복사가 끊긴 것)은 알리고 뺀다."""
    out = []
    for cam in sorted(Path(day_dir).glob('bags/*/camera')):
        try:
            st = bagread.stamps(cam, TOPIC)
        except sqlite3.DatabaseError as exc:
            print(f'  못 읽은 bag {cam}: {exc}')
            continue
        if st:
            out.append((st[0][0] * 1e-9, st[-1][0] * 1e-9, st))
    return out


def frame_at(cams: list, t: float):
    """t(epoch 초)에 가장 가까운 프레임 (ns, db). 그 시각을 덮는 bag 이 없거나 TOL_S 보다 멀면 (None, 이유)."""
    for t0, t1, st in cams:
        if t0 - TOL_S <= t <= t1 + TOL_S:
            i = bisect.bisect_left(st, (int(t * 1e9),))
            ns, db = min(st[max(0, i - 1):i + 1], key=lambda s: abs(s[0] * 1e-9 - t))
            return ((ns, db), None) if abs(ns * 1e-9 - t) <= TOL_S else (None, '프레임이 목표 시각에서 멀다')
    return None, '그 시각의 카메라 bag 없음'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('runs', help='러너 기록 루트(<날짜>/episode/...)')
    ap.add_argument('--cam', required=True, help='카메라 bag 루트(<날짜>/bags/*/camera)')
    ap.add_argument('--spec', required=True, help='에피소드 yaml(sequence 의 auto·lanes 로 경계를 묶는다)')
    ap.add_argument('--out', required=True, help='사진과 bounds.jsonl 을 쓸 새 폴더')
    ap.add_argument('--tz', default='Asia/Seoul', help='러너 기록 시각의 시간대(기록에 시간대가 없다)')
    args = ap.parse_args(argv)
    out, name = Path(args.out), Path(args.spec).stem
    if (out / 'bounds.jsonl').exists():
        ap.error(f"{out / 'bounds.jsonl'} 이 이미 있다. 새 폴더를 준다")
    grp = groups(yaml.safe_load(Path(args.spec).read_text(encoding='utf-8')))
    tz = zoneinfo.ZoneInfo(args.tz)

    def to_s(stamp: str) -> float:
        return datetime.datetime.fromisoformat(stamp).replace(tzinfo=tz).timestamp()

    cams, lines, skipped = {}, [], collections.Counter()
    for path in sorted(Path(args.runs).glob(f'*/episode/*_{name}/episode_*.jsonl')):
        day, seed = path.parts[-4], f'{path.parts[-4]}_{path.stem[-6:]}'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if day not in cams:
            cams[day] = cam_index(Path(args.cam) / day)
        before = None
        for k, b in enumerate(episode_bounds(rows, grp, to_s)):
            got, why = frame_at(cams[day], b['t_photo'])
            if got is None:
                skipped[why] += 1
                before = None
                continue
            ns, db = got
            fmt, data = bagread.cimage(bagread.at(db, TOPIC, ns))
            jpg = f"{seed}/{k:02d}_{b['node']}.jpg"
            (out / seed).mkdir(parents=True, exist_ok=True)
            (out / jpg).write_bytes(data)
            lines.append({'id': jpg[:-4], 'seed': seed, 'k': k, **b, 't_frame': ns * 1e-9, 'format': fmt,
                          'auto': bool(rows[0].get('auto')), 'jpg': jpg, 'before': before, 'episode_jsonl': str(path)})
            before = jpg
    out.mkdir(parents=True, exist_ok=True)
    (out / 'bounds.jsonl').write_text(''.join(json.dumps(b, ensure_ascii=False) + '\n' for b in lines), encoding='utf-8')
    print(f"경계 {len(lines)}개, 에피소드 {len({b['seed'] for b in lines})}개 → {out}")
    for why, n in skipped.most_common():
        print(f'  뺌 {n}: {why}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
