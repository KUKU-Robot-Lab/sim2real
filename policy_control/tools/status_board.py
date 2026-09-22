#!/usr/bin/env python3
"""읽기 전용 운영 상태판. 버튼 없음, 명령 없음, 새 상태기계 없음.

실기 세션에서 터미널 네 개를 오가지 않으려고 만든 것이다. 보여주는 세 가지는 전부
**이미 있는 출처의 문자열을 그대로** 옮긴 것이고, 이 파일은 판정을 새로 하지 않는다.

  ⓐ 미션 게이트   `scripts/mission_core.gate()` 의 사유 — 무엇이 막고 있는지
  ⓑ 노드 상태     `/policy_control/status/{obs,policy,fabric,pd}` 의 phase / ok / reasons 원문
  ⓒ 흐름          `policy_control.status_join` 의 지연 p50/p95 · seq 결손 (status_to_csv 와 같은 숫자)

의존성은 표준 라이브러리뿐이다(`scripts/vision/stream_head_view.py` 와 같은 방식). 기본 바인딩은
`127.0.0.1` — 원격에서 볼 때는 ssh 터널을 쓴다. `--no-ros` 로는 ROS 없이 ⓐ 만 띄운다.

    python3 policy_control/tools/status_board.py --mission config/mission_policy_control.yaml
    # 다른 PC 에서:  ssh -L 8090:127.0.0.1:8090 <robot-pc>  →  http://127.0.0.1:8090
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "policy_control"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "ops"))

from policy_control.status_join import NODES, StatusJoiner, summarize  # noqa: E402

REFRESH_S = 2
STALE_S = 2.0          # 이보다 오래 소식이 없으면 그 노드는 '끊김'으로 그린다
KEEP = 4000            # 링버퍼 — 상태판은 기록물이 아니다


class Board:
    """구독으로 들어온 것을 모아 두기만 한다. 판정은 하지 않는다."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.joiner = StatusJoiner()
        self.last: dict[str, tuple[float, dict]] = {}
        self.episode: dict = {}
        self.n_seen = 0

    def put(self, node: str, payload: dict) -> None:
        with self.lock:
            self.last[node] = (time.monotonic(), payload)
            self.joiner.offer(node, payload)
            self.n_seen += 1
            if len(self.joiner) > KEEP:
                self.joiner = StatusJoiner()      # 통째로 버린다 — 부분 삭제는 seq 결손을 거짓말하게 만든다

    def put_episode(self, payload: dict) -> None:
        with self.lock:
            self.episode = payload

    def snapshot(self) -> tuple[dict, dict, list, int]:
        now = time.monotonic()
        with self.lock:
            last = {n: (now - t, dict(d)) for n, (t, d) in self.last.items()}
            return last, dict(self.episode), self.joiner.rows(), self.n_seen


def mission_rows(mission_path: Path):
    """`mission_core.plan()` 을 그대로 쓴다. 사유 문자열은 한 글자도 고치지 않는다.

    승인은 **막힘이 아니라 별도 열**로 뺀다. `--approve` 는 실행할 때 주는 것이라 모든 실기 단계가
    항상 '승인 없음'으로 뜨고, 그게 아홉 줄 반복되면 진짜 막힌 것이 묻힌다. 대신 승인을 다 가진
    셈 치고 **구조적 막힘**(blocked 문구·선행·산출물·체크포인트)만 사유로 보여주고,
    승인이 필요한 단계에는 표시를 남긴다. `mission_run.plan_evidence()` 는 부르지 않는다 —
    그 함수는 이 가정을 숨기지만 여기서는 열 이름으로 드러낸다.
    """
    import yaml

    import mission_core as MC
    from mission_run import gather_evidence

    raw = yaml.safe_load(mission_path.read_text())
    mission = MC.load_mission(raw)
    state = MC.initial_state(mission)
    ev = gather_evidence(mission, repo=REPO,
                         approvals=frozenset(st.id for st in mission.stages))
    return mission, [(p.stage, p.result) for p in MC.plan(mission, state, ev)]


def _esc(v) -> str:
    return html.escape(str(v))


def _node_table(last: dict) -> str:
    rows = []
    for n in NODES:
        age, d = last.get(n, (None, {}))
        if age is None:
            rows.append(f"<tr class=off><td>{n}</td><td colspan=4>수신 없음</td></tr>")
            continue
        cls = "off" if age > STALE_S else ("bad" if not d.get("ok", True) else "ok")
        reasons = " · ".join(_esc(r) for r in d.get("reasons", [])) or "—"
        extra = []
        if n == "pd":
            extra = [f"execute={d.get('execute')}", f"estop={d.get('estop')}"]
            if d.get("thermal"):
                extra.append(f"thermal={_esc(d['thermal'])}")
        rows.append(
            f"<tr class={cls}><td>{n}</td><td>{_esc(d.get('phase', '—'))}</td>"
            f"<td>{_esc(d.get('seq', '—'))}</td>"
            f"<td>{age:.1f}s 전{'' if not extra else ' · ' + _esc(' '.join(extra))}</td>"
            f"<td>{reasons}</td></tr>")
    return ("<table><tr><th>노드</th><th>phase</th><th>seq</th><th>소식</th><th>reasons (원문)</th></tr>"
            + "".join(rows) + "</table>")


def _mission_table(mission_path: Path | None) -> str:
    if mission_path is None:
        return "<p class=dim>미션 파일이 지정되지 않았다 (--mission).</p>"
    try:
        mission, rows = mission_rows(mission_path)
    except Exception as exc:                                    # noqa: BLE001 — 화면에 그대로 띄운다
        return f"<p class=bad>미션을 읽지 못했다: {_esc(exc)}</p>"
    blocked = sum(1 for _, r in rows if not r.ok)
    out = [f"<p class=dim>{_esc(mission.name)} · {len(rows)}단계 · 구조적 막힘 {blocked} · "
           f"<code>{_esc(mission_path)}</code></p>",
           "<table><tr><th></th><th>단계</th><th>실기 승인</th>"
           "<th>막는 사유 (mission_core 원문, 승인 제외)</th></tr>"]
    for stage, res in rows:
        mark, cls = ("통과", "ok") if res.ok else ("막힘", "bad")
        reasons = " · ".join(_esc(r) for r in res.reasons) or "—"
        real = f"<code>--approve {_esc(stage.id)}</code>" if stage.touches_real else "—"
        out.append(f"<tr class={cls}><td>{mark}</td><td>{_esc(stage.id)}<br>"
                   f"<span class=dim>{_esc(stage.title)}</span></td>"
                   f"<td>{real}</td><td>{reasons}</td></tr>")
    out.append("</table><p class=dim>승인은 실행 시점에 단계마다 따로 받는다 — 이 표에서는 묻지 않는다. "
               "여기서 '통과'는 <b>구조적으로</b> 준비됐다는 뜻이지 지금 실기를 돌려도 된다는 뜻이 아니다.</p>")
    return "".join(out)


def _flow_block(rows: list, policy_dt: float | None, n_seen: int) -> str:
    return (f"<p><b>{_esc(summarize(rows, policy_dt))}</b></p>"
            f"<p class=dim>받은 status 메시지 {n_seen} 건. 같은 숫자를 파일로 남기려면 "
            f"<code>status_to_csv.py</code>.</p>")


def _page(board: Board, mission_path: Path | None, policy_dt: float | None, ros: bool) -> bytes:
    last, episode, rows, n_seen = board.snapshot()
    ep = (f"episode {episode.get('episode', '—')} · {episode.get('event', '—')}"
          if episode else "에피소드 이벤트 없음")
    body = f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>policy_control 상태판</title><meta http-equiv=refresh content={REFRESH_S}>
<style>
 body{{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:1.4rem;max-width:1100px;
   background:#fbfbfa;color:#1a1a19}}
 h1{{font-size:1.15rem;margin:0 0 .2rem}} h2{{font-size:.95rem;margin:1.6rem 0 .4rem}}
 table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:.3rem .5rem;
   text-align:left;vertical-align:top}} th{{background:#f0efed;font-weight:600}}
 tr.ok td:first-child{{color:#1c7a3e}} tr.bad td:first-child{{color:#b3261e;font-weight:600}}
 tr.off td{{color:#8a8a85}} .dim{{color:#77776f}} .bad{{color:#b3261e}}
 code{{background:#f0efed;padding:0 .25rem}}
 .banner{{padding:.5rem .7rem;border:1px solid #ddd;background:#f0efed;margin-bottom:1rem}}
</style></head><body>
<h1>policy_control 상태판 <span class=dim>— 읽기 전용</span></h1>
<div class=banner>이 화면은 아무것도 발행하지 않는다. 실행은 <code>mission_run.py</code> ·
<code>episode_ctl.py</code> 로 한다. {_esc(ep)}
{'' if ros else ' · <b>ROS 미연결(--no-ros)</b>'}</div>
<h2>ⓐ 미션 게이트</h2>{_mission_table(mission_path)}
<h2>ⓑ 노드</h2>{_node_table(last) if ros else '<p class=dim>ROS 미연결.</p>'}
<h2>ⓒ 흐름</h2>{_flow_block(rows, policy_dt, n_seen) if ros else '<p class=dim>ROS 미연결.</p>'}
<p class=dim>{REFRESH_S}초마다 새로고침 · {time.strftime('%H:%M:%S')}</p>
</body></html>"""
    return body.encode()


def _handler(board: Board, mission_path, policy_dt, ros):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):                                    # noqa: N802
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            page = _page(board, mission_path, policy_dt, ros)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *_a):                          # 접속 로그는 소음이다
            return
    return H


def _spin_ros(board: Board) -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("status_board")

    def on(name):
        def cb(msg):
            try:
                board.put(name, json.loads(msg.data))
            except json.JSONDecodeError:
                pass
        return cb

    def on_episode(msg):
        try:
            board.put_episode(json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    for n in NODES:
        node.create_subscription(String, f"/policy_control/status/{n}", on(n), 50)
    node.create_subscription(String, "/policy_control/episode", on_episode, 10)
    rclpy.spin(node)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--mission", type=Path, default=None, help="config/mission_*.yaml")
    ap.add_argument("--bind", default="127.0.0.1", help="기본 로컬 전용. 바꾸려면 이유가 있어야 한다")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--policy-dt", type=float, default=None, help="지연 예산 표시용 (s)")
    ap.add_argument("--no-ros", action="store_true", help="ROS 없이 미션 게이트만 본다")
    args = ap.parse_args(argv)

    board = Board()
    if not args.no_ros:
        threading.Thread(target=_spin_ros, args=(board,), daemon=True).start()
    server = ThreadingHTTPServer((args.bind, args.port),
                                 _handler(board, args.mission, args.policy_dt, not args.no_ros))
    print(f"[status_board] http://{args.bind}:{args.port} — 읽기 전용, 발행 0", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
