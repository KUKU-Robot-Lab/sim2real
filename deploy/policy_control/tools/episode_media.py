#!/usr/bin/env python3
"""에피소드 한 회의 보고서용 영상(mp4) · 장면 사진을 만든다 — 카메라 PC 의 vision_recorder 프레임 + 에피소드 기록(노드 시각).

10.19 사용자 "작성할 보고서 사진/영상도 필요함". 1-B(grasp_b_*) 에피소드 단계는 카메라 영상을 15 Hz 로 남긴다
(make_rh56f1_missions REPORT_VIDEO_HZ · scripts/nodes/vision_recorder.py — 카메라 PC 의 sim2real/logs/vision_runs/<시각>_episode_<이름>).

  1) 에피소드 기록(logs/episode/<id>_<이름>/episode_<id>.jsonl)에서 처음 enter ~ 마지막 사건 시각과 노드마다 enter · exit 시각
  2) 카메라 PC 의 그 에피소드 영상 폴더에서 그 구간(앞뒤 2 s) 프레임만 가져온다(scp — 프레임 이름 = 시각 ns)
  3) 영상: 프레임마다 위에 [에피소드 · 지금 노드 · 경과 s] 를 얹어 실측 프레임 간격의 fps 로 mp4
  4) 사진: 시작(홈 뒤) · 다가간 뒤(reach exit) · 쥔 뒤(grasp exit) · 들어 올린 뒤(lift exit = 3 s 유지 끝) — 가까운 프레임,
     + 네 장 모음(2×2). 결과 = logs/report_media/<id>_<이름>/{video.mp4, 01_start.jpg …, sheet.jpg, summary.json}
두 PC 시계 차는 10.19 에 ±50 ms(ssh date) — 장면 사진은 그만큼 어긋날 수 있다.

    python3 deploy/policy_control/tools/episode_media.py --latest grasp_b_right
    python3 deploy/policy_control/tools/episode_media.py --episode-dir logs/episode/20261019_230000_grasp_b_right
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

SIM2REAL = Path(__file__).resolve().parents[3]
EPISODE_LOGS = SIM2REAL / "logs" / "episode"
OUT_ROOT = SIM2REAL / "logs" / "report_media"
REMOTE_ROOT = "rl_ws/sim2real/logs/vision_runs"     # 카메라 PC 홈 기준
PAD_S = 2.0
#: 장면 사진 — (파일 이름, 노드, 사건, 설명)
KEYS = (("01_start", "scene", "exit", "시작 — 홈 · 컵 배치 기록"),
        ("02_reached", "reach", "exit", "다가간 뒤 — 편 손 안에 컵"),
        ("03_grasped", "grasp", "exit", "쥔 뒤 — 3 s 유지 끝"),
        ("04_lifted", "lift", "exit", "들어 올린 뒤 — 10 cm · 3 s 유지 끝"))


class MediaError(RuntimeError):
    pass


# ============================================================================ 순수
def stamp(text: str) -> float:
    """'2026-10-19T22:13:42.227'(로컬 시각) → epoch s."""
    return datetime.fromisoformat(text).timestamp()


def timeline(events: list[dict]) -> dict:
    """에피소드 사건 → {start, end, result, nodes: [(id, enter, exit|None, result)]}."""
    if not events:
        raise MediaError("에피소드 기록이 비었다")
    nodes, open_ = [], {}
    for e in events:
        t, sid = stamp(e["timestamp"]), e.get("state_id")
        if e.get("event") == "enter":
            open_[sid] = len(nodes)
            nodes.append([sid, t, None, ""])
        elif e.get("event") in ("exit", "failure") and sid in open_:
            nodes[open_[sid]][2], nodes[open_[sid]][3] = t, str(e.get("result") or e.get("event"))
    result = next((e.get("result") for e in reversed(events) if e.get("event") in ("terminal", "safe_stop", "done")), None)
    return {"start": stamp(events[0]["timestamp"]), "end": stamp(events[-1]["timestamp"]), "result": result,
            "episode": events[0].get("episode"), "episode_id": events[0].get("episode_id"),
            "nodes": [tuple(n) for n in nodes]}


def node_at(tl: dict, t: float) -> str:
    cur = ""
    for sid, t0, t1, _r in tl["nodes"]:
        if t0 <= t and (t1 is None or t <= t1 + 1e-6):
            cur = sid
    return cur


def pick_run(lines, t0: float, t1: float) -> str | None:
    """"DIR <폴더> <첫 프레임> <끝 프레임>" 줄들(최근 것부터) → 에피소드 구간과 겹치는 첫 폴더."""
    for line in lines:
        w = line.split()
        if len(w) == 4 and w[0] == "DIR" and w[2].endswith(".jpg") and w[3].endswith(".jpg"):
            a, b = int(w[2][:-4]) * 1e-9, int(w[3][:-4]) * 1e-9
            if a <= t1 and b >= t0:
                return w[1]
    return None


def frame_times(names) -> np.ndarray:
    return np.array([int(Path(n).stem) * 1e-9 for n in names])


def keyframes(tl: dict, times: np.ndarray) -> list[dict]:
    """장면마다 그 노드 사건 시각에 가장 가까운 프레임 번호(없으면 빠진다)."""
    out = []
    for name, sid, ev, text in KEYS:
        hit = [n for n in tl["nodes"] if n[0] == sid and (n[2] if ev == "exit" else n[1]) is not None]
        if not hit or len(times) == 0:
            continue
        t = hit[-1][2] if ev == "exit" else hit[-1][1]
        k = int(np.argmin(np.abs(times - t)))
        if abs(times[k] - t) <= 1.0:
            out.append({"name": name, "node": sid, "t": t, "frame": k, "gap_s": round(float(times[k] - t), 3), "text": text})
    return out


# ============================================================================ 가져오기 · 만들기
def find_episode_dir(name: str) -> Path:
    hits = sorted(EPISODE_LOGS.glob(f"*_{name}"))
    if not hits:
        raise MediaError(f"{EPISODE_LOGS}/*_{name} 가 없다")
    return hits[-1]


def load_events(d: Path) -> list[dict]:
    files = sorted(d.glob("episode_*.jsonl"))
    if not files:
        raise MediaError(f"{d}: episode_*.jsonl 이 없다")
    return [json.loads(line) for line in files[-1].read_text(encoding="utf-8").splitlines() if line.strip()]


def remote_frames(host: str, episode: str, t0: float, t1: float) -> tuple[str, list[str]]:
    """카메라 PC 에서 그 에피소드 영상 폴더(이름 끝 _episode_<이름>, 구간이 겹치는 마지막) · 구간 프레임 이름."""
    script = (f"cd {REMOTE_ROOT} 2>/dev/null || exit 3; for d in $(ls -d *_episode_{shlex.quote(episode)} 2>/dev/null | sort -r); "
              'do echo "DIR $d $(ls $d/frames 2>/dev/null | head -1) $(ls $d/frames 2>/dev/null | tail -1)"; done')
    got = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, script], capture_output=True, text=True,
                         timeout=60)
    if got.returncode == 3:
        raise MediaError(f"{host}:~/{REMOTE_ROOT} 가 없다 — 영상 기록이 돈 적이 없다")
    pick = pick_run(got.stdout.splitlines(), t0, t1)
    if pick is None:
        raise MediaError(f"{host}: 에피소드 {episode} 구간({datetime.fromtimestamp(t0):%H:%M:%S}~{datetime.fromtimestamp(t1):%H:%M:%S})"
                         f"과 겹치는 영상 폴더가 없다(~/{REMOTE_ROOT}/*_episode_{episode})")
    ls = subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"ls {REMOTE_ROOT}/{pick}/frames"], capture_output=True, text=True,
                        timeout=60).stdout.split()
    names = [n for n in ls if n.endswith(".jpg") and t0 - PAD_S <= int(n[:-4]) * 1e-9 <= t1 + PAD_S]
    return f"{REMOTE_ROOT}/{pick}", sorted(names)


def local_frames(tl: dict, dst: Path) -> Path:
    root = SIM2REAL / "logs" / "vision_runs"
    lines = []
    for d in sorted(root.glob(f"*_episode_{tl['episode']}"), reverse=True):
        fs = sorted((d / "frames").glob("*.jpg"))
        if fs:
            lines.append(f"DIR {d.name} {fs[0].name} {fs[-1].name}")
    pick = pick_run(lines, tl["start"], tl["end"])
    if pick is None:
        raise MediaError(f"{root}/*_episode_{tl['episode']} 에 이 회 구간 영상이 없다")
    dst.mkdir(parents=True, exist_ok=True)
    for f in (root / pick / "frames").glob("*.jpg"):
        if tl["start"] - PAD_S <= int(f.stem) * 1e-9 <= tl["end"] + PAD_S and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f)
    return dst


def fetch(host: str, rdir: str, names: list[str], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in dst.glob("*.jpg")}
    need = [n for n in names if n not in have]
    for i in range(0, len(need), 400):            # 명령줄 길이
        chunk = need[i:i + 400]
        tar = subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"cd {rdir}/frames && tar cf - {' '.join(chunk)}"],
                             capture_output=True, timeout=600)
        if tar.returncode != 0:
            raise MediaError(f"프레임 가져오기 실패: {tar.stderr.decode(errors='replace')[-300:]}")
        subprocess.run(["tar", "xf", "-", "-C", str(dst)], input=tar.stdout, check=True, timeout=600)


def caption(img, lines: list[str]):
    import cv2
    out = img.copy()
    h = 28 * len(lines) + 12
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def build(frames_dir: Path, tl: dict, out: Path) -> dict:
    import cv2
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        raise MediaError(f"{frames_dir}: 프레임이 없다")
    times = frame_times([f.name for f in files])
    dt = np.diff(times)
    fps = float(np.clip(1.0 / np.median(dt), 1.0, 60.0)) if len(dt) else 15.0
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    out.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    title = f"{tl['episode']} {tl['episode_id']}"
    for f, t in zip(files, times):
        im = cv2.imread(str(f))
        if im is None:
            continue
        vw.write(caption(im, [f"{title} | {node_at(tl, t) or '-'} | {t - tl['start']:6.1f} s"]))
    vw.release()
    keys = keyframes(tl, times)
    tiles = []
    for k in keys:
        im = cv2.imread(str(files[k["frame"]]))
        cv2.imwrite(str(out / f"{k['name']}.jpg"), im)                                   # 보고서 원본(글자 없음)
        tiles.append(caption(im, [f"{k['name'][3:]} ({k['t'] - tl['start']:.1f} s)"]))
    if tiles:
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
        cv2.imwrite(str(out / "sheet.jpg"), cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2)))
    summary = {"episode": tl["episode"], "episode_id": tl["episode_id"], "result": tl["result"], "frames": len(files),
               "fps": round(fps, 2), "size": [w, h], "duration_s": round(float(times[-1] - times[0]), 2),
               "nodes": [{"id": n[0], "enter_s": round(n[1] - tl["start"], 2),
                          "exit_s": None if n[2] is None else round(n[2] - tl["start"], 2), "result": n[3]} for n in tl["nodes"]],
               "keyframes": [{k: v for k, v in x.items() if k != "frame"} | {"file": f"{x['name']}.jpg"} for x in keys],
               "missing_keyframes": [n for n, *_ in KEYS if n not in {x["name"] for x in keys}]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest", help="에피소드 이름(가장 최근 회) — 예 grasp_b_right")
    g.add_argument("--episode-dir", type=Path)
    g.add_argument("--pending", help="에피소드 이름 — 아직 만들지 않은 회(report_media/<회>/summary.json 없음)를 모두")
    ap.add_argument("--host", default="vision-3090", help="카메라 PC(ssh 이름) · local = 이 PC 의 logs/vision_runs")
    ap.add_argument("--frames", type=Path, default=None, help="프레임이 이미 이 PC 에 있으면 그 폴더(가져오지 않는다)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.pending:
        todo = [d for d in sorted(EPISODE_LOGS.glob(f"*_{a.pending}")) if not (OUT_ROOT / d.name / "summary.json").is_file()]
        if not todo:
            print(f"[media] {a.pending}: 새 회 없음")
            return 0
        rcs = [main(["--episode-dir", str(d), "--host", a.host]) for d in todo]
        return 0 if all(rc == 0 for rc in rcs) else 2
    try:
        d = a.episode_dir if a.episode_dir else find_episode_dir(a.latest)
        tl = timeline(load_events(d))
        out = a.out or OUT_ROOT / d.name
        frames = a.frames
        if frames is None and a.host == "local":          # 카메라가 이 PC — logs/vision_runs 에서 구간 프레임을 이어 붙인다(복사 없음)
            frames = local_frames(tl, out / "frames")
        elif frames is None:
            rdir, names = remote_frames(a.host, tl["episode"], tl["start"], tl["end"])
            print(f"[media] {a.host}:~/{rdir} · 구간 프레임 {len(names)} 장 가져오기", flush=True)
            frames = out / "frames"
            fetch(a.host, rdir, names, frames)
        s = build(frames, tl, out)
    except (MediaError, subprocess.SubprocessError, OSError) as exc:
        print(f"✗ {exc}")
        return 2
    print(f"[media] {out}/video.mp4 · {s['frames']} 장 {s['fps']} fps {s['duration_s']} s · 사진 {[k['file'] for k in s['keyframes']]}"
          + (f" · 못 찾은 장면 {s['missing_keyframes']}" if s["missing_keyframes"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
