#!/usr/bin/env python3
"""학습 런에서 **계약 생성 입력만** 골라 `sim2real/policies/<id>/` 로 받는다 (정책 등록의 첫 단계).

hdgp 의 미러(`scripts/reward_gen/t2r_round.py`)는 설계상 `summaries/` 와 `test_history.md`
만 가져온다. 계약을 만들려면 그 외에 `params/{env,agent}.yaml` · 체크포인트 1개 ·
sim meta(`<trace>_meta.json`)가 필요하다. hdgp 는 건드리지 않고 여기서 따로 당긴다.

    # 무엇이 있는지 먼저 본다 (다운로드 없음)
    python3 policy_control/tools/fetch_run.py --run t2r_i18 --list
    # ep_2500 과 그 trace 의 meta 를 policies/pour_i18 로
    python3 policy_control/tools/fetch_run.py --run t2r_i18 --checkpoint ep:2500 --out policies/pour_i18
    # 이미 이 PC 에 골라 둔 한 벌에서 (ssh 없음) — 체크포인트는 파일명으로 **지정**한다
    python3 policy_control/tools/fetch_run.py --host local --root ~/rl_ws/our_source --run s2r_init_right \
        --checkpoint fj_rand_i01_best_ep5000.pth --sim-meta none --out policies/grasp_fj_rand_i01
    # 그대로 계약 생성
    python3 policy_control/tools/build_deploy_contract.py --run policies/pour_i18 \
        --sim-meta policies/pour_i18/trace_meta.json
    # 무엇이 등록돼 있나
    python3 policy_control/tools/policies.py

규약
  - `nn/` 에는 **정확히 하나**만 둔다 → `contract_build` 의 "exactly one .pth" 규칙이 저절로 만족된다.
  - `.staging/` 에 받고 원격 sha256 과 재해시가 일치할 때만 제자리로 옮긴다(반쪽 디렉터리 불가).
  - 재실행 시 해시가 같으면 0 바이트 전송. 서버에 못 붙으면 `fetch.json` 으로 로컬 검증만 한다.
  - 서버에서 도는 것은 `ls`/`sha256sum`/`git rev-parse`/rsync read 뿐 — **GPU 를 건드리지 않는다.**
  - `--host local` 은 같은 명령을 ssh 없이 이 PC 에서 돌린다. 출처는 읽기만 하고 **복사**한다(링크 아님) —
    출처 폴더를 누가 정리해도 등록된 정책은 남는다.
  - 받은 자리에 `policy.yaml` 카드가 없으면 `status: candidate` 초안을 한 번 쓴다. 있으면 **건드리지 않는다.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_HOST = "server"
LOCAL = "local"                  # ssh 없이 이 PC 의 디렉터리를 출처로 쓴다
POLICIES = Path(__file__).resolve().parents[2] / "policies"
CARD = "policy.yaml"
NOTES = "README.md"              # 출처에 사람이 쓴 설명이 있으면 같이 가져온다
NOTES_LOCAL = "SOURCE_README.md"
DEFAULT_ROOT = "~/rl_ws/hdgp/log/rl_games/open-short/both/pour-fab"
HDGP_REMOTE = "~/rl_ws/hdgp"
META_LOCAL = "trace_meta.json"
TRACE_LOCAL = "trace.npz"
MANIFEST = "fetch.json"
HISTORY = "test_history.md"      # 런의 **학습 커밋**이 여기 적혀 있다 (record_test_snapshot.py)
COMMIT_RE = re.compile(r"Commit\*\*:\s*`([0-9a-f]{7,40})`")
PARAMS = ("env.yaml", "agent.yaml")
NEAR = 4  # 에폭이 안 맞을 때 보여줄 이웃 후보 수

CKPT_RE = re.compile(r"^last_(?P<task>.+)_ep_(?P<ep>\d+)_rew_(?P<rew>-?[0-9.]+)\.pth$")
EP_IN_NAME = re.compile(r"ep_?(\d+)")

PLAY_HOWTO = ("sim meta 가 없다 — 학습 호스트에서 hdgp play.py 로 만든다: "
              "play.py --task open-short_b_pour_fab --checkpoint <pth> --trace_steps N --trace_out <path> "
              "→ <path 확장자 제거>_meta.json")


class FetchError(RuntimeError):
    """원격 런의 내용이 요청과 맞지 않는다 — 사람이 인자를 고쳐야 한다."""


class RemoteError(RuntimeError):
    """서버에 닿지 못했다."""


@dataclass(frozen=True)
class Ckpt:
    name: str
    epoch: int | None       # None = 에폭 접미사 없는 rl_games best
    rew: float | None


@dataclass(frozen=True)
class FetchFile:
    remote_rel: str
    local_rel: str


# ------------------------------------------------------------------ 순수 선택 로직

def parse_checkpoints(names: Iterable[str]) -> tuple[Ckpt, ...]:
    out = []
    for n in names:
        if not n.endswith(".pth"):
            continue
        m = CKPT_RE.match(n)
        out.append(Ckpt(n, int(m["ep"]), float(m["rew"])) if m else Ckpt(n, None, None))
    return tuple(out)


def pick_checkpoint(names: Sequence[str], query: str) -> str:
    """체크포인트를 **고르지 않고 요구한다** — 자동 선택은 조용한 오배포의 지름길이다."""
    cks = parse_checkpoints(names)
    if not cks:
        raise FetchError("nn/ 에 .pth 가 없다")
    if not query:
        raise FetchError("--checkpoint 가 필요하다: best | last | ep:NNNN | <파일명>  (--list 로 후보를 본다)")
    if query == "best":
        best = [c for c in cks if c.epoch is None]
        if not best:
            raise FetchError("에폭 접미사 없는 best 체크포인트가 없다 — ep:NNNN 또는 last 를 써라")
        if len(best) > 1:
            raise FetchError("best 후보가 여러 개다: " + ", ".join(sorted(c.name for c in best)))
        return best[0].name
    if query == "last":
        tagged = [(c.epoch, c.name) for c in cks if c.epoch is not None]
        if not tagged:
            raise FetchError("에폭 태그가 붙은 체크포인트가 없다 — best 를 써라")
        return max(tagged)[1]
    if query.startswith("ep:"):
        want = int(query[3:])
        hit = [c for c in cks if c.epoch == want]
        if len(hit) == 1:
            return hit[0].name
        if hit:
            raise FetchError(f"ep {want} 후보가 여러 개다: " + ", ".join(sorted(c.name for c in hit)))
        tagged = [(ep, c.rew) for c in cks if (ep := c.epoch) is not None]
        near = sorted(tagged, key=lambda t: abs(t[0] - want))[:NEAR]
        raise FetchError(f"ep {want} 이 없다. 가까운 후보: "
                         + ", ".join(f"ep {ep} (rew {rew})" for ep, rew in near))
    if query in {c.name for c in cks}:
        return query
    raise FetchError(f"{query!r} 이 nn/ 에 없다 — --list 로 후보를 봐라")


def _meta_names(top: Iterable[str]) -> list[str]:
    return sorted(n for n in top if n.endswith("_meta.json"))


def pick_sim_meta(top: Sequence[str], checkpoint: str) -> str:
    metas = _meta_names(top)
    if not metas:
        raise FetchError(PLAY_HOWTO)
    m = EP_IN_NAME.search(checkpoint)
    if m:
        same = [n for n in metas if f"ep{m[1]}" in n.replace("ep_", "ep")]
        if len(same) == 1:
            return same[0]
    if len(metas) == 1:
        return metas[0]
    raise FetchError("sim meta 후보가 여러 개다 — --sim-meta 로 지정해라: " + ", ".join(metas))


def pick_trace(top: Sequence[str], sim_meta: str) -> str:
    want = sim_meta[: -len("_meta.json")] + ".npz"
    if want not in top:
        raise FetchError(f"{sim_meta} 옆에 {want} 가 없다")
    return want


def plan_files(listing: Mapping[str, Sequence[str]], *, checkpoint: str,
               sim_meta: str, trace: str) -> tuple[FetchFile, ...]:
    top, nn, params = listing.get("", ()), listing.get("nn", ()), listing.get("params", ())
    missing = [p for p in PARAMS if p not in params]
    if missing:
        raise FetchError("params/ 에 " + ", ".join(missing) + " 가 없다 — 계약의 모든 숫자가 여기서 나온다")
    ck = pick_checkpoint(nn, checkpoint)
    out = [FetchFile(f"params/{p}", f"params/{p}") for p in PARAMS]
    out.append(FetchFile(f"nn/{ck}", f"nn/{ck}"))
    if HISTORY in top:          # 1 KB 남짓이고 런의 학습 커밋·자산·가설이 여기 있다
        out.append(FetchFile(HISTORY, HISTORY))
    if NOTES in top:            # 골라 둔 한 벌에 사람이 쓴 지표·인터페이스 메모
        out.append(FetchFile(NOTES, NOTES_LOCAL))
    # sim meta 는 pour 계열만 요구한다. 단일팔 계약은 런 덤프만으로 만들어진다.
    meta = ""
    if sim_meta != "none":
        meta = pick_sim_meta(top, ck) if sim_meta == "auto" else sim_meta
        if sim_meta != "auto" and meta not in top:
            raise FetchError(f"{meta} 가 런 디렉터리에 없다")
        out.append(FetchFile(meta, META_LOCAL))
    if trace != "none":
        if not meta:
            raise FetchError("--trace 는 sim meta 가 있어야 짝을 찾는다 — --sim-meta 를 none 이 아니게 해라")
        npz = pick_trace(top, meta) if trace == "auto" else trace
        out.append(FetchFile(npz, TRACE_LOCAL))
    return tuple(out)


def manifest_is_current(manifest: Mapping, local: Mapping[str, str],
                        remote: Mapping[str, str] | None,
                        want: Sequence[str] | None = None) -> bool:
    """매니페스트가 가리키는 모든 파일이 지금도 같은가. remote 가 None 이면 로컬만 본다.

    ``want`` 는 이번에 요청한 local_rel 목록이다. 매니페스트가 그것을 다 덮지 못하면
    (예: trace 를 새로 요청했는데 기록은 4개뿐) '최신'이 아니다 — 안 그러면 조용히 건너뛴다.
    """
    files = manifest.get("files") or []
    if not files:
        return False
    if want is not None and not set(want) <= {f["local_rel"] for f in files}:
        return False
    for f in files:
        digest = f["sha256"]
        if local.get(f["local_rel"]) != digest:
            return False
        if remote is not None and remote.get(f["remote_rel"]) != digest:
            return False
    return True


# ------------------------------------------------------------------ 원격 (읽기 전용)

def run_cmd(argv: Sequence[str], **kw) -> str:
    p = subprocess.run(list(argv), capture_output=True, text=True, timeout=kw.get("timeout", 600))
    if p.returncode != 0:
        raise RemoteError((p.stderr or p.stdout).strip() or f"rc={p.returncode}: {' '.join(argv)}")
    return p.stdout.strip()


def _ssh(runner, host: str, script: str) -> str:
    """원격이면 ssh, `local` 이면 같은 스크립트를 이 PC 의 bash 로 — 선택 로직은 하나만 둔다."""
    return runner(["bash", "-c", script] if host == LOCAL else ["ssh", host, script])


def _source(host: str, run_dir: str, rel: str) -> str:
    # rsync 는 셸을 거치지 않으므로 로컬 경로의 ~ 는 여기서 편다.
    return f"{os.path.expanduser(run_dir)}/{rel}" if host == LOCAL else f"{host}:{run_dir}/{rel}"


def remote_listing(runner, host: str, run_dir: str) -> dict[str, list[str]]:
    # summaries/ 와 videos/ 는 계약과 무관하고 파일이 많다 — 가지치기해서 목록을 작게 유지한다.
    raw = _ssh(runner, host, f"cd {run_dir} && find . -maxdepth 2 -mindepth 1 "
                             r"\( -name summaries -o -name videos \) -prune -o -printf '%P\n'")
    out: dict[str, list[str]] = {"": [], "nn": [], "params": []}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("nn/"):
            out["nn"].append(line[3:])
        elif line.startswith("params/"):
            out["params"].append(line[7:])
        else:
            out[""].append(line)
    return out


def remote_hashes(runner, host: str, run_dir: str, rels: Sequence[str]) -> dict[str, str]:
    if not rels:
        return {}
    quoted = " ".join(f"'{r}'" for r in rels)
    raw = _ssh(runner, host, f"cd {run_dir} && sha256sum {quoted}")
    out = {}
    for line in raw.splitlines():
        h, _, path = line.partition("  ")
        if path:
            out[path.strip()] = h.strip()
    return out


def remote_commit(runner, host: str) -> str:
    """**fetch 시점의** hdgp HEAD. 런이 학습된 커밋이 아니다 — 그건 test_history.md 에 있다."""
    try:
        return _ssh(runner, host, f"git -C {HDGP_REMOTE} rev-parse HEAD")
    except RemoteError:
        return ""


def run_commit(out: Path) -> str:
    """런이 학습된 hdgp 커밋 — `test_history.md` 의 '코드 스냅샷' 에서 읽는다."""
    h = out / HISTORY
    if not h.is_file():
        return ""
    m = COMMIT_RE.search(h.read_text(errors="replace"))
    return m[1] if m else ""


# ------------------------------------------------------------------ 로컬

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_hashes(out: Path, rels: Iterable[str]) -> dict[str, str]:
    got = {}
    for rel in rels:
        p = out / rel
        if p.is_file():
            got[rel] = sha256_file(p)
    return got


def write_card_stub(out: Path, host: str, run_dir: str) -> bool:
    """카드가 없을 때만 초안을 쓴다. 카드는 사람의 파일이다 — 재등록이 status 를 되돌리면 안 된다."""
    card = out / CARD
    if card.exists():
        return False
    card.write_text(f"# 정책 카드 — 사람이 쓴다. fetch_run.py 는 이 파일을 다시 쓰지 않는다.\n"
                    f"id: {out.name}\n"
                    f"status: candidate        # candidate | verified | deployed | hold\n"
                    f"task: ''\n"
                    f"side: ''                 # left | right | both\n"
                    f"source: '{host}:{run_dir}'\n"
                    f"note: ''\n")
    return True


def _load_manifest(out: Path) -> dict | None:
    p = out / MANIFEST
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _offline(out: Path, host: str, run: str, root: str, why: str) -> int:
    man = _load_manifest(out)
    if man is None:
        print(f"[fetch] 서버에 못 붙었고 {out/MANIFEST} 도 없다: {why}")
        print(f"[fetch] 다른 호스트에서 이걸 치면 된다:\n"
              f"  scp -r {host}:{root}/{run}/params {out}/params\n"
              f"  scp {host}:{root}/{run}/nn/<체크포인트>.pth {out}/nn/\n"
              f"  scp {host}:{root}/{run}/<trace>_meta.json {out}/{META_LOCAL}")
        return 2
    rels = [f["local_rel"] for f in man.get("files", [])]
    if manifest_is_current(man, local_hashes(out, rels), None, want=rels):
        print(f"[fetch] offline: {out} 의 {len(rels)} 개 파일이 매니페스트와 일치한다 ({why})")
        return 0
    print(f"[fetch] offline: {out} 의 파일이 매니페스트와 다르다 — 서버 연결 후 --force 로 다시 받아라")
    return 3


# ------------------------------------------------------------------ CLI

def _parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--run", required=True, help="원격 런 디렉터리 이름 (예: t2r_i18)")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"ssh 호스트, 또는 '{LOCAL}' (이 PC 의 --root 아래)")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="원격 로그 루트")
    ap.add_argument("--out", type=Path, default=None, help="기본 sim2real/policies/<run>")
    ap.add_argument("--checkpoint", default="", help="best | last | ep:NNNN | <파일명>")
    ap.add_argument("--sim-meta", default="auto",
                    help="auto | none | <파일명>. pour 계열만 필요하다 — 단일팔 런은 none")
    ap.add_argument("--trace", default="none", help="none | auto | <파일명>  (npz 는 보통 143 MB)")
    ap.add_argument("--list", action="store_true", help="후보만 보여주고 끝낸다")
    ap.add_argument("--dry-run", action="store_true", help="계획만 찍고 아무것도 쓰지 않는다")
    ap.add_argument("--verify-only", action="store_true", help="이미 받은 것의 해시만 확인한다")
    ap.add_argument("--force", action="store_true", help="해시가 같아도 다시 받는다")
    return ap.parse_args(argv)


def _print_list(listing: Mapping[str, Sequence[str]]) -> None:
    cks = sorted(parse_checkpoints(listing.get("nn", ())),
                 key=lambda c: (c.epoch is None, c.epoch or 0))
    print(f"[fetch] nn/ {len(cks)} 개")
    for c in cks:
        tag = "best" if c.epoch is None else f"ep {c.epoch}"
        print(f"    {tag:>10}  rew {c.rew if c.rew is not None else '-':>12}  {c.name}")
    print("[fetch] params/ " + (", ".join(sorted(listing.get("params", ()))) or "(없음)"))
    metas = _meta_names(listing.get("", ()))
    print("[fetch] sim meta " + (", ".join(metas) or "(없음 — pour 계열이면 --sim-meta 가 필요하다, "
                                 "단일팔 런이면 --sim-meta none)"))


def _fetch(runner, host: str, run_dir: str, out: Path, plan: Sequence[FetchFile],
           rhash: Mapping[str, str], commit: str) -> int:
    staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=out))
    try:
        for f in plan:
            dst = staging / f.local_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            runner(["rsync", "-a", "--partial", _source(host, run_dir, f.remote_rel), str(dst)])
            got = sha256_file(dst)
            if got != rhash.get(f.remote_rel):
                print(f"[fetch] 해시 불일치 {f.remote_rel}: 원격 {rhash.get(f.remote_rel)} / 받은 것 {got}")
                return 3
        files = []
        for f in plan:
            final = out / f.local_rel
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / f.local_rel, final)
            files.append({"local_rel": f.local_rel, "remote_rel": f.remote_rel,
                          "sha256": rhash[f.remote_rel], "md5": md5_file(final),
                          "size": final.stat().st_size})
        (out / MANIFEST).write_text(json.dumps(
            {"host": host, "remote_dir": run_dir,
             "hdgp_head_at_fetch": commit,          # 내려받은 시점의 HEAD
             "run_commit": run_commit(out),          # 이 런이 **학습된** 커밋 (test_history.md)
             "files": files}, indent=1, ensure_ascii=False) + "\n")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if write_card_stub(out, host, run_dir):
        print(f"[fetch] {out/CARD} 초안을 썼다 — task · side · note 를 채워라")
    rc = run_commit(out)
    print(f"[fetch] {len(plan)} 개 파일 → {out}   학습 커밋 {rc[:8] or '?'} (fetch 시점 HEAD {commit[:8] or '?'})")
    for f in plan:
        print(f"    {f.remote_rel}  →  {f.local_rel}")
    return 0


def main(argv=None, runner=run_cmd) -> int:
    args = _parse(argv)
    out = args.out or POLICIES / args.run
    run_dir = f"{args.root}/{args.run}"

    try:
        listing = remote_listing(runner, args.host, run_dir)
    except RemoteError as e:
        return _offline(out, args.host, args.run, args.root, str(e))

    if args.list:
        _print_list(listing)
        return 0

    try:
        plan = plan_files(listing, checkpoint=args.checkpoint, sim_meta=args.sim_meta, trace=args.trace)
    except FetchError as e:
        print(f"[fetch] {e}")
        return 2

    if args.dry_run:
        print(f"[fetch] dry-run — {args.host}:{run_dir} → {out}")
        for f in plan:
            print(f"    {f.remote_rel}  →  {f.local_rel}")
        return 0

    rhash = remote_hashes(runner, args.host, run_dir, [f.remote_rel for f in plan])
    lhash = local_hashes(out, [f.local_rel for f in plan])
    man = _load_manifest(out)
    if man is not None and not args.force:
        by_local = {f.local_rel: rhash.get(f.remote_rel, "") for f in plan}
        seen = {f["remote_rel"]: by_local.get(f["local_rel"], "") for f in man.get("files", [])}
        if manifest_is_current(man, lhash, seen, want=[f.local_rel for f in plan]):
            print(f"[fetch] up to date — {out} ({len(plan)} 파일, 전송 0)")
            return 0
    if args.verify_only:
        print(f"[fetch] verify-only: {out} 가 원격과 다르다")
        return 3

    out.mkdir(parents=True, exist_ok=True)
    return _fetch(runner, args.host, run_dir, out, plan, rhash, remote_commit(runner, args.host))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as exc:
        print(f"[fetch] {exc}")
        raise SystemExit(2) from None
