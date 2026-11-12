#!/usr/bin/env python3
"""Head JSON-lines helper. stdout is protocol only; EOF/lease expiry restores; disables only after verified restoration."""
from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

S2R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S2R / "scripts"))


class SessionEnded(BaseException):
    pass


def watch_stdin():
    """A 15 s renewable lease also handles a half-open SSH connection."""
    requests = queue.Queue()
    last = [time.monotonic()]

    def end(*_):
        raise SessionEnded()

    signal.signal(signal.SIGTERM, end)
    signal.signal(signal.SIGHUP, end)

    def reader():
        try:
            for line in sys.stdin:
                last[0] = time.monotonic()
                request = json.loads(line)
                if request.get("cmd") != "ping":
                    requests.put(request)
        finally:
            os.kill(os.getpid(), signal.SIGTERM)

    def lease():
        while True:
            time.sleep(1)
            if time.monotonic() - last[0] > 15:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=lease, daemon=True).start()
    return requests


def cleanup_signals():
    # EOF must not interrupt the bounded restoration already in progress.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


@contextlib.contextmanager
def head_deadline(seconds=20):
    """Bound bring_to + verify (including SDK calls); timeout -k 10 is the outer backstop."""
    def expired(*_):
        raise TimeoutError(f"head operation exceeded {seconds} s")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class LocalHead:
    def __init__(self):
        from head_compliant_hold import CompliantController
        from head_home import load_head_home
        self.cfg = load_head_home(S2R / "config/head_home_rh56f1.yaml")
        self.c = CompliantController(self.cfg.port, self.cfg.baud)
        self.start = None
        self.moved = False
        self.restored = False
        self.restore_error = None
        try:
            with head_deadline():
                for i in self.cfg.targets_deg:
                    self.c.ping(i)
                self.start = self.ticks()
        except BaseException:
            self.c.close()  # constructor has not written torque or goals
            raise

    def ticks(self):
        return {self.cfg.names[i]: int(self.c.read_present_tick(i)) for i in self.cfg.targets_deg}

    def goto_deg(self, targets):
        from head_home import HeadHome, bring_to, verify
        ids = {n: i for i, n in self.cfg.names.items()}
        cfg = HeadHome(**{**self.cfg.__dict__, "targets_deg": {ids[n]: float(v) for n, v in targets.items()}})
        self.moved, self.restored = True, False
        with head_deadline():
            bring_to(self.c, cfg, tuple(cfg.targets_deg))
            if not verify(self.c, cfg):
                self.restore_error = "목 검증 ±0.3° 밖"
                raise RuntimeError(self.restore_error)

    def back_to_start(self):
        from head_position_hold_node import tick_to_deg
        if self.restore_error:
            raise RuntimeError(self.restore_error)
        if self.restored or not self.moved or self.start is None:
            return
        try:
            tgt = {n: tick_to_deg(t) for n, t in self.start.items()}
            home = {self.cfg.names[i]: d for i, d in self.cfg.targets_deg.items()}
            if any(abs(tgt[n] - home[n]) > 40 for n in tgt):
                print(f"목 시작 자리 {self.start} 가 홈에서 40° 넘게 떨어져 있어 홈으로 복원", file=sys.stderr)
                tgt = home
            self.goto_deg(tgt)
            self.restored = True
        except Exception as e:
            self.restore_error = repr(e)
            raise

    def close(self):
        from head_position_hold_node import ADDR_GOAL_POSITION, ADDR_TORQUE_ENABLE, TORQUE_OFF
        try:
            self.back_to_start()
        except Exception as e:
            # Restore failed: never let gravity drop the head. Keep torque and seed
            # the present raw position as goal (no mode/torque-on/profile changes).
            errors = {}
            for i in self.cfg.targets_deg:
                try:
                    with head_deadline(2):
                        tick = self.c.read_present_tick(i)
                        self.c.write4(i, ADDR_GOAL_POSITION, tick, "restore failed: hold here")
                except Exception as hold_error:
                    errors[i] = repr(hold_error)
            result = {"error": repr(e), "torque_held": True, "hold_errors": errors}
        else:
            errors = {}
            for i in self.cfg.targets_deg:
                try:
                    with head_deadline(2):
                        self.c.write1(i, ADDR_TORQUE_ENABLE, TORQUE_OFF, "torque off")
                except Exception as e:
                    errors[i] = repr(e)
            result = {"closed": True, "torque_held": False}
            if errors:
                result = {"error": f"torque off failed: {errors}", "torque_held": True}
        finally:
            self.c.close()
        return result


def main():
    head = None
    protocol = sys.stdout
    status = 0
    try:
        requests = watch_stdin()
        with contextlib.redirect_stdout(sys.stderr):
            head = LocalHead()
        print(json.dumps({"start": head.start, "targets": {head.cfg.names[i]: d for i, d in head.cfg.targets_deg.items()}}), file=protocol, flush=True)
        while True:
            request = requests.get()
            cmd = request["cmd"]
            with contextlib.redirect_stdout(sys.stderr):
                if cmd == "read":
                    result = head.ticks()
                elif cmd == "goto":
                    head.goto_deg(request["targets"])
                    result = head.ticks()
                elif cmd == "restore":
                    head.back_to_start()
                    result = head.ticks()
                elif cmd == "close":
                    break
                else:
                    raise ValueError(f"unknown command: {cmd}")
            print(json.dumps({"ticks": result}), file=protocol, flush=True)
    except SessionEnded:
        pass
    except Exception as e:
        print(json.dumps({"error": repr(e), "torque_held": bool(head and head.moved)}), file=protocol, flush=True)
        status = 1
    finally:
        cleanup_signals()
        if head:
            with contextlib.redirect_stdout(sys.stderr):
                try:
                    result = head.close()
                except Exception as e:
                    result = {"error": repr(e), "torque_held": True}
            if result.get("error"):
                status = 1
            try:
                print(json.dumps(result), file=protocol, flush=True)
            except BrokenPipeError:
                print(json.dumps(result), file=sys.stderr, flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
