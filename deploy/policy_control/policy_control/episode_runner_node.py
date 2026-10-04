"""episode_runner_node — 에피소드 하나를 들고 상황판 · CLI 의 명령으로 굴린다(10.04 사용자: 구분 실행 → 최종 연속 실행).

  srv  /policy_control/episode_runner/{next,run,stop,reset}  (std_srvs/Trigger, 응답 message = {"ok","reasons"} JSON)
         next  지금 노드 하나(또는 기다리는 복구 하나) — 승인 = 그 이름(next_action)
         run   끝까지 — 승인 = "episode:<이름>" 한 번(복구도 그 안, 재시도 상한 · 안전 정지는 그대로)
         stop  돌던 것을 멈춘다(정책 노드 episode/stop → pd 가 붙든다 · 재생 프로세스 SIGTERM) — 승인 없음
         reset 새 에피소드(WorldState 처음부터) — 도는 중이면 거부
  out  /policy_control/status/episode_runner (2 Hz JSON — EpisodeManager.view() + busy · auto_approve)
  log  logs/episode/<episode_id>/episode.jsonl · scene.json · tools.log

승인: auto_approve(fake 도메인만 — 실기 126 이면 노드가 거부) 또는 approvals 파일(JSONL, tools/episode_cmd.py 와 상황판이
쓴다)에 이 노드가 뜬 뒤 쓰인 {"what": <이름>, "t": epoch} 가 있고 아직 안 쓴 것. 한 줄은 한 번만 쓴다.

    python3 deploy/policy_control/policy_control/episode_runner_node.py --ros-args \\
        -p episode:=config/episodes/pick_place_right.yaml -p robot:=rh56f1 [-p auto_approve:=true]
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "policy_control"     # noqa: A001

from policy_control import _paths  # noqa: E402
from policy_control import episode_spec as S  # noqa: E402
from policy_control.episode_runner import EpisodeManager, jsonl_logger  # noqa: E402

NS = "/policy_control"
NAME = "episode_runner"
REAL_DOMAIN = "126"
APPROVALS = _paths.SIM2REAL / "logs" / "episode" / "approvals.jsonl"


class Approvals:
    """approvals 파일에서 이 이름의 새 승인 한 줄을 꺼낸다(뜬 뒤 쓰인 것 · 한 번만)."""

    def __init__(self, path: Path, since: float) -> None:
        self.path, self.since, self.used = path, since, set()

    def take(self, what: str) -> bool:
        if not self.path.is_file():
            return False
        for i, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            key = (i, row.get("what"), row.get("t"))
            if row.get("what") == what and float(row.get("t", 0)) >= self.since and key not in self.used:
                self.used.add(key)
                return True
        return False


def write_approval(what: str, operator: str, path: Path = APPROVALS) -> dict:
    row = {"what": what, "t": time.time(), "operator": operator, "at": time.strftime("%F %T")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def _main_node():
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from policy_control.episode_ros import RosExecutor

    class EpisodeRunnerNode(Node):
        def __init__(self) -> None:
            super().__init__(NAME)
            for name, default in (("episode", ""), ("robot", "rh56f1"), ("auto_approve", False),
                                  ("approvals", str(APPROVALS))):
                self.declare_parameter(name, default)
            p = lambda n: self.get_parameter(n).value  # noqa: E731
            path = Path(str(p("episode")))
            path = path if path.is_absolute() else _paths.SIM2REAL / path
            self.ep = S.load(path)
            self.auto = bool(p("auto_approve"))
            if self.auto and os.environ.get("ROS_DOMAIN_ID", "") in ("", "0", REAL_DOMAIN):
                raise SystemExit(f"auto_approve 는 fake 도메인에서만 — ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')!r}")
            self.approvals = Approvals(Path(str(p("approvals"))), time.time())
            self.robot = str(p("robot"))
            self._busy = threading.Lock()
            self._worker: threading.Thread | None = None
            self._String = String
            self._pub = self.create_publisher(String, f"{NS}/status/{NAME}", QoSProfile(depth=10))
            self.executor_ros = None
            self.manager = None
            self._new_episode()
            for name in ("next", "run", "stop", "reset"):
                self.create_service(Trigger, f"{NS}/{NAME}/{name}", getattr(self, f"_srv_{name}"))
            self.create_timer(0.5, self._publish)
            self.get_logger().info(f"{NAME} up · {self.ep.name} ({len(self.ep.nodes)} 노드) · 승인 "
                                   f"{'auto(fake)' if self.auto else self.approvals.path}")

        def _new_episode(self) -> None:
            if self.executor_ros is None:
                eid = time.strftime("%Y%m%d_%H%M%S")
                run_dir = _paths.SIM2REAL / "logs" / "episode" / f"{eid}_{self.ep.name}"
                self.executor_ros = RosExecutor(self, self.ep, robot=self.robot, run_dir=run_dir)
            ex = self.executor_ros
            ex.clear_cancel()
            eid = time.strftime("%Y%m%d_%H%M%S")
            log = jsonl_logger(ex.run_dir / f"episode_{eid}.jsonl")
            self.manager = EpisodeManager(self.ep, ex, approve=self._approve, log=log, episode_id=eid)

        def _approve(self, what: str, why: str) -> bool:
            ok = self.auto or self.approvals.take(what)
            self.get_logger().info(f"승인 {'✓' if ok else '없음'} {what} — {why}")
            return ok

        def _reply(self, res, ok: bool, reasons):
            res.success = bool(ok)
            res.message = json.dumps({"ok": bool(ok), "reasons": [str(r) for r in reasons]}, ensure_ascii=False)
            return res

        def _start(self, fn, label: str, res):
            if self._worker is not None and self._worker.is_alive():
                return self._reply(res, False, [f"busy — {self.manager.node.id if self.manager.node else '?'} 실행 중"])

            def work():
                try:
                    out = fn()
                    self.get_logger().info(f"{label} → {out} · {self.manager.last}")
                except Exception as exc:              # 실행기 밖 예외 — 멈추고 남긴다
                    self.get_logger().error(f"{label} 예외: {type(exc).__name__}: {exc}")
                    self.manager.stop(f"runner exception: {exc}")
            self._worker = threading.Thread(target=work, daemon=True)
            self._worker.start()
            return self._reply(res, True, [f"{label} started · next {self.manager.next_action()}"])

        def _srv_next(self, _req, res):
            return self._start(self.manager.step, "next", res)

        def _srv_run(self, _req, res):
            return self._start(self.manager.run, "run", res)

        def _srv_stop(self, _req, res):
            self.manager.stop("operator stop (episode_runner/stop)")
            return self._reply(res, True, ["stopped"])

        def _srv_reset(self, _req, res):
            if self._worker is not None and self._worker.is_alive():
                return self._reply(res, False, ["running — stop first"])
            self._new_episode()
            return self._reply(res, True, [f"new episode {self.manager.episode_id}"])

        def _publish(self) -> None:
            body = {"node": NAME, "phase": self.manager.status, "ok": self.manager.status != "FAILURE",
                    "busy": bool(self._worker is not None and self._worker.is_alive()), "auto_approve": self.auto,
                    "run_dir": str(self.executor_ros.run_dir), **self.manager.view(),
                    "t_pub_ns": self.get_clock().now().nanoseconds}
            self._pub.publish(self._String(data=json.dumps(body, ensure_ascii=False, default=str)))

    return EpisodeRunnerNode


def main(argv=None) -> int:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init(args=argv)
    node = None
    try:
        node = _main_node()()
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(node)
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
