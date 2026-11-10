"""ROS-free transaction controller and lazy ROS/process adapter."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import yaml

from .derive import derive, node_side, walk

ROOT = Path(__file__).resolve().parents[2]


class ShellController:
    """Serialize generations; adapter owns all ROS and process effects."""

    def __init__(self, episode, episode_path, run_dir, adapter):
        self.original = deepcopy(episode)
        self.episode = deepcopy(episode)
        self.path = Path(episode_path).resolve()
        self.directory = Path(run_dir).resolve() / 'shell'
        self.adapter = adapter
        self.generation = 0
        self.lineage = []
        self.cache = {}
        self.lock = threading.Lock()
        self.serial = 0

    def status(self):
        return {'generation': self.generation, 'episode_path': str(self.path),
                'lineage': deepcopy(self.lineage), 'runner_pid': self.adapter.pid()}

    def handle(self, command):
        cmd_id = command.get('cmd_id') if isinstance(command, dict) else None
        if not isinstance(cmd_id, str) or not cmd_id:
            return {'cmd_id': cmd_id, 'ok': False, 'reasons': ['cmd_id must be nonempty string'],
                    **self.status()}
        with self.lock:
            if cmd_id in self.cache:
                return deepcopy(self.cache[cmd_id])
            reasons = []
            stopped = False
            candidate = None
            try:
                st = self.adapter.status()
                if not st or st.get('busy') or st.get('status') not in ('READY', 'FAILURE', 'STOPPED', 'SUCCESS'):
                    raise ValueError('runner busy or status unavailable/not quiescent')
                command = deepcopy(command)
                args = command.setdefault('args', {})
                if command.get('verb') == 'home' and isinstance(args, dict) and not args.get('side'):
                    failed = (st.get('last') or {}).get('node')
                    args['side'] = node_side(self.original, failed) or node_side(self.episode, failed)
                source = self.episode
                if command.get('verb') == 'retry' and isinstance(args, dict):
                    if any(n.get('id') == args.get('node_id') for n in walk(self.original.get('sequence', []))):
                        source = self.original
                derived = derive(source, st, command)
                if not self.adapter.stop().get('ok'):
                    raise RuntimeError('runner stop service rejected')
                self.adapter.terminate()
                stopped = True
                self.serial += 1
                candidate = self.directory / f'gen_{self.serial}.yaml'
                self.directory.mkdir(parents=True, exist_ok=True)
                # Never overwrite an earlier generation, including after shell restart.
                while candidate.exists():
                    self.serial += 1
                    candidate = self.directory / f'gen_{self.serial}.yaml'
                candidate.write_text(yaml.safe_dump(derived, allow_unicode=True, sort_keys=False), encoding='utf-8')
                if not self.adapter.validate(candidate):
                    raise RuntimeError('derived episode --plan validation failed')
                self.adapter.start(candidate)
                if not self.adapter.wait_ready():
                    raise RuntimeError('new runner did not reach READY')
                self.lineage.append({'generation': self.serial, 'parent': self.generation,
                                     'cmd_id': cmd_id, 'verb': command['verb'], 'episode_path': str(candidate)})
                self.generation, self.path, self.episode = self.serial, candidate, derived
            except Exception as exc:
                reasons.append(str(exc))
                if stopped:
                    try:
                        self.adapter.terminate()
                        self.adapter.start(self.path)
                        if not self.adapter.wait_ready():
                            raise RuntimeError('rollback runner did not reach READY')
                    except Exception as rollback:
                        reasons.append(f'rollback failed: {rollback}')
            ack = {'cmd_id': cmd_id, 'ok': not reasons, 'reasons': reasons, **self.status()}
            self.cache[cmd_id] = ack
            return deepcopy(ack)


class RosProcessAdapter:
    def __init__(self, node, runner, runner_cmd, ready_timeout=30.0):
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        self.node, self.runner, self.runner_cmd = node, runner, list(runner_cmd)
        self.ready_timeout = ready_timeout
        self.process = None
        self.current = {}
        self.condition = threading.Condition()
        self.Trigger = Trigger
        self.client = node.create_client(Trigger, f'/policy_control/{runner}/stop')
        self.subscription = node.create_subscription(
            String, f'/policy_control/status/{runner}', self._status, 10)
        self.received_at = 0.0

    def _status(self, message):
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict):
                return
        except (ValueError, TypeError):
            return
        with self.condition:
            self.current = value
            self.received_at = time.monotonic()
            self.condition.notify_all()

    def status(self):
        with self.condition:
            return deepcopy(self.current) if time.monotonic() - self.received_at < 3 else {}

    def pid(self):
        return self.process.pid if self.process is not None and self.process.poll() is None else None

    def stop(self):
        if not self.client.wait_for_service(timeout_sec=3):
            return {'ok': False}
        future = self.client.call_async(self.Trigger.Request())
        deadline = time.monotonic() + 8
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            future.cancel()
            return {'ok': False}
        response = future.result()
        if response is None or not response.success:
            return {'ok': False}
        try:
            body = json.loads(response.message)
        except (ValueError, TypeError):
            body = {}
        return {'ok': not isinstance(body, dict) or body.get('ok', True) is True}

    def terminate(self):
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)

    def validate(self, path):
        proc = subprocess.run([sys.executable, str(ROOT / 'deploy/policy_control/tools/episode_run.py'),
                               '--episode', str(path), '--plan'], cwd=ROOT, timeout=60,
                              capture_output=True, text=True,
                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        if proc.returncode:
            self.node.get_logger().error((proc.stdout + proc.stderr)[-4000:])
        return proc.returncode == 0

    def start(self, path):
        if self.pid() is not None:
            raise RuntimeError('owned runner still alive')
        # Appending ROS overrides preserves robot/approvals/object parameters in
        # the supplied command while pinning episode and approval runner identity.
        command = self.runner_cmd + ['--ros-args', '-r', f'__node:={self.runner}',
                                     '-p', f'episode:={Path(path).resolve()}']
        with self.condition:
            self.current, self.received_at = {}, 0.0
        self.process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

    def wait_ready(self):
        deadline = time.monotonic() + self.ready_timeout
        with self.condition:
            while time.monotonic() < deadline:
                if self.process is None or self.process.poll() is not None:
                    return False
                if self.current.get('status') == 'READY' and not self.current.get('busy'):
                    return True
                self.condition.wait(timeout=min(0.2, max(0, deadline - time.monotonic())))
        return False


def run_node(args):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node('vlm_shell_' + args.runner)
    adapter = RosProcessAdapter(node, args.runner, args.runner_cmd, args.ready_timeout)
    episode_path = Path(args.episode).expanduser().resolve()
    controller = ShellController(yaml.safe_load(episode_path.read_text()), episode_path, args.run_dir, adapter)
    base = f'/vlm_shell/{args.runner}'
    ack_pub = node.create_publisher(String, base + '/ack', 10)
    status_pub = node.create_publisher(String, base + '/status', 10)
    workers = []

    def publish(pub, value):
        message = String()
        message.data = json.dumps(value, ensure_ascii=False)
        pub.publish(message)

    def on_command(message):
        try:
            command = json.loads(message.data)
        except (ValueError, TypeError):
            publish(ack_pub, {'cmd_id': None, 'ok': False, 'reasons': ['invalid JSON']})
            return
        # A worker may wait for service/status while the executor keeps receiving.
        def work():
            publish(ack_pub, controller.handle(command))
        thread = threading.Thread(target=work, daemon=True)
        workers[:] = [worker for worker in workers if worker.is_alive()]
        workers.append(thread)
        thread.start()

    node.create_subscription(String, base + '/cmd', on_command, 10)
    node.create_timer(0.5, lambda: publish(status_pub, controller.status()))
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        adapter.start(episode_path)
        if not adapter.wait_ready():
            raise RuntimeError('initial runner did not reach READY')
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for worker in workers:
            worker.join(timeout=100)
        adapter.terminate()
        executor.shutdown()
        spinner.join(timeout=3)
        node.destroy_node()
        rclpy.try_shutdown()
