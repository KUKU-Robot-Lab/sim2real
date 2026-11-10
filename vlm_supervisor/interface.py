"""외부 계약: 상태/카메라 구독, 승인 JSONL append, 네 Trigger 서비스.

ROS imports are deliberately lazy and confined to RosInterface.__init__.
"""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from PIL import Image

CAMERA_TOPIC = '/camera/camera/color/image_raw/compressed'
SERVICES = frozenset({'next', 'run', 'stop', 'reset'})
SHELL_ACTIONS = frozenset({'retry', 'rescan', 'home', 'insert', 'human'})
STATUS_FIELDS = {'status', 'node', 'world', 'nodes', 'last', 'pending', 'history'}


def parse_status(raw: str) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict) or not STATUS_FIELDS <= value.keys():
        raise ValueError('status JSON requires: ' + ', '.join(sorted(STATUS_FIELDS)))
    if (not isinstance(value['status'], str) or not isinstance(value['world'], dict)
            or not isinstance(value['nodes'], list) or not isinstance(value['history'], list)
            or not isinstance(value['last'], (dict, type(None)))
            or not isinstance(value['node'], (str, dict, type(None)))):
        raise ValueError('invalid status field types')
    return value


def node_id(node) -> str | None:
    if isinstance(node, str):
        return node or None
    if isinstance(node, dict):
        value = node.get('id') or node.get('node')
        return value if isinstance(value, str) and value else None
    return None


def write_approval(what: str, operator: str, path, runner: str) -> dict:
    if not all(isinstance(v, str) and v.strip() for v in (what, operator, runner)):
        raise ValueError('what/operator/runner must be nonempty strings')
    now = time.time()
    row = {'what': what, 't': now, 'operator': operator,
           'at': time.strftime('%F %T', time.localtime(now)), 'runner': runner}
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One append write preserves the existing runner JSONL format.
    data = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise OSError('short approval write')
    finally:
        os.close(fd)
    return row


class Interface:
    """Transport boundary; tests may implement status/frame/call without ROS."""

    def __init__(self, runner: str, approvals, operator='vlm_supervisor'):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', runner):
            raise ValueError('runner must be a ROS name segment')
        self.runner = runner
        self.approvals = Path(approvals).expanduser()
        self.operator = operator
        self.events: list[dict] = []

    def status(self) -> dict:
        raise NotImplementedError

    def frame(self, after=0.0, wait=3.0):
        raise NotImplementedError

    def call(self, service: str) -> dict:
        raise NotImplementedError

    def shell_command(self, verb: str, args: dict):
        return None

    def perform(self, action: str, what: str | None = None, *, active=False, args=None) -> dict:
        if action in SHELL_ACTIONS and active:
            payload = dict(args or {})
            if action == 'retry' and what and 'node_id' not in payload:
                payload['node_id'] = what
            ack = self.shell_command(action, payload)
            if ack is not None:
                return ack
        if action not in SERVICES and action != 'wait':
            event = {'error': 'NotSupported', 'action': action, 'human_required': True,
                     'reason': 'shell unavailable or ack timeout; stop and ask operator', 'shadow': not active}
            self.events.append(event)
            if active:
                event['stop'] = self.call('stop')
            return event
        if not active or action == 'wait':
            return {'ok': True, 'shadow': not active, 'action': action}
        if action in {'next', 'run'}:
            if not what:
                raise ValueError('next/run require an explicit approval target')
            write_approval(what, self.operator, self.approvals, self.runner)
        return self.call(action)

    def close(self):
        pass


class RosInterface(Interface):
    def __init__(self, runner, approvals, operator='vlm_supervisor', status_timeout=10.0):
        super().__init__(runner, approvals, operator)
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CompressedImage
            from std_msgs.msg import String
            from std_srvs.srv import Trigger
        except ImportError as exc:
            raise RuntimeError('ROS 2 환경을 source해야 RosInterface를 사용할 수 있습니다') from exc
        self.rclpy, self.Trigger, self.String = rclpy, Trigger, String
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = rclpy.create_node('vlm_supervisor')
        self.lock = threading.Lock()
        self._status: dict = {}
        self._status_at = 0.0
        self.shell_condition = threading.Condition()
        self.shell_acks = {}
        self.shell_timeout = 180.0
        self.shell_pub = self.node.create_publisher(String, f'/vlm_shell/{runner}/cmd', 10)
        self.node.create_subscription(String, f'/vlm_shell/{runner}/ack', self._on_shell_ack, 10)
        self.status_timeout = status_timeout
        self._image = None
        self._image_at = 0.0
        self.node.create_subscription(String, f'/policy_control/status/{runner}', self._on_status, 10)
        self.node.create_subscription(CompressedImage, CAMERA_TOPIC, self._on_image, qos_profile_sensor_data)
        self.clients = {name: self.node.create_client(Trigger, f'/policy_control/{runner}/{name}')
                        for name in SERVICES}
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _on_shell_ack(self, message):
        try:
            ack = json.loads(message.data)
            if (not isinstance(ack, dict) or not isinstance(ack.get('cmd_id'), str)
                    or type(ack.get('ok')) is not bool):
                return
        except (ValueError, TypeError):
            return
        with self.shell_condition:
            if ack['cmd_id'] in self.shell_acks:
                self.shell_acks[ack['cmd_id']] = ack
                self.shell_condition.notify_all()

    def shell_command(self, verb, args):
        cmd_id = uuid.uuid4().hex
        message = self.String()
        message.data = json.dumps({'cmd_id': cmd_id, 'verb': verb, 'args': args})
        # Avoid waiting the full recovery timeout when no shell is connected.
        discovery_deadline = time.monotonic() + 3
        while self.shell_pub.get_subscription_count() == 0:
            if time.monotonic() >= discovery_deadline:
                return None
            time.sleep(0.05)
        with self.shell_condition:
            self.shell_acks[cmd_id] = None
            try:
                self.shell_pub.publish(message)
                deadline = time.monotonic() + self.shell_timeout
                while self.shell_acks[cmd_id] is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self.shell_condition.wait(timeout=remaining)
                return self.shell_acks[cmd_id]
            finally:
                self.shell_acks.pop(cmd_id, None)

    def _on_status(self, message):
        try:
            value = parse_status(message.data)
        except (ValueError, TypeError) as exc:
            self.node.get_logger().warning(f'invalid status: {exc}')
            return
        with self.lock:
            self._status, self._status_at = value, time.monotonic()

    def _on_image(self, message):
        with self.lock:
            self._image, self._image_at = bytes(message.data), time.monotonic()

    def status(self):
        with self.lock:
            if time.monotonic() - self._status_at > self.status_timeout:
                return {}
            return copy.deepcopy(self._status)

    def frame(self, after=0.0, wait=3.0):
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            with self.lock:
                data = self._image if self._image_at >= after else None
            if data:
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        return image.convert('RGB')
                except OSError:
                    return None
            time.sleep(0.05)
        return None

    def call(self, service):
        if service not in SERVICES:
            raise ValueError('unsupported Trigger service')
        client = self.clients[service]
        if not client.wait_for_service(timeout_sec=3):
            return {'ok': False, 'reasons': ['service unavailable: ' + service]}
        future = client.call_async(self.Trigger.Request())
        deadline = time.monotonic() + 8
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            future.cancel()
            return {'ok': False, 'reasons': ['service response timeout']}
        try:
            response = future.result()
            try:
                body = json.loads(response.message)
            except (ValueError, TypeError):
                body = {}
            if not isinstance(body, dict):
                body = {}
            return {'ok': bool(response.success) and body.get('ok', True) is True,
                    'reasons': body.get('reasons', [response.message])}
        except Exception as exc:
            return {'ok': False, 'reasons': [str(exc)]}

    def close(self):
        self.executor.shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()
        if self.owns_context:
            self.rclpy.try_shutdown()
