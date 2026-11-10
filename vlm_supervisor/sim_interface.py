"""hdgp sim 브리지(HTTP)용 Interface. rclpy 없이 RosInterface 와 같은 반환 모양을 낸다."""
from __future__ import annotations

import io
import json
import time

import requests
from PIL import Image

from .interface import Interface, parse_status

# Supervisor call() names -> bridge /cmd verbs; others are not offered by the bridge.
CALL_VERBS = {'next': 'next', 'stop': 'stop'}
SHELL_VERBS = frozenset({'retry', 'rescan', 'home'})


class SimInterface(Interface):
    def __init__(self, url='http://127.0.0.1:8790', approvals='logs/runs/episode_approvals.jsonl',
                 operator='vlm_supervisor', runner='sim_bridge'):
        super().__init__(runner, approvals, operator)
        self.url = url.rstrip('/')
        self.http = requests.Session()
        self.http.trust_env = False  # HTTP(S)_PROXY would otherwise relay the local bridge and mask connect errors

    def _error(self, where, exc):
        self.events.append({'error': 'BridgeError', 'where': where, 'reason': str(exc)})

    def status(self):
        try:
            response = self.http.get(self.url + '/status', timeout=3)
            response.raise_for_status()
            value = response.json()
            if isinstance(value, dict):
                value.setdefault('history', [])  # bridge reports attempts/episode_id instead of history
            return parse_status(json.dumps(value))
        except (requests.RequestException, ValueError, TypeError) as exc:
            # RosInterface reports a missing/invalid status as {} (status outage).
            self._error('status', exc)
            return {}

    def frame(self, after=0.0, wait=3.0):
        # after is time.monotonic(); X-Frame-Time is epoch on the same host.
        offset = time.time() - time.monotonic()
        deadline = time.monotonic() + wait
        error = None
        while time.monotonic() < deadline:
            try:
                response = self.http.get(self.url + '/frame', timeout=3)
                if response.status_code == 200 and float(response.headers['X-Frame-Time']) - offset > after:
                    with Image.open(io.BytesIO(response.content)) as image:
                        return image.convert('RGB')
                if response.status_code not in (200, 404):
                    response.raise_for_status()
            except (requests.RequestException, KeyError, ValueError) as exc:
                error = exc
            except OSError:  # undecodable JPEG, same as RosInterface
                return None
            time.sleep(0.05)
        if error is not None:
            self._error('frame', error)
        return None

    def _cmd(self, verb, args):
        try:
            response = self.http.post(self.url + '/cmd', json={'verb': verb, 'args': args}, timeout=8)
        except requests.RequestException as exc:
            self._error('cmd', exc)
            return {'ok': False, 'reasons': [f'service unavailable: {verb} ({exc})']}
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return {'ok': False, 'reasons': [f'HTTP {response.status_code}: {response.text[:200]}']}
        reasons = body.get('reasons')
        return {'ok': response.ok and body.get('ok') is True,
                'reasons': reasons if isinstance(reasons, list) else [f'HTTP {response.status_code}']}

    def call(self, service):
        if service not in {'next', 'run', 'stop', 'reset'}:
            raise ValueError('unsupported Trigger service')
        if service not in CALL_VERBS:
            return {'ok': False, 'reasons': ['service unavailable: ' + service + ' (sim bridge)']}
        return self._cmd(CALL_VERBS[service], {})

    def shell_command(self, verb, args):
        # None = no shell for this verb; Interface.perform turns it into NotSupported + stop.
        return self._cmd(verb, args) if verb in SHELL_VERBS else None

    def close(self):
        self.http.close()
