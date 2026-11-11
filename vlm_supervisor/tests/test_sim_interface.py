import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import time

import pytest
from PIL import Image

from vlm_supervisor.judge import Judge
from vlm_supervisor.sim_interface import SimInterface
from vlm_supervisor.supervisor import Supervisor


@pytest.fixture(autouse=True)
def require_local_socket():
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
    except PermissionError:
        pytest.skip('sandbox denies local TCP sockets; HTTP integration unverified')


def jpeg(color):
    buf = io.BytesIO()
    Image.new('RGB', (4, 4), color).save(buf, 'JPEG')
    return buf.getvalue()


class Bridge:
    def __init__(self):
        self.state = {'status': 'READY', 'busy': False, 'confirm': None, 'episode_id': 'ep1',
                      'node': 'grasp_blue', 'next_action': 'next', 'pending': None, 'last': {},
                      'nodes': [{'id': 'grasp_blue'}], 'world': {'objects': {}}, 'tick': 0,
                      'sim_t': 0.0, 'attempts': {}}
        self.frame = None  # (epoch, bytes)
        self.cmds = []
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body, ctype='application/json', headers=()):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/status':
                    self.reply(200, json.dumps(bridge.state).encode())
                elif self.path == '/frame' and bridge.frame:
                    self.reply(200, bridge.frame[1], 'image/jpeg', [('X-Frame-Time', str(bridge.frame[0]))])
                else:
                    self.reply(404, b'{}')

            def do_POST(self):
                cmd = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                bridge.cmds.append(cmd)
                if bridge.state['busy']:
                    self.reply(409, json.dumps({'ok': False, 'reasons': ['busy']}).encode())
                else:
                    self.reply(200, json.dumps({'ok': True, 'reasons': []}).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def bridge():
    b = Bridge()
    yield b
    b.server.shutdown()
    b.server.server_close()


def test_status_frame_cmd_roundtrip(bridge, tmp_path):
    sim = SimInterface(bridge.url, tmp_path / 'a.jsonl')
    st = sim.status()
    assert st['status'] == 'READY' and st['episode_id'] == 'ep1' and st['history'] == []
    assert sim.frame(wait=0.2) is None  # 404 until a frame exists
    bridge.frame = (time.time(), jpeg('red'))
    image = sim.frame(wait=1)
    assert image.mode == 'RGB' and image.getpixel((1, 1))[0] > 200
    assert sim.call('next') == {'ok': True, 'reasons': []}
    assert sim.shell_command('retry', {'node_id': 'grasp_blue'})['ok']
    assert sim.shell_command('insert', {}) is None
    assert sim.call('reset')['ok'] is False
    assert bridge.cmds == [{'verb': 'next', 'args': {}}, {'verb': 'retry', 'args': {'node_id': 'grasp_blue'}}]


def test_frame_after_filter(bridge, tmp_path):
    sim = SimInterface(bridge.url, tmp_path / 'a.jsonl')
    bridge.frame = (time.time(), jpeg('red'))
    after = time.monotonic()
    assert sim.frame(after=after, wait=0.3) is None  # old frame ignored
    bridge.frame = (time.time() + 0.01, jpeg('blue'))
    assert sim.frame(after=after, wait=1).getpixel((1, 1))[2] > 200


def test_busy_rejected(bridge, tmp_path):
    bridge.state['busy'] = True
    sim = SimInterface(bridge.url, tmp_path / 'a.jsonl')
    assert sim.call('next') == {'ok': False, 'reasons': ['busy']}


def test_connection_failure(tmp_path):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    sim = SimInterface(f'http://127.0.0.1:{port}', tmp_path / 'a.jsonl')
    assert sim.status() == {}
    assert sim.frame(wait=0.2) is None
    result = sim.call('stop')
    assert result['ok'] is False and 'service unavailable' in result['reasons'][0]
    assert {e['where'] for e in sim.events} == {'status', 'frame', 'cmd'}


class Backend:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def ask(self, images, text, level='low', **kwargs):
        self.prompts.append(text)
        return self.answer, {'model': 'fake'}


def test_supervisor_ready_sends_next(bridge, tmp_path):
    sim = SimInterface(bridge.url, tmp_path / 'approvals.jsonl')
    sup = Supervisor(sim, Judge(Backend('{}')), mode='active')
    sup.step(sim.status())
    assert bridge.cmds == [{'verb': 'next', 'args': {}}]
    assert json.loads(sim.approvals.read_text())['what'] == 'grasp_blue'
