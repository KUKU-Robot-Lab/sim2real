"""Video forwarding never starts ROS nodes or opens a camera."""
from unittest.mock import Mock
import subprocess

import pytest

import perception_launcher_node as launcher


def test_video_tunnel_uses_loopback_and_stops_only_its_child(monkeypatch):
    monkeypatch.setattr(launcher.socket, 'create_connection', Mock(side_effect=OSError))
    proc = Mock()
    proc.wait.side_effect = [subprocess.TimeoutExpired('ssh', 1), 0]
    proc.poll.return_value = None
    popen = Mock(return_value=proc)
    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    remote = launcher.RemoteExec('vision-3090')
    remote.ensure_video_tunnel()
    argv = popen.call_args.args[0]
    assert argv[-1] == 'vision-3090'
    assert '127.0.0.1:8080:127.0.0.1:8080' in argv
    assert 'ExitOnForwardFailure=yes' in argv
    remote.close_video_tunnel()
    proc.terminate.assert_called_once()
    assert remote._video_tunnel is None


@pytest.mark.parametrize('existing,host', [(True, 'vision-3090'), (False, 'local')])
def test_local_video_or_existing_tunnel_is_not_replaced(monkeypatch, existing, host):
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(launcher.socket, 'create_connection', Mock(return_value=connection))
    popen = Mock()
    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    remote = launcher.RemoteExec(host)
    remote.ensure_video_tunnel()
    remote.close_video_tunnel()
    popen.assert_not_called()


def test_tunnel_start_failure_is_reported(monkeypatch):
    monkeypatch.setattr(launcher.socket, 'create_connection', Mock(side_effect=OSError))
    proc = Mock()
    proc.poll.return_value = 255
    proc.stderr.read.return_value = b'forwarding failed'
    monkeypatch.setattr(launcher.subprocess, 'Popen', Mock(return_value=proc))
    remote = launcher.RemoteExec('vision-3090')
    with pytest.raises(RuntimeError, match='forwarding failed'):
        remote.ensure_video_tunnel()
    assert remote._video_tunnel is None
