"""DDS 를 허브 랜 NIC 에 묶는 Fast-DDS 프로파일 — config/dds/lan_profile.sh · fastdds_lan.xml.in (10.30).

5090 ↔ 4090 실기 DDS(도메인 50)가 Wi-Fi · tailscale · docker 브리지로 새지 않게 한다. 실측 근거는 lan_profile.sh 머리말.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SH = REPO / "config" / "dds" / "lan_profile.sh"
TMPL = REPO / "config" / "dds" / "fastdds_lan.xml.in"
ENV_FILE = REPO / "config" / "ros_domain.env"
SITE_V2 = REPO / "deploy" / "policy_control" / "config" / "sites" / "arm_v2.yaml"


def _source(extra: dict[str, str], tmp: Path) -> tuple[dict[str, str], str]:
    """set -e 호출자(common.sh · console.sh)처럼 source 하고 결과 변수를 읽는다."""
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp), "XDG_RUNTIME_DIR": str(tmp), **extra}
    script = (f'set -eo pipefail; source "{SH}"; echo "F=${{FASTRTPS_DEFAULT_PROFILES_FILE:-}}"; '
              f'echo "A=${{#FPP_DDS_ARGS[@]}}"; echo "IP=${{S2R_LAN_IP:-}}"')
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    kv = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    return kv, out.stderr


def test_template_uses_the_element_name_fastdds_26_accepts():
    text = TMPL.read_text(encoding="utf-8")
    assert "interfaceWhiteList" in text and "interfaceWhitelist" not in text
    assert "@LAN_IP@" in text
    assert "<useBuiltinTransports>false</useBuiltinTransports>" in text
    assert "<type>SHM</type>" in text and 'is_default_profile="true"' in text


def test_override_ip_renders_the_profile_and_docker_args(tmp_path: Path):
    kv, _ = _source({"S2R_LAN_IP": "10.9.8.7"}, tmp_path)
    out = Path(kv["F"])
    assert out.exists() and out.parent == tmp_path, kv
    xml = out.read_text(encoding="utf-8")
    assert "<address>10.9.8.7</address>" in xml and "@LAN_IP@" not in xml
    assert kv["A"] == "4" and kv["IP"] == "10.9.8.7"


def test_localhost_only_drops_the_profile(tmp_path: Path):
    kv, _ = _source({"S2R_LAN_IP": "10.9.8.7", "ROS_LOCALHOST_ONLY": "1",
                     "FASTRTPS_DEFAULT_PROFILES_FILE": "/stale.xml"}, tmp_path)
    assert kv["F"] == "" and kv["A"] == "0"


def test_host_without_lan_ip_warns_and_exports_nothing(tmp_path: Path):
    fake_ip = tmp_path / "ip"
    fake_ip.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_ip.chmod(0o755)
    kv, err = _source({"PATH": f"{tmp_path}:{os.environ['PATH']}"}, tmp_path)
    assert kv["F"] == "" and kv["A"] == "0"
    assert "랜" in err


def test_subnet_in_env_file_matches_the_site_lan_addresses():
    m = re.search(r"^S2R_LAN_SUBNET=(\S+)$", ENV_FILE.read_text(encoding="utf-8"), re.M)
    assert m, "config/ros_domain.env 에 S2R_LAN_SUBNET= 줄이 없다"
    subnet = m.group(1)
    hosts = yaml.safe_load(SITE_V2.read_text(encoding="utf-8"))["hosts"]
    for role in ("control", "vision"):
        assert hosts[role]["ip"].startswith(subnet), (role, hosts[role]["ip"], subnet)


def test_launch_entry_points_source_the_profile():
    for rel in ("scripts/vision/common.sh", "deploy/s2r_console/tools/console.sh"):
        assert "config/dds/lan_profile.sh" in (REPO / rel).read_text(encoding="utf-8"), rel
    for rel in ("scripts/vision/fpp_up.sh", "scripts/vision/fpp_group_up.sh"):
        assert '"${FPP_DDS_ARGS[@]}"' in (REPO / rel).read_text(encoding="utf-8"), rel


LAUNCH_DIRS = ("scripts/vision", "scripts/ops", "deploy/policy_control/tools", "deploy/s2r_console/s2r_console")
LOCALHOST_ONE = re.compile(r"""ROS_LOCALHOST_ONLY["']?\s*[:=]\s*["']?1\b""")


def test_localhost_only_launch_points_drop_the_profile():
    """ROS_LOCALHOST_ONLY=1 로 띄우는 코드는 같은 파일에서 FASTRTPS_DEFAULT_PROFILES_FILE 을 지워야 한다(둘이 같이 있으면 격리가 깨진다)."""
    bad = []
    for top in LAUNCH_DIRS:
        for path in (REPO / top).rglob("*"):
            if path.suffix not in (".sh", ".py") or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                s = line.strip()
                if s.startswith("#") or not LOCALHOST_ONE.search(line):
                    continue
                if "FASTRTPS_DEFAULT_PROFILES_FILE" not in text:
                    bad.append(f"{path.relative_to(REPO)}:{i}: {s[:90]}")
    assert not bad, "\n".join(bad)
