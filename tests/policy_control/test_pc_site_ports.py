"""사이트별 포트(config/sites/*.yaml) — 10.26 사용자: kitech(기존 배선) · arm(새 배선)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from policy_control import rh56f1_ecat_node as NODE
from policy_control import site_ports as SITE

PC = Path(__file__).resolve().parents[2] / "deploy" / "policy_control"
PORTS = PC / "config" / "rh56f1_ports.yaml"


def test_sites_present():
    assert {"kitech", "arm"} <= set(SITE.available())


def test_kitech_equals_base_file():
    base = yaml.safe_load(PORTS.read_text())
    got = SITE.load_ports(PORTS, "kitech")
    for side in SITE.SIDES:
        assert got[side]["ifname"] == base[side]["ifname"]
        assert got[side]["transport"] == base[side]["transport"]
    assert got["ethercat"] == base["ethercat"]


def test_arm_both_hands():
    got = SITE.load_ports(PORTS, "arm")
    assert got["site"] == "arm"
    assert NODE.ecat_config(got, "right")[0] == "enx00e04c6806e1"
    ifname, cfg = NODE.ecat_config(got, "left")
    assert ifname == "enx00e04c36034e"
    assert cfg == yaml.safe_load(PORTS.read_text())["ethercat"]
    for side in SITE.SIDES:
        SITE.check_side(got, side)


def test_unconnected_side_is_refused():
    site = {"site": "x", "rh56f1": {"right": {"ifname": "eth9"}, "left": {"ifname": None}}}
    got = SITE.apply_site(yaml.safe_load(PORTS.read_text()), site)
    SITE.check_side(got, "right")
    with pytest.raises(SITE.SiteError, match="left 손 NIC 미정"):
        SITE.check_side(got, "left")


def test_apply_does_not_mutate():
    base = yaml.safe_load(PORTS.read_text())
    before = yaml.safe_dump(base)
    SITE.apply_site(base, SITE.load_site("arm"))
    assert yaml.safe_dump(base) == before


def test_resolve_order(tmp_path):
    hf = tmp_path / "site"
    assert SITE.resolve_name(env={}, host_file=hf) is None
    hf.write_text("Kitech\n")
    assert SITE.resolve_name(env={}, host_file=hf) == "kitech"
    assert SITE.resolve_name(env={"S2R_SITE": "arm"}, host_file=hf) == "arm"
    assert SITE.resolve_name("kitech", env={"S2R_SITE": "arm"}, host_file=hf) == "kitech"


def test_unknown_site():
    with pytest.raises(SITE.SiteError, match="있는 것"):
        SITE.load_site("nowhere")


def test_no_site_keeps_base():
    base = yaml.safe_load(PORTS.read_text())
    assert SITE.load_ports(PORTS, None) == base
