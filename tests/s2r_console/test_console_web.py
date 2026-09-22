"""정적 화면 자산(index.html / console.js / console.css)의 구조 검사.

브라우저 없이 잡을 수 있는 것만 본다. 2026-09-21 에 실제로 난 사고가 기준이다:
노드 표의 `<tr class="stale">` 가 "서버와 끊겼다" 전체 화면 덮개의 `.stale`
(position: fixed; inset: 0) 과 이름이 겹쳐, 노드가 stale 인 동안(= 기동 직후 항상)
화면 전체가 덮였다. API 테스트와 `node --check` 는 전부 통과하고 있었다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "deploy" / "s2r_console" / "s2r_console" / "web"
CSS = re.sub(r"/\*.*?\*/", "", (WEB / "console.css").read_text(encoding="utf-8"), flags=re.S)   # 주석은 셀렉터가 아니다
JS = (WEB / "console.js").read_text(encoding="utf-8")
HTML = (WEB / "index.html").read_text(encoding="utf-8")

RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
BARE_CLASS = re.compile(r"^\.([A-Za-z_][\w-]*)$")


def fixed_classes() -> set[str]:
    """`position: fixed` 를 거는, 클래스 하나짜리 셀렉터의 클래스 이름."""
    out: set[str] = set()
    for selectors, body in RULE.findall(CSS):
        if not re.search(r"position\s*:\s*fixed", body):
            continue
        for sel in selectors.split(","):
            m = BARE_CLASS.match(sel.strip())
            if m:
                out.add(m.group(1))
    return out


def js_template_classes() -> set[str]:
    """console.js 의 템플릿이 `class="..."` 로 내보내는 클래스 이름 (보간 `${}` 은 뺀다)."""
    out: set[str] = set()
    for attr in re.findall(r'class="([^"]*)"', JS):
        attr = re.sub(r"\$\{[^}]*\}", " ", attr)
        out.update(t for t in attr.split() if re.fullmatch(r"[\w-]+", t))
    return out


def test_the_scan_sees_the_overlays():
    # 검사가 공허하지 않다는 확인 — 덮개 셋은 실제로 fixed 다.
    assert {"modal", "drawer", "disconnected"} <= fixed_classes()


def test_js_never_renders_a_fixed_overlay_class_into_the_page_body():
    # fixed 덮개는 index.html 에만 있고, js 는 hidden 을 켜고 끌 뿐이다.
    # js 템플릿이 같은 이름을 쓰면 그 행·칸이 화면 전체를 덮는다.
    clash = fixed_classes() & js_template_classes()
    assert not clash, f"js 템플릿이 전체 화면 덮개의 클래스를 쓴다: {sorted(clash)}"


def test_the_stale_row_is_not_an_overlay():
    assert "stale" in js_template_classes()          # 노드 표의 stale 행은 그대로 있고
    assert "stale" not in fixed_classes()            # 덮개가 아니다
    assert re.search(r"tr\.stale\s+td\s*\{", CSS)    # 흐리게만 한다


@pytest.mark.parametrize("el_id", ["landing", "work", "stopbar", "logdrawer", "modal", "stale", "diagram", "dg-extra"])
def test_everything_toggled_by_hidden_starts_hidden(el_id):
    # 첫 state 가 오기 전에는 아무것도 보이면 안 된다 (빈 화면이 거짓 화면보다 낫다).
    tag = re.search(rf'<[a-z]+[^>]*\bid="{el_id}"[^>]*>', HTML)
    assert tag, f"index.html 에 #{el_id} 가 없다"
    assert re.search(r"\bhidden\b", tag.group(0)), f"#{el_id} 가 hidden 없이 시작한다"


def test_hidden_beats_display_rules():
    # `.modal { display: grid }` 같은 규칙이 UA 의 [hidden] 을 이기지 못하게 하는 한 줄.
    assert re.search(r"\[hidden\]\s*\{\s*display\s*:\s*none\s*!important", CSS)


def test_every_id_the_js_touches_exists_in_the_html():
    wanted = set(re.findall(r'\$\("([\w-]+)"\)', JS))
    have = set(re.findall(r'\bid="([\w-]+)"', HTML))
    rendered = set(re.findall(r'\bid="([\w-]+)"', JS))       # js 가 스스로 그려 넣는 id
    missing = wanted - have - rendered
    assert not missing, f"js 가 찾는 id 가 어디에도 없다: {sorted(missing)}"


def rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert m, f"{selector} 규칙이 없다"
    return m.group(1)


def test_the_full_width_strip_cannot_be_squeezed_to_nothing():
    # body 는 height 100% 라 .work 의 높이가 정해져 있다. overflow:hidden 인 패널은
    # 자동 최소 높이가 0 이라 auto 행에서는 0 으로 눌린다 (09.21 연결 줄이 접혀 세 칸에 가려졌다).
    assert "overflow: hidden" in rule(".panel")
    assert re.search(r"grid-column:\s*1\s*/\s*-1", rule(".links-panel"))
    assert HTML.index('class="panel links-panel"') < HTML.index('class="col ')   # 첫 행이어야 아래 규칙이 맞는다
    assert re.search(r"grid-template-rows:\s*max-content", rule(".work"))


def test_a_link_block_is_never_narrower_than_its_rows():
    # 09.21 스택 블록: 폭을 균등 분배(flex-basis 0)하고 min-width 를 px 로 고정하니, 이름이 긴 토픽 행의
    # 상태·Hz 열이 블록 밖으로 밀려 잘렸다. 블록의 최소 폭은 내용(nowrap 행)이 정해야 하고, 모자라면 줄 전체가 가로로 스크롤한다.
    link = rule(".link")
    assert re.search(r"min-width:\s*min-content", link)
    assert not re.search(r"flex:\s*1\s+1\s+0(?![\d.])", link)
    assert re.search(r"overflow-x:\s*auto", rule(".links"))
    assert re.search(r"min-width:\s*\d+px", rule(".link-head"))   # 행이 없는 블록도 읽을 만한 폭을 갖는다


# ── 연결 그림 ───────────────────────────────────────────────────────────
def test_the_wire_layer_never_eats_a_click_meant_for_a_switch():
    # 전선 SVG 는 상자 전체를 덮는 크기다. 클릭을 받으면 그 아래의 스위치·로그 버튼이 전부 죽는다.
    wires = rule(".dg-wires")
    assert re.search(r"pointer-events:\s*none", wires)
    assert re.search(r"position:\s*absolute", wires) and "fixed" not in wires
    assert re.search(r"position:\s*relative", rule(".dg"))            # 전선 좌표의 기준 — 없으면 페이지 기준으로 그려진다
    assert re.search(r"overflow-x:\s*auto", rule(".dg"))              # 좁은 화면에서는 그림째 가로로 민다


def test_the_switch_is_a_button_that_only_shows_what_the_server_said():
    # 체크박스는 누르는 순간 제 모양을 바꾼다 — 서버가 거절해도 켜진 것처럼 남는다. 스위치는 상태 없는 버튼이어야 한다.
    assert re.search(r'<button[^>]*role="switch"[^>]*data-act="unit', JS) or re.search(r'<button[^>]*data-act="unit[^>]*role="switch"', JS)
    assert 'type="checkbox" data-act="unit' not in JS
    assert re.search(r'"/api/unit"', JS)
    assert "argv" not in re.search(r'call\("POST", "/api/unit",\s*\{([^}]*)\}', JS).group(1)   # 보내는 것은 키와 켬/끔뿐


def test_live_wires_move_and_the_motion_can_be_switched_off():
    assert re.search(r"\.wire\.flow\s*\{[^}]*animation", CSS)
    assert re.search(r"prefers-reduced-motion:\s*reduce", CSS)


def test_the_picture_divides_its_width_by_the_number_of_columns():
    # 열이 8개인 그림(카메라 → FPP → 센서 → obs → policy → fabric → pd → 구동)도 한 화면에 들어와야 한다.
    assert re.search(r"container-type:\s*inline-size", rule(".dg"))
    assert "var(--dg-n)" in rule(".dg-box") and "cqw" in rule(".dg-box")
    assert re.search(r'setProperty\("--dg-n",\s*D\.cols\.length\)', JS)


def test_the_stages_inside_a_node_and_the_outside_list_are_drawn_from_what_the_server_sent():
    assert re.search(r"b\.stages", JS) and ".dg-stage" in CSS
    assert re.search(r"renderExtra\(D\.extra\)", JS)
    body = re.search(r"function renderExtra\(extra\) \{(.*?)\n\}", JS, re.S).group(1)
    assert "esc(t.name)" in body and "esc(t.pubs.join" in body                # 그래프에서 온 이름은 전부 이스케이프한다


def test_the_diagram_headline_is_painted_by_its_own_verdict():
    # 요약이 warn 을 낼 수 있게 됐다 — 색이 붙지 않으면 운영자에게는 안 보이는 것과 같다.
    assert re.search(r"#links-meta \[class\^=\"tone-\"\] \{[^}]*color: var\(--tone\)", CSS)


def test_narrow_box_titles_get_the_whole_line_to_themselves():
    # 8열 실기 그림에서 이름이 "robot_cont / rol" 로 쪼개졌다 — 판정 낱말이 같은 줄을 먹어 77px 만 남았다(실측).
    assert re.search(r"\.dg-tight \.dg-head \{[^}]*flex-wrap: wrap", CSS)
    assert re.search(r"\.dg-tight \.dg-state \{[^}]*flex: 1 0 100%", CSS)
    assert re.search(r"\.dg-tight \.dg-head b \{[^}]*overflow-wrap: break-word", CSS)
    assert "overflow-wrap: anywhere" not in CSS


def test_a_box_that_runs_on_another_machine_says_so():
    # 카메라·FP++ 는 vision-3090 에서 돈다 — 어디로 가서 고쳐야 하는지가 상자에 있어야 한다.
    assert "dg-host" in JS and re.search(r"b\.host", JS)
    assert re.search(r"\.dg-host \{", CSS)


def test_many_columns_fit_by_tightening_both_the_gap_and_the_floor():
    # 인지 런처까지 붙으면 실기 체인은 9~10열이다 — 간격만 줄여서는 1500 px 창을 넘는다(실측).
    assert re.search(r"\.dg-tight \{[^}]*--dg-gap:[^}]*--dg-min:", CSS)
    assert "clamp(var(--dg-min" in CSS
