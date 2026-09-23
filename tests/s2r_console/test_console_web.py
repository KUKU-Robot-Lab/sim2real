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
    title = re.search(r"\.dg-tight \.dg-head b \{([^}]*)\}", CSS).group(1)
    assert "anywhere" not in title                      # 제목은 낱자로 쪼개지 않는다(노드 이름 표는 따로 — 줄바꿈이 낫다)


def test_a_box_that_runs_on_another_machine_says_so():
    # 카메라·FP++ 는 vision-3090 에서 돈다 — 어디로 가서 고쳐야 하는지가 상자에 있어야 한다.
    assert "dg-host" in JS and re.search(r"b\.host", JS)
    assert re.search(r"\.dg-host \{", CSS)


def test_many_columns_fit_by_tightening_both_the_gap_and_the_floor():
    # 인지 런처까지 붙으면 실기 체인은 9~10열이다 — 간격만 줄여서는 1500 px 창을 넘는다(실측).
    assert re.search(r"\.dg-tight \{[^}]*--dg-gap:[^}]*--dg-min:", CSS)
    assert "clamp(var(--dg-min" in CSS


# ── 접근성 바닥(ui-ux-pro-max 1·2순위: 대비 4.5:1 · 보이는 포커스 · 클릭 대상 · 모션 줄이기) ──
def _token(name):
    return re.search(rf"--{name}\s*:\s*(#[0-9a-fA-F]{{6}})", CSS).group(1)


def _contrast(a, b):
    def lum(h):
        rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, bl = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("fg", ["text", "dim", "faint", "mute", "ok", "warn", "bad", "live"])
def test_every_text_colour_reads_at_4_5_to_1_on_every_surface(fg):
    # 상태 낱말(모름·꺼짐)이 --mute 로, 상자 안 메모가 --faint 로 나온다. 흐리면 상태를 잘못 읽는다.
    for bg in ("bg", "panel", "panel-2"):
        assert _contrast(_token(fg), _token(bg)) >= 4.5, (fg, bg, round(_contrast(_token(fg), _token(bg)), 2))


def test_keyboard_focus_is_drawn_on_every_control():
    rule = re.search(r"([^{}]*:focus-visible[^{]*)\{([^}]*)\}", CSS)
    assert rule and "outline" in rule.group(2)
    for sel in ("button", ".sw", "summary", "a"):
        assert sel in rule.group(1), sel


def test_the_process_switch_has_at_least_a_24px_target():
    # 32×17 px 였다 — WCAG 2.2 최소 대상(24×24)에 못 미친다. 드라이버를 끄는 스위치다.
    hit = re.search(r"\.sw::before\s*\{([^}]*)\}", CSS)
    assert hit and "inset" in hit.group(1)


def test_reduced_motion_stops_every_looping_animation():
    block = re.search(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\}\s*\n", CSS, re.S).group(1)
    for sel in (".wire.flow", ".stage.current.running .dot", ".chip.waiting"):
        assert sel in block, sel


def test_a_lock_reason_wraps_to_two_lines_instead_of_hiding_behind_hover():
    rule = re.search(r"\.dg-tight \.dg-lock, \.dg-tight \.dg-manual \{([^}]*)\}", CSS).group(1)
    assert "line-clamp: 2" in rule and "nowrap" not in rule


# ── "노드를 어떻게 켜나" 를 화면이 답한다 (09.22 실기 첫 세션) ────────────
def test_one_control_panel_under_the_banner_carries_every_stage_action():
    # 09.22 사용자: "메인 status 화면 창에서 승인 및 진행" — 카드를 줄줄이 내려가며 버튼을 찾지 않는다.
    assert re.search(r'<section id="control"[^>]*>\s*<ol id="gb"', HTML), "진행 막대가 조작판 맨 위"
    assert HTML.index('id="control"') < HTML.index('id="work"')
    assert "function renderControl" in JS and "renderControl(" in JS.split("function renderControl")[0]
    panel = re.search(r"function renderControl[\s\S]*?\n}\n", JS).group(0)
    for act in ("approve", "run", "skip", "abort-stage"):
        assert f'data-act="{act}"' in panel, act
    assert "cmdsHtml(cur, mine, can, true)" in panel, "수동 확인 버튼도 조작판에"


def test_restart_is_offered_only_when_the_stage_has_live_units_and_it_says_what_it_kills():
    # 09.23 실기: 다시 실행하면 살아 있는 유닛을 "kept" 로 넘겨 망가진 드라이버가 그대로 남았다.
    panel = re.search(r"function renderControl[\s\S]*?\n}\n", JS).group(0)
    assert re.search(r'if \(liveUnits\(cur\.id\)\.length\) actions \+= `<button[^`]*data-act="restart"', panel)
    act = re.search(r"async restart\(id\) \{[\s\S]*?\n  \},", JS).group(0)
    assert "confirm(" in act and "liveUnits(id)" in act                    # 무엇을 내리는지 보여주고 묻는다
    assert 'restart: true' in act and '"/api/stage/run"' in act


def test_the_full_stage_list_is_folded_and_read_only():
    assert re.search(r'<details id="all-stages"', HTML) and 'id="all-stages" class="panel all-stages" open' not in HTML
    listing = re.search(r"function renderStages[\s\S]*?\n}\n", JS).group(0)
    for act in ('data-act="approve"', 'data-act="run"', 'data-act="skip"', 'data-act="ack"'):
        assert act not in listing, act
    assert "cmdsHtml(r, steps, can, false)" in listing and "stage-group" in listing


def test_each_stage_card_has_an_anchor_to_jump_to():
    assert re.search(r'<li id="stage-\$\{esc\(r\.id\)\}"', JS)


def test_stop_steps_say_what_they_bring_down():
    assert 'stop: "■ 정지"' in JS and "정지 → " in JS


def test_a_lock_line_takes_you_to_the_stage_that_turns_the_node_on():
    assert re.search(r'data-act="goto-stage" data-arg="\$\{esc\(u\.goto\)\}"', JS)
    assert re.search(r'"goto-stage"\s*\(', JS) or re.search(r'async "goto-stage"', JS)


def test_jumping_to_a_stage_does_not_land_under_the_sticky_stop_bar():
    assert re.search(r"\.stage \{[^}]*scroll-margin-bottom", CSS) or re.search(r"\.stage\s*\{[^}]*scroll-margin", CSS)


def test_long_node_names_wrap_instead_of_being_cut():
    assert re.search(r"\.node-name \{[^}]*overflow-wrap: anywhere", CSS)


def test_the_stop_bar_is_not_trapped_inside_a_one_screen_body():
    # body 가 height:100% 로 화면 한 장에 묶여 있었다 — sticky 정지 바가 스크롤하면 화면 중간에 떠서 내용을 덮다가
    # 위로 사라졌다(09.22 실기 첫 세션 사진). 단계를 실행하러 미션 패널로 내려가면 정지 버튼이 없었다.
    assert not re.search(r"html\s*,\s*body\s*\{[^}]*height:\s*100%", CSS)
    body = re.search(r"(?m)^body \{([^}]*)\}", CSS).group(1)
    assert "min-height: 100vh" in body and not re.search(r"(?<!min-)height:\s*100%", body)


def test_the_jump_highlight_survives_a_re_render():
    assert "flashStage" in JS and re.search(r'r\.id === flashStage', JS)


def test_shown_commands_paste_into_a_shell_as_is():
    """09.22 실기: 수동 스텝 `bash -lc "<스크립트>"` 를 공백으로 이어 보여줘서, 붙여넣으면 bash -lc 가
    `sudo` 한 단어만 받아 사용법만 찍었다. 화면의 명령은 셸에 그대로 붙여넣어 같은 일을 해야 한다."""
    import json
    import shutil
    import subprocess

    assert 'argv.join(" ")' not in JS, "명령 표시는 shellLine() 으로만"
    node = shutil.which("node")
    if not node:
        pytest.skip("node 없음")
    helper = re.search(r"const SHELL_SAFE[\s\S]*?const shellLine[^\n]*\n", JS).group(0)
    cases = [
        ["bash", "-lc", "sudo ip link set can0 down && sudo ip link set can0 up"],
        ["ros2", "launch", "p", "x.launch.py", "a:=1"],
        ["python3", "-c", "print('it''s')", "a b", ""],
    ]
    out = subprocess.run([node, "-e", helper + f"for (const a of {json.dumps(cases)}) console.log(shellLine(a));"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == cases[0][2]
    assert out[1] == "ros2 launch p x.launch.py a:=1"
    echoed = subprocess.run(["bash", "-c", f"printf '%s\\n' {out[2]}"], capture_output=True, text=True, check=True).stdout
    assert echoed.split("\n")[:-1] == cases[2]


def test_manual_commands_have_a_copy_button_but_the_console_never_runs_them():
    """수동 명령은 복사만 된다 — 복사 동작은 클립보드에만 쓰고 서버를 부르지 않는다."""
    assert JS.count("copyBtn(shellLine(") >= 3, "단계 카드 목록 · 확인 대기 상자 · 연결 그림의 수동 줄"
    body = re.search(r"async copy\(text\) \{([\s\S]*?)\n  \},", JS).group(1)
    assert "clipboard.writeText" in body and "call(" not in body


def test_diagram_boxes_can_be_moved_and_the_layout_stays_in_this_browser():
    """상자 배치는 localStorage(프로파일별)에만 — 서버로 가지 않고, 저장이 막혀도 화면은 돈다."""
    assert 'data-act="layout-reset"' in HTML and 'id="dg-reset"' in HTML
    assert "`s2r.layout.${profileId}`" in JS
    layout = JS[JS.index("// ── 연결 그림 배치"):JS.index("function unitModal")]
    assert "call(" not in layout, "배치는 서버와 무관하다"
    assert layout.count("localStorage") == 2 and layout.count("try {") >= 2, "읽기·쓰기 모두 try 로 감싼다"
    assert 'addEventListener("keydown"' in layout, "끌기만이 아니라 방향키로도 옮긴다"
    # 오프셋은 HTML 이 아니라 그린 뒤에 입힌다 — 끄는 도중 다시 그려져도 끊기지 않게
    assert "translate" not in re.search(r"function boxHtml[\s\S]*?\n}", JS).group(0)
    assert re.search(r'put\("dg-cols"[^\n]*\n\s*layoutFor\(s\.profile\.id\);\n\s*applyLayout\(\);', JS)
    assert re.search(r"\.dg-head\s*\{[^}]*touch-action:\s*none", CSS)


def test_the_side_column_shows_robot_joints_instead_of_processes_and_events():
    # 09.23 사용자: "프로세스 창은 굳이 없어도 될 것 같고. 사건도. 이미 연결 창에서 로그로 다 보이잖아."
    assert 'id="robot"' in HTML and "로봇 상태" in HTML
    assert 'id="procs"' not in HTML and 'id="events"' not in HTML
    assert "renderProcs" not in JS and "renderEvents" not in JS
    body = re.search(r"function renderRobot[\s\S]*?\n}\n", JS).group(0)
    for field in ("groups", "channels", "stale"):
        assert field in body, field                     # 판정·채널은 서버(robot_view.py)가 준 값으로만
    assert "j-limit" in CSS and "j-off" in CSS           # 끝점·벗어남을 색으로


def test_the_panel_is_left_table_art_right_table_and_uses_the_generated_svgs():
    # 09.23 사용자: "오른팔 상태 <- 정면 오픈암 -> 왼 상태 / 오른손 상태 <오른손·왼손> 왼손 상태"
    body = re.search(r"function renderRobot[\s\S]*?\n}\n", JS).group(0)
    assert re.search(r'jointPanel\(by\("오른팔"\).*art\("arms"\).*jointPanel\(by\("왼팔"\)', body, re.S)
    assert re.search(r'jointPanel\(by\("오른손"\)[\s\S]*art\("right"\)[\s\S]*art\("left"\)[\s\S]*jointPanel\(by\("왼손"\)', body)
    assert re.search(r'grid-template-columns:\s*minmax\(0,1fr\)[^;]*minmax\(0,1fr\);', CSS)
    for stem in ("robot_arms", "robot_hand_right", "robot_hand_left"):
        assert stem in JS, stem
        assert (WEB / f"{stem}.svg").exists() and (WEB / f"{stem}.png").exists(), stem   # 실루엣 + 음영 렌더
    # 09.23 사용자 "렌더에 실루엣과 같이 되면 좋을 것 같음" — PNG 위에 SVG 를 겹친다(같은 투영·같은 창)
    assert re.search(r'<img src="\$\{ROBOT_ART\[key\]\}\.png"', JS)
    assert re.search(r"\.rart svg\s*\{[^}]*position:\s*absolute", CSS)
    assert re.search(r"\.rart svg polygon\s*\{[^}]*fill:\s*none", CSS)               # 상태 없는 링크는 렌더가 보인다
    # 관절 id → 링크 id 로 바꿔 색칠한다(r_hj_index_2 → r_hl_index_2)
    assert re.search(r'replace\(/_\(\[ah\]\)j_/', JS)


def test_every_channel_gets_a_chip_and_position_is_the_default():
    # 09.23 사용자: "디폴트는 joint state 고, vel 이나 effort 들도"
    body = re.search(r"function renderRobot[\s\S]*?\n}\n", JS).group(0)
    assert 'data-act="robot-chan"' in body and 'r.channels.map' in body
    assert re.search(r'S\.robotChan[^;]*\?[^;]*:\s*"pos"', body)
    assert re.search(r'"robot-chan"\(key\)\s*\{\s*S\.robotChan = key;', JS)
