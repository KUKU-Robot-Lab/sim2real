"use strict";
/* S2R 배포 콘솔 — 화면. 규칙 셋:
 *  1. 상태 지식은 서버에 있다. 여기서는 서버가 준 can_run / tone / reasons 를 그릴 뿐, 스스로 판단하지 않는다.
 *  2. ROS 에서 온 문자열은 전부 esc() 를 거친다.
 *  3. 서버와 끊기면 화면을 덮는다 — 오래된 값을 살아 있는 값처럼 보여 주지 않는다.
 */
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// argv 를 셸에 그대로 붙여넣을 수 있는 한 줄로. ["bash","-lc","<스크립트>"] 는 스크립트만 보여준다 —
// 공백으로 이어 붙이면 bash -lc 가 첫 단어(sudo)만 명령으로 받아 사용법만 찍고 끝난다(09.22 실기).
const SHELL_SAFE = /^[A-Za-z0-9_@%+=:,./-]+$/;
const shq = (a) => (a !== "" && SHELL_SAFE.test(a) ? a : `'${String(a).replace(/'/g, `'\\''`)}'`);
const shellLine = (argv) => (argv.length === 3 && /^(ba)?sh$/.test(argv[0]) && /^-l?c$/.test(argv[1]) ? argv[2] : argv.map(shq).join(" "));
// 수동 명령 복사 — 붙여넣을 한 줄(shellLine)을 그대로 클립보드로. 콘솔은 여전히 실행하지 않는다.
// 물리 확인만 하는 스텝(`bash -lc true` — 모터 전원 등)은 붙여넣을 명령이 없다.
const copyBtn = (text) => text === "true" ? "" : `<button class="btn btn-sm btn-ghost btn-copy" data-act="copy" data-arg="${esc(text)}" title="클립보드로 복사 — 다른 셸에 붙여넣어 실행">복사</button>`;
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));

let S = null;                 // 마지막 스냅샷
// 화면에서 고른 것은 S 밖에 둔다 — S 는 폴링마다 서버 스냅샷으로 통째로 바뀐다(09.28: 로봇 상태 채널을 눌러도 바로 '위치'로 돌아갔다)
let robotChan = "pos";
let lastAt = 0;               // 그것을 "끝까지 그린" 시각 [ms] — 받기만 하고 못 그린 것은 세지 않는다
let drawError = null;         // 마지막 그리기 예외. 있으면 화면을 덮는다
const bootAt = Date.now();
const html = {};              // 패널 id → 마지막으로 넣은 HTML (같으면 DOM 을 건드리지 않는다)
const openKeys = new Set();   // 펼쳐 둔 <details> — 다시 그려도 유지한다
let logKey = null;

const me = {
  get name() { return localStorage.getItem("s2r.operator") || ""; },
  set name(v) { localStorage.setItem("s2r.operator", v); },
  get token() { return sessionStorage.getItem("s2r.token") || ""; },
  set token(v) { v ? sessionStorage.setItem("s2r.token", v) : sessionStorage.removeItem("s2r.token"); },
};
const holding = () => !!(S && me.token && S.lease.holder && S.lease.holder === me.name);

function put(id, markup) {
  if (html[id] === markup) return false;
  html[id] = markup;
  $(id).innerHTML = markup;
  return true;
}
const det = (key, summary, body, cls = "more") =>
  `<details class="${cls}" data-key="${esc(key)}"${openKeys.has(key) ? " open" : ""}><summary>${summary}</summary>${body}</details>`;

// ── 서버 ────────────────────────────────────────────────────────────────
async function call(method, path, body) {
  const headers = { "Content-Type": "application/json", "X-S2R-Console": "1" };
  if (me.token) headers["X-S2R-Lease"] = me.token;
  let r, j;
  try {
    r = await fetch(path, { method, headers, body: JSON.stringify(body || {}) });
    j = await r.json();
  } catch (e) {
    toast("콘솔 서버에 닿지 않는다", [String(e)]);
    throw e;
  }
  if (!r.ok || j.ok === false) {
    toast(j.error || `HTTP ${r.status}`, j.reasons);
    throw new Error(j.error);
  }
  return j;
}

function toast(title, reasons, ok = false) {
  const el = document.createElement("div");
  el.className = "toast" + (ok ? " ok" : "");
  const extra = (reasons || []).filter((r) => r !== title);
  el.innerHTML = `<b>${esc(title)}</b>` + (extra.length ? `<ul>${extra.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : "");
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ok ? 2500 : 9000);
}

// ── 그리기 ──────────────────────────────────────────────────────────────
// ── 스크롤 고정 ───────────────────────────────────────────────────────────
// 09.28 창 모드(WebKitGTK)에서 "혼자서 자꾸 스크롤이 올라간다": put() 이 매 폴링 innerHTML 을 바꿔 위쪽 칸 높이가
// 바뀌면, Chrome 은 scroll anchoring 으로 보이는 자리를 지켜 주지만 WebKit 에는 그 기능이 없다. 안쪽 목록(.events ·
// .node-json)은 다시 그려지면 scrollTop 이 0 으로 돌아간다. 그래서 그리기 전후에 직접 맞춘다 — 그리는 동안에는
// 사용자가 스크롤할 수 없으므로(동기 실행) 사용자 스크롤을 덮어쓸 일은 없다.
const INNER_SCROLL = ".events, .node-json";

function keepScroll(draw) {
  const targets = Object.keys(html).map((id) => $(id)).filter(Boolean);
  const anchor = targets.find((el) => el.offsetParent !== null && el.getBoundingClientRect().bottom > 0);
  const top0 = anchor ? anchor.getBoundingClientRect().top : 0;
  const inner = [...document.querySelectorAll(INNER_SCROLL)].map((el) => el.scrollTop);
  draw();
  [...document.querySelectorAll(INNER_SCROLL)].forEach((el, i) => {
    if (inner[i] && el.scrollTop !== inner[i]) el.scrollTop = inner[i];
  });
  if (anchor && anchor.isConnected) {
    const d = anchor.getBoundingClientRect().top - top0;
    if (Math.abs(d) >= 1) window.scrollBy(0, d);
  }
}

function render() {
  keepScroll(renderAll);
}

function renderAll() {
  if (!S) return;
  const s = S.session;
  document.body.classList.toggle("is-real", !!s && s.profile.domain_class === "real");
  renderTop(s);
  renderBanner(s);
  $("landing").hidden = !!s;
  $("work").hidden = !s;
  $("stopbar").hidden = !s;
  $("control").hidden = !s;
  if (!s) return renderLanding();
  renderControl(s);
  renderStages(s);
  renderDiagram(s);
  renderLinks(s);
  renderNodes(s);
  renderMetrics(s);
  renderFpp(s);
  renderPolicy(s);
  renderExperiments(s);
  renderEpisode(s);
  renderTilt(s);
  renderWorkspace();
  renderPolicyTable(s);
  renderRobot(s);
  put("quick", stopButtons(s));
}

// 정지 바 — pd 서비스는 팔마다 따로다(09.23). PD 해제는 **팔마다 한 개**로 나눠 둔다:
// 한 버튼이 양팔을 푸는 것처럼 보이면 안 된다(누른 사람이 반대 팔도 풀렸다고 믿는다).
const SIDE_KO = { right: "오른팔", left: "왼팔" };

function stopButtons(s) {
  const sides = (s && s.mission && (s.mission.lanes || []).map((l) => l.side).filter(Boolean)) || [];
  const arms = sides.length ? sides : ["right", "left"];
  return (S.quick || []).filter((q) => q.where !== "hand").flatMap((q) => {
    if (!q.per_side) return [`<button class="btn-stop" data-act="quick" data-arg="${esc(q.name)}" title="${esc(q.help)}">■ ${esc(q.label)}</button>`];
    return arms.map((a) => `<button class="btn-stop" data-act="quick" data-arg="${esc(q.name)}:${esc(a)}" title="${esc(q.help)}">■ ${esc(q.label)} ${esc(SIDE_KO[a] || a)}</button>`);
  }).join("");
}

function domainBadge(p) {
  const real = p.domain_class === "real";
  return `<span class="badge ${real ? "real" : "fake"}">${real ? "실기" : "FAKE"} · 도메인 ${esc(p.domain)}</span>`;
}

function renderTop(s) {
  put("run-info", s ? `${domainBadge(s.profile)}<span class="run-title">${esc(s.profile.title)}</span><span class="run-id">${esc(s.run_id)}</span>` : "");
  const L = S.lease;
  let m;
  if (holding()) {
    m = `<span class="held">● 조작 중 · ${esc(L.holder)}</span><span class="meta">${fmt(L.expires_in_s, 0)} s</span><button class="btn btn-sm btn-ghost" data-act="lease-drop">놓기</button>`;
  } else if (L.holder) {
    m = `<span class="other">◌ ${esc(L.holder)} 가 조작 중</span><span class="meta">${fmt(L.expires_in_s, 0)} s</span><button class="btn btn-sm btn-ghost" data-act="lease-force">가져오기…</button>`;
  } else {
    m = `<input type="text" id="op-name" placeholder="운영자 이름" value="${esc(me.name)}" maxlength="24"><button class="btn btn-sm btn-primary" data-act="lease-take">조작 권한 잡기</button>`;
  }
  // 입력 중인 이름 칸을 2 Hz 로 갈아엎지 않는다
  if (!(document.activeElement && document.activeElement.id === "op-name")) put("lease", m);
}

// 지금 무엇을 하면 되는가 — 노드는 상자 스위치가 아니라 미션 단계의 ▶ 실행으로 켜진다(09.22 실기 첫 세션에서 못 찾았다)
// ── 조작판: 지금 할 단계 하나 + 묶음 진행 막대 ─────────────────────────────
// 승인 · 실행 · 수동 확인 · 건너뛰기는 전부 여기서 한다. 아래 '전체 단계' 목록은 읽기용이다(09.22 사용자 요청 —
// 카드를 줄줄이 내려가며 찾지 않는다). 무엇을 할 수 있는지는 전부 서버가 정한다(can_run · can_skip …).
let flashStage = null, flashUntil = 0;                                       // 목록의 단계로 간 뒤 잠깐 강조 — 다시 그려도 유지
const KIND_LABEL = { manual: "✋ 수동", background: "⟳ 배경", foreground: "▶ 실행", stop: "■ 정지" };
const cmdLine = (c) => (c.kind === "stop" ? `정지 → ${(c.stop || []).join(", ")}` : shellLine(c.argv));
const stageGroups = (m) => (m.groups && m.groups.length ? m.groups : [{ id: "", title: "미션", motion: false }]);

function groupBar(m) {
  const cur = m.rows.find((r) => r.current);
  return stageGroups(m).map((g, i) => {
    const rows = m.rows.filter((r) => (r.group || "") === g.id);
    const done = rows.filter((r) => r.done).length;
    const here = !!cur && (cur.group || "") === g.id;
    const state = here ? "now" : rows.length && done === rows.length ? "done" : "todo";
    return `<li class="gb-step ${state}${g.motion ? " motion" : ""}"${here ? ' aria-current="step"' : ""}>
      <span class="gb-num">${state === "done" ? "✓" : i + 1}</span><span class="gb-title">${esc(g.title)}</span><span class="gb-count">${done}/${rows.length}</span></li>`;
  }).join("");
}

function groupChips(m, cur) {
  const rows = m.rows.filter((r) => (r.group || "") === (cur.group || ""));
  return rows.map((r) => `<span class="gc${r.current ? " now" : r.done ? " done" : ""}" title="${esc(r.title)}">${r.done ? "✓ " : r.current ? "● " : ""}${esc(r.id)}</span>`).join("");
}

// 한 단계의 명령 목록. live = 조작판(수동 확인 버튼을 단다) · 아니면 목록(읽기용)
function cmdsHtml(r, steps, can, live) {
  return r.commands.map((c, k) => {
    const st = steps ? steps[k] : null;
    const chip = st && st.status !== "pending" ? `<span class="chip ${esc(st.status)}">${esc(stepLabel(st))}</span>` : "";
    const logBtn = st && !["manual", "stop"].includes(st.kind) && st.status !== "pending" && st.status !== "kept" ? `<button class="btn btn-sm btn-ghost" data-act="log" data-arg="${esc(st.key)}">로그</button>` : "";
    const waiting = live && st && st.status === "waiting";
    const manual = waiting ? `<div class="manual-box"><b>다른 셸에서 직접 실행할 것</b> — 콘솔은 이 명령을 실행하지 않는다.
        <div class="cmd-copy"><pre>${esc(shellLine(c.argv))}</pre>${copyBtn(shellLine(c.argv))}</div>
        <div class="actions"><button class="btn btn-primary btn-sm" data-act="ack" data-arg="${esc(r.id)}:${k}:1" ${can ? "" : "disabled"}>실행했고 정상이다 → 다음</button>
        <button class="btn btn-sm" data-act="ack" data-arg="${esc(r.id)}:${k}:0" ${can ? "" : "disabled"}>정상이 아니다 → 중단</button></div></div>` : "";
    const detail = st && st.detail ? `<span class="cmd-argv" style="color:var(--warn)">${esc(st.detail)}</span>` : "";
    const copy = c.kind === "manual" && !manual ? copyBtn(shellLine(c.argv)) : "";
    return `<li class="cmd${waiting ? " waiting" : ""}"><span class="cmd-kind ${esc(c.kind)}">${KIND_LABEL[c.kind] || esc(c.kind)}</span><span>${esc(c.note || cmdLine(c))}${detail}<span class="cmd-argv">${esc(cmdLine(c))}</span></span><span>${chip} ${logBtn}${copy}</span>${manual}</li>`;
  }).join("");
}

// 직전에 끝낸 단계로 한 칸 돌아가기 — 더 앞은 '전체 단계' 목록의 ↶ 로
function prevBtn(m, can) {
  const cur = m.rows.findIndex((r) => r.current);
  const prev = [...m.rows.slice(0, cur < 0 ? m.rows.length : cur)].reverse().find((r) => r.can_rewind);
  return prev ? `<button class="btn btn-sm btn-ghost" data-act="rewind" data-arg="${esc(prev.id)}" ${can ? "" : "disabled"}>↶ 이전 단계(${esc(prev.id)})로</button>` : "";
}

// 한 단계 카드의 속. `R` = 그 창의 러너(없으면 null). 창이 여럿이면 카드도 여럿이다.
function stageCard(s, cur, R, can) {
  const m = s.mission;
  const g = stageGroups(m).find((x) => x.id === (cur.group || "")) || stageGroups(m)[0];
  const running = cur.status === "RUNNING";
  const failed = !running && (cur.last_outcome === "FAILED" || cur.last_outcome === "ABORTED");
  const mine = R && R.stage === cur.id ? R.steps : null;
  const waiting = mine && mine.some((st) => st.status === "waiting");
  const tags = [
    cur.touches_real ? `<span class="badge real">실기</span>` : "",
    g.motion ? `<span class="badge bad">팔·목이 움직인다</span>` : `<span class="badge">움직임 없음</span>`,
    cur.touches_real && cur.approved ? `<span class="badge ok">승인됨</span>` : "",
    failed ? `<span class="badge bad">${esc(cur.last_outcome)}</span>` : "",
  ].join("");
  let say;
  if (!can) say = `<b>먼저 조작 권한을 잡을 것</b> — 오른쪽 위에 이름을 넣고 <b>조작 권한 잡기</b>.`;
  else if (waiting) say = `<b>✋ 확인을 기다린다</b> — 아래 명령을 다른 셸에서 실행하거나 확인한 뒤 <b>실행했고 정상이다</b>.`;
  else if (running) say = `실행 중… 명령이 끝나는 대로 다음 단계로 넘어간다.`;
  else if (failed) say = `<span class="warn">${esc(cur.last_note || "실패했다")}</span> — 원인을 고친 뒤 다시 실행하거나, 건너뛸 수 있으면 건너뛴다.`;
  else if (cur.reasons.length) say = `<span class="warn">막힘: ${esc(cur.reasons[0])}</span>`;
  else say = cur.touches_real && !cur.approved ? `실기 단계다 — <b>승인…</b> 뒤 <b>▶ 실행</b>.` : `<b>▶ 실행</b>으로 시작한다.`;
  let actions = "";
  if (running) actions = `<button class="btn btn-sm" data-act="abort-stage" data-arg="${esc(cur.id)}" ${can ? "" : "disabled"}>■ 이 단계 중단</button>`;
  else {
    if (cur.touches_real && !cur.approved) actions += `<button class="btn btn-real" data-act="approve" data-arg="${esc(cur.id)}" ${can && cur.can_approve ? "" : "disabled"}>승인…</button>`;
    actions += `<button class="btn ${cur.touches_real ? "btn-real" : "btn-primary"}" data-act="run" data-arg="${esc(cur.id)}" ${can && cur.can_run ? "" : "disabled"}>▶ ${failed ? "다시 실행" : "실행"}</button>`;
    // 09.23 실기: 살아 있는 유닛은 "kept" 로 건너뛴다 — 망가진 드라이버를 갈아 끼우려면 먼저 내려야 한다.
    if (liveUnits(cur.id).length) actions += `<button class="btn btn-ghost" data-act="restart" data-arg="${esc(cur.id)}" ${can && cur.can_run ? "" : "disabled"} title="이 단계가 띄운 ${liveUnits(cur.id).length} 개를 내리고 새로 띄운다">↻ 다시 띄우기</button>`;
    if (cur.skippable) {
      const why = (cur.skip_why || []).join(" · ");
      actions += `<button class="btn btn-ghost" data-act="skip" data-arg="${esc(cur.id)}" ${can && cur.can_skip ? "" : "disabled"} title="${esc(why || "실행하지 않고 다음 단계로")}">이 단계 건너뛰기</button>`;
    }
  }
  const reasons = cur.reasons.length > 1 ? `<ul class="reasons">${cur.reasons.slice(1).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  const stale = cur.approval_stale.length ? `<ul class="reasons bad"><li><b>이전 승인이 무효가 됐다</b> — 승인한 뒤 파일이 바뀌었다</li>${cur.approval_stale.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  // 창을 나눠 놓으면 "다른 창의 것을 내린다"를 반드시 말해야 한다 — pd 노드 이름이 하나뿐이라 왼팔을 띄우면 오른팔이 풀린다.
  const steals = (cur.stops_lanes || []).length
    ? `<ul class="reasons bad"><li><b>이 단계는 ${cur.stops_lanes.map((id) => esc(laneTitle(m, id))).join(" · ")} 창의 프로세스를 내린다</b> — 그 팔은 pd 가 풀리고 JTC 가 잡는다</li></ul>` : "";
  const cmds = cur.commands.length ? (mine || waiting || cur.commands.length <= 3
    ? `<ul class="cmds">${cmdsHtml(cur, mine, can, true)}</ul>`
    : det(`cp:${cur.id}`, `실행할 명령 ${cur.commands.length}개`, `<ul class="cmds">${cmdsHtml(cur, mine, can, true)}</ul>`)) : "";
  return `<div class="cp-head"><span class="cp-where">${esc(g.title)} ›</span><span class="cp-title">${esc(cur.id)}</span>${tags}</div>
    <div class="cp-desc">${esc(cur.title)}</div>
    <p class="cp-note">${say}</p>${reasons}${stale}${steals}${cmds}
    <div class="actions cp-actions">${actions}</div>`;
}

const laneTitle = (m, id) => ((m.lanes || []).find((l) => l.id === id) || { title: id }).title;

// 창 하나 = 따로 도는 장치. 그 창에서 아직 안 끝낸 첫 단계를 카드로, 나머지는 칩으로.
function laneHtml(s, lane, can) {
  const m = s.mission, rows = m.rows.filter((r) => r.lane === lane.id);
  const optional = new Set(stageGroups(m).filter((g) => g.optional).map((g) => g.id));
  const done = rows.filter((r) => r.done).length;
  const chips = rows.map((r) => `<span class="gc${r.id === lane.next ? " now" : r.done ? " done" : ""}" title="${esc(r.title)}">${r.done ? "✓ " : r.id === lane.next ? "● " : ""}${esc(r.id)}</span>`).join("");
  let body;
  // 카드는 **돌고 있는 단계**가 먼저다 — 09.28 실기: '다른 단계 고르기'로 policy_left 를 띄웠더니 카드는 차례인
  // selftest_left 에 머물러, policy_left 의 수동 확인(컵 · 주변) 버튼이 화면 어디에도 없었다.
  const shown = lane.busy || lane.next;
  if (!shown) body = `<p class="cp-note">이 창의 단계를 모두 끝냈다.</p>`;
  else {
    const cur = rows.find((r) => r.id === shown);
    const last = lane.last && lane.last.stage === cur.id ? lane.last : null;
    body = stageCard(s, { ...cur, last_outcome: last ? last.outcome : "", last_note: last ? last.note : "" },
                     s.runners ? s.runners[lane.id] : null, can);
  }
  return `<section class="panel lane${lane.busy ? " lane-busy" : ""}">
    <div class="panel-head"><h2>${esc(lane.title)}</h2><span class="meta">${done}/${rows.length}${lane.busy ? ` · ${esc(lane.busy)} 실행 중` : ""}</span></div>
    <div class="cp-chips">${chips}</div>${body}${exitHtml(rows, lane, can, optional)}</section>`;
}

// 정리 단계(비상 복귀 · 차렷 복귀 · pd 해제)는 순서를 기다리지 않고 언제나 고를 수 있다 — 09.28 실기: selftest 가
// 실패하면 pd 가 팔을 잡은 동안 건너뛰기가 막혀(안전 규칙) 창이 거기 멈췄고, 차렷으로 돌아올 길이 화면에 없었다.
// 서버는 창 안의 어느 단계든 needs · 승인 · 창이 비었는지로만 판정한다(_current_stage) — 버튼만 없던 것이다.
// 09.28 실기 2: 같은 이유로 selftest_left 실패 뒤 policy_left 로 갈 길도 없었다 — 나머지 단계도 따로 묶어 보인다.
function exitHtml(rows, lane, can, optional = new Set()) {
  // 정리 · 정책 단계는 끝난 뒤에도 다시 고른다 — 09.28 사용자: "정책 → 홈 자세 → 정책" 을 반복한다.
  // 선택 묶음(진단: selftest · FP++ 재등록)도 — 리셋이 FP++ 때문에 거부될 때마다 다시 돌린다.
  const again = (r) => r.group === "finish" || r.group === "policy" || optional.has(r.group);
  const left = rows.filter((r) => (!r.done || again(r)) && r.id !== lane.next && r.id !== lane.busy);
  const outs = left.filter((r) => r.group === "finish");
  const jumps = left.filter((r) => r.group !== "finish");
  const btn = (r) => {
    const need = r.touches_real && !r.approved;
    const act = need ? `<button class="btn btn-sm btn-real" data-act="approve" data-arg="${esc(r.id)}" ${can && r.can_approve ? "" : "disabled"}>승인…</button>` : "";
    const why = r.reasons && r.reasons.length ? `<span class="hint" title="${esc(r.reasons.join(" · "))}"> · 막힘: ${esc(r.reasons[0])}</span>` : "";
    return `<div class="exit-row"><span class="exit-title" title="${esc(r.title)}"><b>${r.done ? "✓ " : ""}${esc(r.id)}</b> — ${esc(r.title)}${why}</span>
      <span class="actions">${act}<button class="btn btn-sm ${r.touches_real ? "btn-real" : "btn-primary"}" data-act="run" data-arg="${esc(r.id)}" ${can && r.can_run ? "" : "disabled"}>▶ 실행</button></span></div>`;
  };
  const exits = outs.length ? det(`exit:${lane.id}`, `정리 · 빠져나가기 (${outs.length}) — 순서와 상관없이`, `<div class="exits">${outs.map(btn).join("")}</div>`) : "";
  const skips = jumps.length ? det(`jump:${lane.id}`, `다른 단계 고르기 (${jumps.length}) — 차례를 건너뛰어 실행 (선행 단계 · 승인은 그대로 본다)`, `<div class="exits">${jumps.map(btn).join("")}</div>`) : "";
  return exits + skips;
}

// 손 창 — 단계가 아니라 **서비스 버튼**이다(09.23 사용자). pd 서비스에는 쪽이 없어서 떠 있는 pd 가 대상을 정한다.
function handPanel(s, side, can) {
  const label = side === "right" ? "오른손" : "왼손";
  const sides = s.pd_sides || [];
  const mine = sides.includes(side);
  const why = mine ? "" : `${label} 의 pd status 가 오지 않는다 — 그 팔의 pd 단계를 먼저 실행할 것`;
  // pd 서비스는 팔마다 따로다(09.23) — 버튼이 쪽을 함께 보낸다.
  const btns = (S.quick || []).filter((q) => q.where === "hand").map((q) =>
    `<button class="btn btn-sm" data-act="quick" data-arg="${esc(q.name)}:${esc(side)}" ${mine && can ? "" : "disabled"} title="${esc(q.help)}">${esc(q.label)}</button>`).join("");
  return `<section class="panel lane hand${mine ? "" : " lane-off"}">
    <div class="panel-head"><h2>${esc(label)}</h2><span class="meta">${mine ? `pd_${esc(side)} 살아 있음` : "대기"}</span></div>
    ${why ? `<p class="cp-note warn">${esc(why)}</p>` : `<p class="cp-note">손만 움직인다 — 팔은 그대로다. 손가락을 펴는 것은 팔이 홈에 정착한 뒤에 한다.
      주먹(경로 자세)으로 오므리는 것은 팔 창의 홈 단계가 한다 — 그 이동은 충돌 검사를 먼저 해야 한다.</p>`}
    <div class="actions">${btns}</div></section>`;
}

function renderControl(s) {
  const can = holding(), m = s.mission;
  put("gb", groupBar(m));
  const lanes = m.lanes || [];
  if (!lanes.length) {                       // 창을 선언하지 않은 미션 — 예전처럼 카드 하나
    const cur = m.rows.find((r) => r.current);
    put("lanes", "");
    put("hands", "");
    if (!cur) {
      return put("cp", `<div class="cp-head"><span class="cp-title">모든 단계가 끝났다</span></div>
        <p class="cp-note">정리가 끝났으면 오른쪽 아래 <b>run 끝내기</b>로 닫는다.</p>`);
    }
    const R = s.runners ? s.runners[""] : null;
    return put("cp", stageCard(s, { ...cur, last_outcome: m.status, last_note: m.note }, R, can)
      + `<div class="actions cp-actions"><span class="spacer"></span>${prevBtn(m, can)}<button class="btn btn-sm btn-ghost" data-act="show-all">전체 단계 ▾</button></div>`);
  }
  put("cp", "");
  put("lanes", lanes.map((l) => laneHtml(s, l, can)).join(""));
  put("hands", ["right", "left"].map((side) => handPanel(s, side, can)).join(""));
}

function renderBanner(s) {
  const el = $("banner");
  if (!s) {
    el.className = "banner tone-mute";
    return put("banner", `<div class="banner-state">대기</div><div class="banner-reasons"><span class="quiet">열린 run 이 없다 — 아래에서 로봇 · 정책을 고를 것.</span></div>`);
  }
  const b = s.banner;
  el.className = `banner tone-${b.tone}`;
  const reasons = b.reasons.length ? `<ul>${b.reasons.slice(0, 4).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>` : `<span class="quiet">이상 없음</span>`;
  const real = s.profile.domain_class === "real";
  const armed = b.armed_for_real;
  const pArmed = armed === null ? `<div class="pill off"><small>명령 발행</small><b>pd 없음</b></div>`
    : armed ? `<div class="pill ${real ? "hot" : "warn"}"><small>명령 발행</small><b>${real ? "실기로 발행 중" : "발행 중 (fake)"}</b></div>`
      : `<div class="pill off"><small>명령 발행</small><b>무발행</b></div>`;
  const br = s.bridge;
  const pBridge = !br.enabled ? `<div class="pill off"><small>브리지</small><b>꺼짐</b></div>`
    : br.up ? `<div class="pill good"><small>브리지 · 구독 전용</small><b>도메인 ${esc(br.domain)}</b></div>`
      : `<div class="pill warn"><small>브리지</small><b>끊김</b></div>`;
  // 에피소드 실행기가 떠 있으면 그 상태를, 아니면 정책 노드의 episode 이벤트를 보인다
  const ers = s.episode_runners || [];          // ★10.19 팔마다 실행기 — 떠 있는 것 전부
  const ep = ers.length ? ers.map((er) => `${esc(er.episode)} ${esc(er.phase)}`).join(" · ")
    : s.episode ? `#${esc(s.episode.episode)} ${esc(s.episode.event)}` : "—";
  const m = s.mission;
  put("banner", `<div class="banner-state">${esc(b.state)}</div><div class="banner-reasons">${reasons}</div>
    <div class="pills">${pArmed}${pBridge}
      <div class="pill"><small>에피소드</small><b>${ep}</b></div>
      <div class="pill"><small>미션 · 사이클 ${esc(m.cycle)}</small><b>${esc(m.stage)} ${esc(m.status)}</b></div></div>`);
}

// ── 첫 화면: ① 로봇 → ② 정책 → ③ 실기 · fake (09.29 사용자: "처음 창을 키면 robot 및 실행 가능한 policy 선택창") ──
// 로봇은 deploy/s2r_console/robots/<id>.yaml 하나씩(모듈). 정책 목록 · 고를 수 있는지는 서버가 판정한다 — 여기는 그리기만.
const pick = { robot: "", policies: {} };
const SLOT_KO = { right: "오른팔", left: "왼팔", both: "양팔" };

function robotCard(r, can) {
  const usable = r.policies.filter((p) => !p.why.length).length;
  const on = pick.robot === r.id;
  return `<div class="card robot-card${on ? " on" : ""}"><div><span class="badge">${esc(r.host || "?")}</span> <span class="badge">${esc(r.hand.model || "")}</span></div>
    <h3>${esc(r.title)}</h3>
    <div class="path">자산 ${esc(r.asset)}<br>손 ${esc(r.hand.vendor || "")} ${esc(r.hand.model || "")} · 구동 ${esc(r.hand.actuated || "?")}<br>
      프로파일 ${r.profiles.length ? r.profiles.map((p) => esc(p.id)).join(" · ") : "없음"} · 고를 수 있는 정책 ${usable}/${r.policies.length}</div>
    ${r.note ? `<div class="hint">${esc(r.note)}</div>` : ""}
    <div class="actions"><button class="btn ${on ? "btn-ghost" : "btn-primary"}" data-act="pick-robot" data-arg="${esc(r.id)}">${on ? "선택됨" : "이 로봇"}</button></div></div>`;
}

function slotPicker(r, side) {
  const def = (r.defaults || {})[side] || "";
  const mine = r.policies.filter((p) => p.side === side);
  const cur = pick.policies[side] || "";
  const opt = (p) => `<option value="${esc(p.id)}" ${p.why.length ? "disabled" : ""} ${cur === p.id ? "selected" : ""}>${esc(p.id)} · ${esc(p.status)}${p.why.length ? ` — ${esc(p.why[0])}` : ""}</option>`;
  const chosen = mine.find((p) => p.id === (cur || def));
  return `<div class="slot"><label><b>${esc(SLOT_KO[side] || side)}</b> <span class="hint">미션 산출물 ${esc(r.slots[side])}</span></label>
    <select data-pick="${esc(side)}"><option value="" ${cur ? "" : "selected"}>미션 기본값${def ? ` (${esc(def)})` : ""}</option>${mine.map(opt).join("")}</select>
    ${chosen ? `<div class="hint slot-note">${esc(chosen.task)}${chosen.note ? ` — ${esc(chosen.note.slice(0, 220))}` : ""}</div>` : ""}</div>`;
}

function policyTable(r) {
  if (!r.policies.length) return `<div class="empty">이 로봇의 정책이 deploy/policies 에 아직 없다.</div>`;
  const rows = r.policies.map((p) => `<tr class="${p.why.length ? "off" : ""}"><td><code>${esc(p.id)}</code></td><td>${esc(SLOT_KO[p.side] || p.side)}</td><td>${esc(p.status)}</td>
    <td>${esc(p.contract || "—")}</td><td>${p.why.length ? `<span class="warn">${esc(p.why.join(" · "))}</span>` : "고를 수 있다"}</td></tr>`).join("");
  return `<table class="ptable"><thead><tr><th>정책</th><th>자리</th><th>상태</th><th>계약</th><th>판정</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderLanding() {
  const can = holding();
  const robots = S.robots || [];
  if (!robots.some((r) => r.id === pick.robot)) pick.robot = robots.length === 1 ? robots[0].id : "";
  const r = robots.find((x) => x.id === pick.robot);
  const step = r ? 2 : 1;
  const steps = ["로봇", "정책", "실기 · fake"].map((t, i) => `<li class="${i + 1 < step ? "done" : i + 1 === step ? "now" : ""}">${i + 1}. ${t}</li>`).join("");
  let body = "";
  if (r) {
    const settings = r.settings.map((x) => `<tr><th>${esc(x.label)}</th><td>${esc(x.value)}</td></tr>`).join("");
    const picked = Object.fromEntries(Object.entries(pick.policies).filter(([, v]) => v));
    const runBtns = r.profiles.map((p) => {
      const real = p.domain_class === "real";
      return `<button class="btn ${real ? "btn-real" : "btn-primary"}" data-act="open-picked" data-arg="${esc(p.id)}" ${can ? "" : "disabled"}>${real ? "실기" : "FAKE"} · 도메인 ${esc(p.domain)} — ${esc(p.title)} run 시작</button>`;
    }).join("");
    body = `<section class="panel pick-panel"><div class="panel-head"><h2>${esc(r.title)}</h2><span class="meta">${esc(r.host)}</span></div>
      <div class="pick-grid">
        <div><h3 class="sub">로봇 설정</h3><table class="settings">${settings}</table></div>
        <div><h3 class="sub">② 정책 고르기</h3>${Object.keys(r.slots).map((side) => slotPicker(r, side)).join("")}
          <p class="hint">고른 정책은 이번 run 에서만 미션 산출물을 바꾼다(yaml 은 그대로). 실기 단계 승인은 그 계약으로 다시 받는다.</p>
          <h3 class="sub">③ 실행</h3><div class="actions run-btns">${runBtns || `<span class="warn">프로파일이 없다 — ${esc((r.missing_profiles || []).join(", ") || "robots/*.yaml 의 profiles")}</span>`}</div>
          ${Object.keys(picked).length ? `<p class="hint">고른 것: ${Object.entries(picked).map(([k, v]) => `${esc(SLOT_KO[k] || k)} ${esc(v)}`).join(" · ")}</p>` : ""}
          ${can ? "" : `<p class="hint">먼저 오른쪽 위에서 조작 권한을 잡을 것</p>`}</div>
      </div>
      ${det(`policies:${r.id}`, `이 로봇의 정책 전부 (${r.policies.length}) — 못 고르는 이유 포함`, policyTable(r))}</section>`;
  }
  const inRobot = new Set(robots.flatMap((x) => x.profiles.map((p) => p.id)));
  const others = S.profiles.filter((p) => !inRobot.has(p.id)).map((p) => {
    const real = p.domain_class === "real";
    return `<div class="card ${real ? "real" : "fake"}"><div>${domainBadge(p)}</div><h3>${esc(p.title)}</h3>
      <div class="path">미션 ${esc(rel(p.mission))}<br>노드 ${p.status_nodes.map(esc).join(" · ")}${p.policy_dir ? `<br>정책 ${esc(rel(p.policy_dir))}` : ""}</div>
      <div class="actions"><button class="btn ${real ? "btn-real" : "btn-primary"}" data-act="open" data-arg="${esc(p.id)}" ${can ? "" : "disabled"}>run 시작</button></div></div>`;
  });
  const broken = [...Object.entries(S.bad_profiles), ...Object.entries(S.bad_robots || {})].map(([name, why]) =>
    `<div class="card broken"><div><span class="badge warn">읽지 못함</span></div><h3>${esc(name)}</h3><div class="path">${esc(why)}</div></div>`);
  put("landing", `<h1>로봇 · 정책 고르기</h1><ol class="steps">${steps}</ol>
    <p class="lead">로봇 하나를 고르면 그 로봇의 설정과 고를 수 있는 정책이 보인다. run 을 열면 그 도메인에 <b>구독 전용</b> 브리지가 붙는다 — 그것만으로는 아무 명령도 나가지 않는다.</p>
    <div class="cards">${robots.map((x) => robotCard(x, can)).join("")}</div>${body}
    ${others.length || broken.length ? det("other-profiles", `로봇 모듈 밖의 프로파일 (${others.length})`, `<div class="cards">${others.join("")}${broken.join("")}</div>`) : ""}`);
}
const rel = (p) => String(p).replace(/^.*\/sim2real\//, "");

const runnerOf = (s, id) => Object.values(s.runners || {}).find((r) => r && r.stage === id) || null;

function renderStages(s) {
  const m = s.mission, can = holding();
  put("mission-meta", `${esc(m.name)} · ${m.rows.filter((r) => r.done).length}/${m.rows.length}`);
  let lastGroup = null;
  const rows = m.rows.map((r, i) => {
    const g = stageGroups(m).find((x) => x.id === (r.group || ""));
    const header = g && g.id !== lastGroup ? `<li class="stage-group${g.motion ? " motion" : ""}">${esc(g.title)}</li>` : "";
    lastGroup = g ? g.id : lastGroup;
    const laneLast = ((m.lanes || []).find((l) => l.id === r.lane) || {}).last;
    const lastHere = laneLast && laneLast.stage === r.id ? laneLast.outcome : (r.current ? m.status : "");
    const running = r.status === "RUNNING";
    const failed = !running && (lastHere === "FAILED" || lastHere === "ABORTED");
    const flash = r.id === flashStage && Date.now() < flashUntil ? "flash" : "";
    const cls = ["stage", r.done ? "done" : "", r.current ? "current" : "", running ? "running" : "", failed ? "failed" : "", r.reasons.length ? "blocked" : "", flash].join(" ");
    const badges = [
      r.touches_real ? `<span class="badge real">실기</span>` : "",
      r.current ? `<button class="badge now" data-act="goto-stage" data-arg="${esc(r.id)}">지금 — 조작판 ↑</button>` : "",
      r.skipped ? `<span class="badge">건너뜀</span>` : "",
      r.can_rewind ? `<button class="btn btn-sm btn-ghost rewind" data-act="rewind" data-arg="${esc(r.id)}" ${can ? "" : "disabled"} title="이 단계부터 다시 진행한다">↶ 여기서 다시</button>` : "",
    ].join("");
    const rr = runnerOf(s, r.id);
    const steps = rr ? rr.steps : null;
    const body = r.commands.length ? det(`cmds:${r.id}`, `명령 ${r.commands.length}개`, `<ul class="cmds">${cmdsHtml(r, steps, can, false)}</ul>`) : "";
    return `${header}<li id="stage-${esc(r.id)}" class="${cls}"><div class="stage-rail"><span class="dot">${r.done ? "✓" : ""}</span></div><div class="stage-body">
      <div class="stage-line"><span class="stage-id">${i + 1}. ${esc(r.id)}</span>${badges}</div><div class="stage-title">${esc(r.title)}</div>${body}</div></li>`;
  });
  const loop = m.loop_to ? `<li class="stage"><div class="stage-rail"></div><div class="stage-body"><span class="hint">↺ 마지막 단계 뒤에는 <b>${esc(m.loop_to)}</b> 로 돌아간다</span></div></li>` : "";
  put("stages", rows.join("") + loop);
}
const stepLabel = (st) => ({ running: "실행 중", waiting: "확인 대기", up: "떠 있음", kept: "이미 떠 있음", done: "완료", failed: `실패${st.rc !== null ? " rc=" + st.rc : ""}`, aborted: "중단됨" }[st.status] || st.status);

// 연결 그림: 상자 = 노드, 전선 = 토픽. 상태·색·사유·스위치를 켤 수 있는지는 전부 서버(diagram.py · units.py)가 정한다.
let lastDiagram = null, lastWires = "";
const TIGHT_FROM_COLS = 6;

// 상자 하나 = 노드 하나. 같은 노드를 여러 단계가 띄우면(pd 무발행 → 발행, 단독 정책 → 에피소드) 명령마다 스위치 줄이 하나다.
function boxUnits(b) {
  return b.units && b.units.length ? b.units : b.unit ? [b.unit] : [];
}

function unitHtml(b) {
  const us = boxUnits(b);
  if (!us.length) return "";
  // 잠긴 사유는 떠 있는 명령(없으면 첫 명령)만 크게 쓴다 — 나머지는 스위치 위에 올리면 보인다
  const focus = us.find((u) => u.alive) || us[0];
  const alt = us.length > 1 && b.single ? `<div class="dg-alt" title="같은 노드라 한 번에 하나만 뜬다">띄우는 단계 ${us.length}개 · 한 번에 하나</div>` : "";
  return alt + us.map((u, i) => unitRow(b, u, u === focus, i === 0)).join("");
}

function unitRow(b, u, withLock, first) {
  if (u.error) return `<div class="dg-why">⚠ ${esc(u.key)} — ${esc(u.error)}</div>`;
  const can = u.alive ? u.can_off : u.can_on, why = (u.alive ? u.why_off : u.why_on)[0] || "";
  const locked = !holding() ? "조작 권한이 없다" : can ? "" : why;
  const meta = u.alive ? `pid ${esc(u.pid)} · ${fmt(u.age_s, 0)} s` : u.started ? `rc ${esc(u.rc)}` : "—";
  const share = first && b.shares.length ? `<span class="dg-share" title="같이 켜지고 꺼진다: ${esc(b.shares.join(" · "))}">⛓ ${b.shares.length + 1}개 묶음</span>` : "";
  const manual = u.kind === "manual" ? `<div class="dg-why dg-manual" title="${esc(shellLine(u.argv))}">${copyBtn(shellLine(u.argv))} 운영자 셸에서 직접: <code>${esc(shellLine(u.argv))}</code></div>` : "";
  return `<div class="dg-unit"><button class="sw ${u.alive ? "on" : "off"}" role="switch" data-act="unit" data-arg="${esc(u.key)}:${u.alive ? "0" : "1"}"
      aria-checked="${u.alive}" aria-label="${esc(b.title)} — ${esc(u.key)} 켜기/끄기" title="${esc(locked || (u.alive ? "끄기" : "켜기"))}"${locked ? " disabled" : ""}></button>
    <span>${esc(u.key)} · ${meta}</span>${share}${u.started ? `<button class="btn btn-sm btn-ghost" data-act="log" data-arg="${esc(u.key)}">로그</button>` : ""}</div>
    ${withLock && !can && why ? (u.goto
        ? `<button class="dg-why dg-lock dg-goto" data-act="goto-stage" data-arg="${esc(u.goto)}" title="${esc(why)} — 누르면 그 단계로 간다">🔒 ${esc(why)} ↓</button>`
        : `<div class="dg-why dg-lock" title="${esc(why)}">🔒 ${esc(why)}</div>`) : ""}${manual}`;
}

function boxHtml(b) {
  const ports = b.ports.map((p) => `<div class="dg-port tone-${esc(p.tone)}" id="dg-port-${esc(p.id)}" title="${esc(p.topic)}"><span class="lamp ${esc(p.tone)}"></span>
      <span class="dg-topic">${esc(p.label)}</span><span class="dg-val">${esc(p.text || LINK_WORD[p.state] || p.state)}</span></div>
      ${p.note && p.tone !== "ok" ? `<div class="dg-note">${esc(p.note)}</div>` : ""}`).join("");
  const lines = b.lines.map((l) => `<div class="dg-line tone-${esc(l.tone)}">${esc(l.text)}</div>`).join("");
  // 한 프로세스 안의 단계(obs → policy → fabric IK …) — 노드가 하나여도 안에서 무엇이 도는지 보인다
  const stages = (b.stages || []).length ? `<div class="dg-stages">${b.stages.map((s) => `<span class="dg-stage">${esc(s)}</span>`).join('<span class="dg-stage-arrow">›</span>')}</div>` : "";
  // 다른 PC 에서 도는 것(인지: vision-3090)은 그렇다고 적는다 — 어디로 가서 고칠지가 달라진다
  const host = b.host ? `<span class="dg-host" title="이 프로세스는 ${esc(b.host)} 에서 돈다">${esc(b.host)}</span>` : "";
  return `<div class="dg-box tone-${esc(b.tone)}" id="dg-box-${esc(b.id)}" data-box="${esc(b.id)}"><div class="dg-head" tabindex="0" title="끌어서 옮긴다 · 방향키 8 px · Shift 32 px"><span class="lamp ${esc(b.tone)}"></span><b>${esc(b.title)}</b>
      <span class="dg-state">${esc(LINK_WORD[b.state] || b.state)}</span></div>${host}
    ${b.detail ? `<div class="dg-detail">${esc(b.detail)}</div>` : ""}${stages}${lines}${unitHtml(b)}${ports ? `<div class="dg-ports">${ports}</div>` : ""}</div>`;
}

function renderDiagram(s) {
  const D = baseDiagram(s.diagram);
  $("diagram").hidden = !D;
  lastDiagram = D;
  if (!D) return;
  put("links-meta", `<span class="tone-${esc(D.summary.tone)}">${esc(D.summary.text)}</span>`);
  // 열이 많아도 한 화면에 들어오게: 열 수를 CSS 에 넘겨 상자 폭을 나누고, 많으면 촘촘한 모양으로 바꾼다
  $("diagram").style.setProperty("--dg-n", D.cols.length);
  $("diagram").classList.toggle("dg-tight", D.cols.length >= TIGHT_FROM_COLS);
  put("dg-cols", D.cols.map((col) => `<div class="dg-col">${col.map(boxHtml).join("")}</div>`).join(""));
  layoutFor(s.profile.id);
  applyLayout();
  renderExtra(D.extra);
  drawWires();
}

// 도메인에 실제로 있는데 그림이 선언하지 않은 토픽 — 정책·미션이 모르는 노드가 붙어 있으면 여기서 보인다
function renderExtra(extra) {
  $("dg-extra").hidden = !extra;
  if (!extra) return;
  const n = extra.topics.length;
  put("dg-extra-sum", `그림 밖 연결 ${n}개${extra.nodes.length ? ` · 그림에 없는 노드 ${extra.nodes.length}개` : ""}`);
  const rows = extra.topics.map((t) => `<tr><td class="mono">${esc(t.name)}</td><td class="mono dim">${esc((t.type || "").split("/").pop())}</td>
      <td class="mono">${t.pubs.length ? esc(t.pubs.join(", ")) : '<span class="tone-warn">내는 쪽 없음</span>'}</td><td class="dim">→</td>
      <td class="mono">${t.subs.length ? esc(t.subs.join(", ")) : '<span class="dim">받는 쪽 없음</span>'}</td></tr>`).join("");
  put("dg-extra-body", n ? `<table class="dg-extra-table"><thead><tr><th>토픽</th><th>타입</th><th>내는 노드</th><th></th><th>받는 노드</th></tr></thead><tbody>${rows}</tbody></table>`
    : `<div class="dim">그림에 없는 토픽이 없다 — 도메인의 연결이 전부 그림 안에 있다.</div>`);
}

// 전선은 상자가 자리를 잡은 뒤에 잰다 — 보낸 상자의 오른쪽 변에서 받는 포트 행의 왼쪽 끝으로.
function drawWires() {
  const D = lastDiagram, root = $("diagram"), svg = $("dg-wires");
  if (!D || root.hidden) return;
  const r0 = root.getBoundingClientRect(), sx = root.scrollLeft, sy = root.scrollTop, outs = {};
  D.wires.forEach((w) => (outs[w.from] ||= []).push(w.id));
  const paths = D.wires.map((w) => {
    const a = document.getElementById(`dg-box-${w.from}`), b = document.getElementById(`dg-port-${w.id}`);
    if (!a || !b) return "";
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect(), k = outs[w.from].indexOf(w.id), n = outs[w.from].length;
    const x1 = ra.right - r0.left + sx, y1 = ra.top - r0.top + sy + (ra.height * (k + 1)) / (n + 1);
    const x2 = b.closest(".dg-box").getBoundingClientRect().left - r0.left + sx, y2 = rb.top - r0.top + sy + rb.height / 2;
    const dx = Math.max(28, (x2 - x1) * 0.45);
    return `<path class="wire tone-${esc(w.tone)}${w.flow ? " flow" : ""}" d="M${x1.toFixed(1)},${y1.toFixed(1)} C${(x1 + dx).toFixed(1)},${y1.toFixed(1)} ${(x2 - dx).toFixed(1)},${y2.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}"/>`;
  }).join("");
  svg.setAttribute("width", root.scrollWidth);
  svg.setAttribute("height", root.scrollHeight);
  if (paths !== lastWires) { lastWires = paths; svg.innerHTML = paths; }      // 같으면 건드리지 않는다 — 흐름 애니메이션이 끊기지 않게
}

// ── 연결 그림 배치 ─────────────────────────────────────────────────────
// 상자 머리를 끌어 옮긴다(방향키도). 위치는 이 브라우저에만 프로파일별로 남는다 — 서버·미션·다른 화면과 무관하다.
// 오프셋은 HTML 에 넣지 않고 그린 뒤에 입힌다: 끄는 도중 상태가 바뀌어 상자가 다시 그려져도 끊기지 않게.
const LAYOUT_STEP = 8, LAYOUT_STEP_BIG = 32;
let layout = {}, layoutKey = "", drag = null;
function layoutFor(profileId) {
  const key = `s2r.layout.${profileId}`;
  if (key === layoutKey) return;
  layoutKey = key;
  try { layout = JSON.parse(localStorage.getItem(key) || "{}") || {}; } catch { layout = {}; }
}
function saveLayout() {
  try { localStorage.setItem(layoutKey, JSON.stringify(layout)); } catch { /* 저장이 막혀도 이 화면에서는 유지된다 */ }
}
function applyLayout() {
  const cols = $("dg-cols"), boxes = [...cols.querySelectorAll(".dg-box")];
  boxes.forEach((el) => {
    const o = layout[el.dataset.box];
    el.style.translate = o ? `${o.x}px ${o.y}px` : "";
    el.classList.toggle("moved", !!o);
  });
  cols.style.paddingBottom = "";                                             // 아래로 끌어낸 만큼 그림을 늘린다
  const bottom = cols.getBoundingClientRect().bottom;
  const over = Math.max(0, ...boxes.filter((el) => el.classList.contains("moved")).map((el) => el.getBoundingClientRect().bottom - bottom));
  cols.style.paddingBottom = over > 0 ? `${Math.ceil(over) + 8}px` : "";
  $("dg-reset").hidden = !Object.keys(layout).length;
}
function moveBox(id, x, y) {
  const el = document.getElementById(`dg-box-${id}`);
  if (!el) return;
  const o = layout[id] || { x: 0, y: 0 }, r = el.getBoundingClientRect(), c = $("dg-cols").getBoundingClientRect();
  const nx = Math.round(Math.max(x, c.left - (r.left - o.x))), ny = Math.round(Math.max(y, c.top - (r.top - o.y)));   // 그림 위·왼쪽 밖으로는 못 나간다
  const { [id]: _old, ...rest } = layout;
  layout = nx || ny ? { ...rest, [id]: { x: nx, y: ny } } : rest;
  applyLayout();
  drawWires();
}
$("diagram").addEventListener("pointerdown", (e) => {
  const head = e.target.closest(".dg-head");
  if (!head || e.button !== 0 || e.target.closest("button, a, input")) return;
  const id = head.closest(".dg-box").dataset.box, o = layout[id] || { x: 0, y: 0 };
  drag = { id, pid: e.pointerId, px: e.clientX, py: e.clientY, x0: o.x, y0: o.y };
  $("diagram").setPointerCapture(e.pointerId);
  $("diagram").classList.add("dragging");
  e.preventDefault();
});
$("diagram").addEventListener("pointermove", (e) => {
  if (drag && e.pointerId === drag.pid) moveBox(drag.id, drag.x0 + e.clientX - drag.px, drag.y0 + e.clientY - drag.py);
});
const endDrag = (e) => {
  if (!drag || e.pointerId !== drag.pid) return;
  drag = null;
  $("diagram").classList.remove("dragging");
  saveLayout();
};
$("diagram").addEventListener("pointerup", endDrag);
$("diagram").addEventListener("pointercancel", endDrag);
$("diagram").addEventListener("keydown", (e) => {
  const head = e.target.closest(".dg-head"), dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  if (!head || !dir) return;
  const id = head.closest(".dg-box").dataset.box, o = layout[id] || { x: 0, y: 0 }, step = e.shiftKey ? LAYOUT_STEP_BIG : LAYOUT_STEP;
  moveBox(id, o.x + dir[0] * step, o.y + dir[1] * step);
  saveLayout();
  e.preventDefault();
});

// 이 단계가 띄워 **지금 살아 있는** 유닛 — "다시 띄우기" 가 내릴 것들(09.23).
function liveUnits(stage) {
  const us = (S.session && S.session.units) || {};
  return Object.values(us).filter((u) => u.stage === stage && u.alive);
}

function unitModal(key) {
  const u = S.session.units[key], boxes = S.session.diagram.cols.flat().filter((b) => boxUnits(b).some((x) => x.key === key));
  modal(`<h3>끄기 — ${esc(key)}</h3><p>${esc(u.note)}</p>
    <p>이 프로세스를 그룹째 정지한다(SIGTERM → 5 s → SIGKILL). 그림에서 같이 꺼지는 상자: <b>${boxes.map((b) => esc(b.title)).join(" · ")}</b></p>
    <p>입력이 끊기면 정책 체인은 스스로 abort 하고 pd 는 HOLD 로 간다.</p>
    <div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button>
    <button class="btn btn-primary" data-act="unit-off-go" data-arg="${esc(key)}">끄기</button></div>`);
}

// 연결 사슬: 브리지 → 입력 → 정책 → 실기 → 가드. 상태·색·사유는 서버(links.py)가 정한다.
const LINK_WORD = { live: "연결됨", held: "보유", off: "꺼짐", unknown: "모름", stale: "끊김", missing: "없음", fault: "고장", down: "죽음" };
const ageText = (ms) => (ms === null || ms === undefined ? "—" : ms >= 1000 ? `${fmt(ms / 1000, 1)} s` : `${fmt(ms, 0)} ms`);

function renderLinks(s) {
  const broken = s.links.filter((b) => ["warn", "bad"].includes(b.tone));
  $("links-table").open = s.diagram ? $("links-table").open : true;      // 그림이 없는 프로파일은 표가 본문이다
  if (!s.diagram) put("links-meta", broken.length ? `끊긴 곳: ${broken.map((b) => esc(b.title)).join(" · ")}` : s.links.every((b) => b.tone === "ok") ? "전부 이어짐" : "");
  put("links", s.links.filter((b) => !workspaceNode(b)).map((b) => {
    const rows = b.rows.map((r) => `<tr class="tone-${esc(r.tone)}"><td><span class="lamp ${esc(r.tone)}"></span>${esc(r.name)}</td>
      <td class="num">${ageText(r.age_ms)}</td><td class="link-state">${esc(LINK_WORD[r.state] || r.state)}</td><td class="link-note">${esc(r.note)}</td></tr>`).join("");
    return `<div class="link tone-${esc(b.tone)}"><div class="link-head"><span class="lamp ${esc(b.tone)}"></span><b>${esc(b.title)}</b>
      <span class="link-state">${esc(LINK_WORD[b.state] || b.state)}</span></div>
      ${b.detail ? `<div class="link-detail">${esc(b.detail)}</div>` : ""}${rows ? `<table>${rows}</table>` : ""}</div>`;
  }).join('<div class="link-arrow">▸</div>'));
}

// ★10.21 사용자 "노드는 무슨 의미인지 모르겠네" — PD 상태 이름을 말로. 원래 이름(오류 문구에 나온다)은 옆에 작게.
const PD_PHASE = { IDLE: "대기 · 잡지 않음", RAMPING: "힘 올리는 중", TRACKING: "목표 따라감", HOLD: "자세 유지", RELEASING: "푸는 중" };

function renderNodes(s) {
  const br = s.bridge;
  put("bridge-meta", br.enabled ? `ROS 그래프 ${br.graph.length}개 노드${br.bad_lines ? ` · 해석 못 한 줄 ${br.bad_lines}` : ""}` : "ROS 연결 꺼짐 — 상태를 받지 않는다");
  const rows = s.nodes.map((n) => {
    const st = n.status;
    if (!st) return `<tr class="stale"><td><span class="lamp"></span><span class="node-name">${esc(n.name)}</span></td><td colspan="5">아직 상태를 받지 않았다 — 노드가 떠 있지 않다</td></tr>`;
    const phase = String(st.phase ?? "");
    const busy = ["running", "TRACKING", "RAMPING"].includes(phase);
    const lamp = n.stale ? "" : st.ok === false ? "bad" : busy ? "live" : "ok";
    const why = (st.reasons || []).length ? `<div class="node-reasons">${st.reasons.map(esc).join("<br>")}</div>` : "";
    const raw = det(`node:${n.name}`, "원본 status", `<pre class="node-json">${esc(JSON.stringify(st, null, 1))}</pre>`);
    return `<tr class="${n.stale ? "stale" : ""}"><td><span class="lamp ${lamp}"></span><span class="node-name">${esc(n.name)}</span>${why}${raw}</td>
      <td><span class="phase">${esc(PD_PHASE[phase] || phase || "—")}</span>${PD_PHASE[phase] ? ` <small class="phase-raw">${esc(phase)}</small>` : ""}</td><td class="num">${esc(st.episode ?? "—")}</td><td class="num">${esc(st.seq ?? "—")}</td>
      <td class="num">${fmt(st.proc_ms)}</td><td class="num">${n.stale ? "<b>" + fmt(n.age_s, 1) + " s</b>" : fmt(n.age_s, 1) + " s"}</td></tr>`;
  });
  const graph = br.graph.length ? det("graph", `보이는 ROS 노드 ${br.graph.length}개`, `<pre class="node-json">${br.graph.map(esc).join("\n")}</pre>`) : "";
  put("nodes", `<table><thead><tr><th>노드</th><th>상태</th><th class="num">에피소드</th><th class="num">명령 번호</th><th class="num">처리 ms</th><th class="num">마지막 수신</th></tr></thead><tbody>${rows.join("")}</tbody></table><p class="hint nodes-hint">제어 노드가 매 틱 보내는 상태 · 2 s 넘게 안 오면 흐리게</p><div style="padding:0 12px 10px">${graph}</div>`);
}

function renderMetrics(s) {
  // ★10.21 timing: false 프로파일(RH56F1 — 두 PD 가 팔마다 따로) 은 사슬 지연이 없다. 칸째 숨기고 노드 칸이 폭을 쓴다.
  const off = s.profile.timing === false;
  $("metrics-panel").hidden = off;
  $("col-live").classList.toggle("solo", off);
  if (off) return;
  const M = s.metrics, L = M.latency_ms, lat = s.profile.latency || [s.profile.status_nodes[0], s.profile.status_nodes.at(-1)];
  put("metrics-meta", `에피소드 ${esc(M.episode ?? "—")} · 최근 ${M.rows} seq`);
  const over = (v) => (M.budget_ms !== null && v !== null && v > M.budget_ms ? " over" : "");
  const procs = Object.entries(M.proc_ms).map(([n, v]) => `<span>${esc(n)}</span><canvas data-series="proc:${esc(n)}"></canvas><span>p50 ${fmt(v.p50)} · p95 ${fmt(v.p95)}</span>`).join("");
  const changed = put("metrics", `<div class="kpis">
      <div class="kpi"><small>지연 p50</small><b class="${over(L.p50)}">${fmt(L.p50)}</b> <span>ms</span></div>
      <div class="kpi"><small>지연 p95</small><b class="${over(L.p95)}">${fmt(L.p95)}</b> <span>ms</span></div>
      <div class="kpi"><small>예산 0.5·dt</small><b>${fmt(M.budget_ms)}</b> <span>ms</span></div>
      <div class="kpi"><small>seq 결손</small><b class="${M.seq_missing ? "over" : ""}">${esc(M.seq_missing)}</b></div></div>
    <div class="chart"><div class="chart-title"><span>지연 ${esc(lat[0])} → ${esc(lat[1])} (버킷 최댓값)</span><span>max ${fmt(L.max)} ms</span></div>
      ${M.latency_note && L.p50 === null ? `<p class="chart-note">${esc(M.latency_note)}</p>` : `<canvas data-series="latency"></canvas>`}</div>
    ${M.latency_note && L.p50 !== null ? `<p class="chart-note">${esc(M.latency_note)}</p>` : ""}
    <div class="procbars">${procs}</div>`);
  if (changed) drawCharts(M);
}

function drawCharts(M) {
  document.querySelectorAll("#metrics canvas").forEach((cv) => {
    const key = cv.dataset.series;
    const values = key === "latency" ? M.series.latency_ms : (M.series.proc_ms[key.slice(5)] || []);
    spark(cv, values, key === "latency" ? M.budget_ms : null);
  });
}

function spark(cv, values, budget) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  if (!values.length) { g.fillStyle = "#6b7785"; g.font = "11px monospace"; g.fillText("데이터 없음", 4, h / 2 + 4); return; }
  const top = Math.max(...values, budget || 0) * 1.15 || 1, y = (v) => h - 2 - (v / top) * (h - 6), x = (i) => (values.length === 1 ? w / 2 : (i / (values.length - 1)) * (w - 2) + 1);
  if (budget) { g.strokeStyle = "rgba(255,93,82,.75)"; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(0, y(budget)); g.lineTo(w, y(budget)); g.stroke(); g.setLineDash([]); }
  g.beginPath(); values.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
  g.strokeStyle = "#4da3ff"; g.lineWidth = 1.4; g.stroke();
  g.lineTo(x(values.length - 1), h); g.lineTo(x(0), h); g.closePath(); g.fillStyle = "rgba(77,163,255,.12)"; g.fill();
}

// 학습 가능 영역 지도(10.13 사용자: "Deploy 화면에 가능영역이 오버레이로") — 위에서 본 탁자, base 프레임.
// 화면 위 = +x(앞) · 왼쪽 = +y(왼팔 쪽). 실선 박스 = 정책이 학습한 컵 소환 범위, 점선 = 시작 거부 여유. 판정은 서버(spawn_map.py).
const MAP_TONE = { ok: "var(--ok)", warn: "var(--warn)", bad: "var(--bad)" };
const MAP_SIDE = { right: "#4da3ff", left: "#a78bfa" };
function spawnMap(m) {
  if (!m || !m.boxes.length) return "";
  const t = m.tol_m, xs = [0], ys = [0.3, -0.3];
  m.boxes.forEach((b) => { xs.push(b.lo[0] - t, b.hi[0] + t); ys.push(b.lo[1] - t, b.hi[1] + t); });
  m.objects.forEach((o) => { xs.push(o.x); ys.push(o.y); });
  const pad = 0.04, x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
  const yr = Math.max(...ys.map(Math.abs)) + pad;
  const W = 360, k = W / (2 * yr), H = Math.round((x1 - x0) * k);
  const u = (y) => (yr - y) * k, v = (x) => (x1 - x) * k;          // 위 = +x, 왼쪽 = +y
  const rect = (lo, hi, d) => `x="${u(hi[1] + d).toFixed(1)}" y="${v(hi[0] + d).toFixed(1)}" width="${((hi[1] - lo[1] + 2 * d) * k).toFixed(1)}" height="${((hi[0] - lo[0] + 2 * d) * k).toFixed(1)}"`;
  const boxes = m.boxes.map((b) => { const c = MAP_SIDE[b.side] || "#9aa6b2"; return `
      <rect ${rect(b.lo, b.hi, t)} fill="none" stroke="${c}" stroke-dasharray="4 3" stroke-width="1"/>
      <rect ${rect(b.lo, b.hi, 0)} fill="${c}" fill-opacity=".14" stroke="${c}" stroke-width="1.5"/>
      <text x="${u((b.lo[1] + b.hi[1]) / 2).toFixed(1)}" y="${(v(b.hi[0] + t) - 4).toFixed(1)}" text-anchor="middle" fill="${c}" font-size="11">${esc(b.label)} · ${esc(b.policy)}</text>`; }).join("");
  const grid = [];
  for (let x = Math.ceil(x0 * 10) / 10; x <= x1; x += 0.1) grid.push(`<line x1="0" x2="${W}" y1="${v(x).toFixed(1)}" y2="${v(x).toFixed(1)}"/><text x="2" y="${(v(x) - 2).toFixed(1)}">x ${x.toFixed(1)}</text>`);
  for (let y = -Math.floor(yr * 10) / 10; y <= yr; y += 0.1) grid.push(`<line y1="0" y2="${H}" x1="${u(y).toFixed(1)}" x2="${u(y).toFixed(1)}"/>`);
  const base = v(0) <= H ? `<path d="M${u(0) - 7},${v(0)} L${u(0) + 7},${v(0)} L${u(0)},${v(0) - 10} Z" fill="#9aa6b2"/><text x="${u(0) + 10}" y="${v(0) - 2}" fill="#9aa6b2" font-size="10">base</text>` : "";
  const dots = m.objects.map((o) => `<circle cx="${u(o.y).toFixed(1)}" cy="${v(o.x).toFixed(1)}" r="6" fill="${MAP_TONE[o.tone]}" stroke="#0d1117" stroke-width="1.5"/>
      <text x="${(u(o.y) + (u(o.y) > W * 0.6 ? -9 : 9)).toFixed(1)}" y="${(v(o.x) + 4).toFixed(1)}" text-anchor="${u(o.y) > W * 0.6 ? "end" : "start"}" fill="#e6edf3" font-size="11" paint-order="stroke" stroke="#0d1117" stroke-width="3">${esc(o.name)}</text>`).join("");
  const rows = m.objects.map((o) => `<dt><span class="lamp ${esc(o.tone)}"></span>${esc(o.name)}</dt><dd class="fpp-${esc(o.tone)}">x ${o.x.toFixed(3)} · y ${o.y.toFixed(3)} — ${esc(o.say)}</dd>`).join("");
  const none = m.stale ? "브리지 보고가 늦다 — 물체 자리를 믿지 않는다" : "물체 자세가 아직 없다";
  return `<div class="fpp-map"><div class="fpp-name"><b>학습 가능 영역</b><span class="hint">위 = 앞(+x) · 왼쪽 = 왼팔 쪽(+y) · 점선 = 시작 거부 여유 ${Math.round(t * 100)} cm</span></div>
    <svg class="spawn-map" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="학습 가능 영역과 물체 위치">
      <g class="spawn-grid">${grid.join("")}</g>${boxes}${base}${dots}</svg>
    <dl class="kv fpp-kv">${rows || `<dt></dt><dd class="hint">${esc(none)}</dd>`}</dl></div>`;
}

// FP++ 칸 — 09.29 사용자: "fpp 가 제대로 되는지 … 상태창에서". 줄 · 판정은 서버(object_health.py)가 준다.
// 그날 컨테이너는 Up 인데 추적 노드가 CUDA OOM 으로 죽어 자세가 0 이었다 — 컨테이너 상태만으로는 못 본다.
// ★10.21 사용자 "FPP 도 칸 비율만 많이 차지하고 쓸모없는 정보가 많음": policy 탭 맨 위 한 줄(판정 · 물체마다 램프 · 자리 · 한마디)로.
//   영상은 [영상] 을 눌러야 열린다(닫으면 스트림을 끊는다) · 학습 영역 지도와 줄별 상태는 '인지 상세' 를 펼쳐야 보인다.
let showVideo = false;
function fppOf(s, name) {
  const f = s.fpp || {}, o = (f.objects || []).find((x) => x.name === name);
  const m = ((f.map && f.map.objects) || []).find((x) => x.name === name);
  const worst = o && o.rows.find((r) => r.tone !== "ok");
  return { tone: (worst && o.tone) || (m && m.tone) || (o && o.tone) || "mute", xy: m ? `(${fmt(m.x, 3)}, ${fmt(m.y, 3)})` : "",
           say: worst ? `${worst.label} — ${worst.value}` : m ? m.say : o ? "자리 없음" : "FP++ 칸에 없는 물체" };
}
function renderFpp(s) {
  const f = s.fpp;
  $("pstrip").hidden = !f;
  $("fpp-panel").hidden = !(f && showVideo);
  if (!f) return put("fpp", "");
  if (showVideo) updateVideo(s);
  $("pstrip").className = `pstrip tone-${f.tone}`;
  // 물체마다 같은 이유(예: 인지 런처가 안 떴다)면 한 번만 쓴다 — 같은 문장 다섯 줄은 읽히지 않는다
  const cs = f.objects.map((o) => ({ o, c: fppOf(s, o.name) }));
  const same = cs.length > 1 && cs.every((x) => x.c.say === cs[0].c.say) ? cs[0].c.say : "";
  const objs = cs.map(({ o, c }) => `<li class="ps-obj"><span class="lamp ${esc(c.tone)}"></span><b>${esc(o.name)}</b>${c.xy ? `<span class="mono">${esc(c.xy)}</span>` : ""}${same ? "" : `<span class="ps-say">${esc(c.say)}</span>`}</li>`).join("");
  put("pstrip", `<span class="ps-tag">FP++</span><b class="ps-verdict">${esc(f.say)}${f.stale ? " · 브리지 보고가 늦다" : ""}</b>
    <ul class="ps-objs">${objs || '<li class="hint">추적 물체가 없다</li>'}${same ? `<li class="ps-say ps-common">모두 ${esc(same)}</li>` : ""}</ul>
    <button class="btn btn-sm btn-ghost${showVideo ? " on" : ""}" data-act="xp-video" aria-pressed="${showVideo}">${showVideo ? "영상 닫기" : "영상"}</button>`);
  put("fpp", spawnMap(f.map) + f.objects.map((o) => `<div class="fpp-obj"><div class="fpp-name"><span class="lamp ${esc(o.tone)}"></span><b>${esc(o.name)}</b>
    <span class="hint">${esc(o.topic)}${f.stale ? " · 브리지 보고가 늦다" : ""}</span></div>
    <dl class="kv fpp-kv">${o.rows.map((r) => `<dt><span class="lamp ${esc(r.tone)}"></span>${esc(r.label)}</dt><dd class="fpp-${esc(r.tone)}">${esc(r.value)}</dd>`).join("")}</dl></div>`).join(""));
}

// 에피소드(10.04) — 연결 그림처럼 노드 상자를 순서대로 잇고, 재시도 · 되돌아가기는 피드백 화살표로 그린다(사용자 10.04).
// 상자 = 노드(순서 번호 · 무엇 · 기대 상태 · 몇 번 돌았나 · 걸린 시간), 실선 = 다음 노드, 주황 점선 = 실패 시 되돌아가는 길.
// 버튼: [다음](구분 실행 — 노드 이름 입력) · [연속 실행](episode:<이름> 한 번) · [새 에피소드] · ■ 정지(lease 없이).
const EP_TONE = { READY: "", RUNNING: "warn", SUCCESS: "ok", FAILURE: "bad", STOPPED: "bad" };
const EP_STATE = { pending: "대기", running: "실행 중", done: "완료", failed: "실패", recovering: "복구 대기", stopped: "정지",
                   waiting: "확인 대기" };
let lastEpFlow = null, lastEpWires = "";

function epNodeState(e, n, i) {
  const r = (e.runs || {})[n.id];
  if (r && r.state && !(r.state === "done" && i >= e.index && e.phase !== "SUCCESS")) return r.state;
  return i < e.index ? "done" : "pending";
}

function epBoxHtml(e, n, i) {
  const st = epNodeState(e, n, i), r = (e.runs || {})[n.id] || {};
  const cur = i === e.index && !["SUCCESS", "FAILURE", "STOPPED"].includes(e.phase);
  const busy = cur && e.busy && st === "running";
  const att = (e.attempts || {})[n.id];
  const foot = [r.n ? `${r.n}번 실행` : "", r.s !== undefined && r.s !== null ? `${fmt(r.s, 1)} s` : "",
                att ? `재시도 ${att}` : "", r.code && r.code !== "COMPLETED" ? r.code : ""].filter(Boolean);
  return `<div class="epf-box st-${esc(st)}${cur ? " is-current" : ""}" id="epf-box-${esc(n.id)}">
    <div class="epf-head"><span class="epf-idx" aria-hidden="true">${st === "done" ? "✓" : st === "failed" || st === "stopped" ? "✕" : i + 1}</span>
      <b>${esc(n.id)}</b><span class="epf-state">${busy ? "⟳ " : ""}${esc(EP_STATE[st] || st)}</span></div>
    <div class="epf-kind">${esc(n.title)}${n.checkpoint ? ` <span class="epf-cp" title="되돌아갈 수 있는 자리">⚑ ${esc(n.checkpoint)}</span>` : ""}</div>
    ${(n.lines || []).map((l) => `<div class="epf-line">${esc(l)}</div>`).join("")}
    ${n.expect ? `<div class="epf-expect">→ ${esc(n.expect)}</div>` : ""}
    ${foot.length ? `<div class="epf-foot">${foot.map((x) => `<span>${esc(x)}</span>`).join("")}</div>` : ""}</div>`;
}

// ★10.19 사용자 "좌우 동시 실행": 한 팔 에피소드는 팔마다 실행기(episode_runner_right · _left), 두 팔(tilt)은 episode_runner.
// 패널은 고른 실행기 하나를 보인다(고르지 않았으면 서버가 고른 것) — 명령은 그 실행기로만 간다.
const RUNNER_LABEL = { episode_runner: "두 팔", episode_runner_right: "오른팔", episode_runner_left: "왼팔" };
let epRunner = "";
function shownRunner(s) {
  const all = s.episode_runners || [];
  return all.find((r) => r.runner === epRunner) || s.episode_runner;
}
function renderEpisode(s) {
  const e = shownRunner(s);
  $("epf-panel").hidden = !e;
  lastEpFlow = e && e.flow ? e : null;
  if (!e) return;
  const all = s.episode_runners || [];
  const tabs = all.length > 1 ? `<div class="actions ep-runners" role="group" aria-label="에피소드 실행기">${all.map((r) => `<button class="btn btn-sm${r.runner === e.runner ? " on" : ""}"
    data-act="ep-runner" data-arg="${esc(r.runner)}" aria-pressed="${r.runner === e.runner}">${esc(RUNNER_LABEL[r.runner] || r.runner)} · ${esc(r.episode)}
    <span class="badge ${EP_TONE[r.phase] || ""}">${esc(r.phase)}${r.busy ? " · 실행 중" : ""}</span></button>`).join("")}</div>` : "";
  const can = holding() && !e.stale;
  put("epf-meta", `<span class="badge ${EP_TONE[e.phase] || ""}">${esc(e.phase)}${e.busy ? " · 실행 중" : ""}</span>
    <span class="hint">${esc(e.episode)} · ${esc(e.episode_id)}${e.stale ? " · 상태가 늦다" : ""}</span>`);
  const next = e.next_action, done = ["SUCCESS", "FAILURE", "STOPPED"].includes(e.phase);
  put("epf-ctl", `${tabs}<div class="actions">
    <button class="btn btn-sm btn-real" data-act="ep-next" data-arg="${esc(next || "")}" ${can && next && !e.busy ? "" : "disabled"}
      title="노드 하나(구분 실행) — 이름을 입력해 승인">다음: ${esc(next || "—")}</button>
    <button class="btn btn-sm btn-real" data-act="ep-run" data-arg="episode:${esc(e.episode)}" ${can && !done && !e.busy ? "" : "disabled"}
      title="끝까지(연속 실행) — 승인 한 번, 실패면 정해진 복구 또는 정지">연속 실행</button>
    <button class="btn btn-sm btn-ghost" data-act="ep-reset" ${can && !e.busy ? "" : "disabled"} title="WorldState 처음부터">새 에피소드</button>
    <button class="btn-stop" data-act="ep-stop" title="이 실행기 정지 — 정책 노드 episode/stop(pd 가 붙든다) · 재생 중단">■ ${esc(RUNNER_LABEL[e.runner] || e.runner || "")} 실행기 정지</button>
    ${e.auto_approve ? '<span class="hint">auto_approve(fake)</span>' : ""}</div>`);
  const nodes = (e.flow && e.flow.nodes) || [];
  put("epf-row", nodes.map((n, i) => epBoxHtml(e, n, i)).join(""));
  const w = e.world || {};
  const objs = Object.entries(w.objects || {}).map(([k, v]) => `${esc(k)}: ${esc(v.at)}`).join(" · ") || "—";
  // 사유가 "policy_timeout — 재시도 2/2 다 씀" 처럼 코드로 시작하면 코드를 한 번만 쓴다
  const lc = (e.last && e.last.code) || "", lr = String((e.last && e.last.reason) || "");
  const why = (lc && lr.startsWith(lc) ? lr.slice(lc.length) : lr).replace(/^\s*[—-]\s*/, "").trim();
  // 확인 대기는 아래 conf 줄이 말한다 — 같은 문장을 두 번 쓰지 않는다
  const last = e.last && e.last.node && lc !== "waiting_confirm" ? `<div class="note${lc && lc !== "none" ? " warn" : ""}">${esc(e.last.node)} → <b>${esc(lc)}</b>${why ? ` — ${esc(why)}` : ""}</div>` : "";
  const pend = e.pending ? `<div class="note warn">복구 대기: <b>${esc(e.pending.action)}</b> — ${esc(e.pending.reason)} · [다음] 으로 승인</div>` : "";
  // 10.17 경로 B 확인 노드 — 연속 실행도 여기서 멈춘다. 확인할 것을 크게 보이고 [다음] 으로 승인한다
  const conf = e.confirm ? `<div class="note warn epf-confirm">운영자 확인 <b>${esc(e.confirm.node)}</b> — ${esc(e.confirm.text)} · 확인했으면 [다음]</div>` : "";
  // .events li 는 3열 그리드(시각·점·본문) — 자식을 꼭 3개로 묶는다(ui-pages.md 사고 기록)
  const hist = (e.history || []).slice(-12).reverse().map((h) => `<li class="ev-episode"><time>${esc(String(h.timestamp || "").slice(11, 19))}</time><span class="ev-dot"></span>
    <span>${esc(h.event)} <b>${esc(h.state_id || "")}</b> ${esc(h.result || "")} <span class="hint">${esc(h.termination_reason || "")}</span></span></li>`).join("");
  put("epf-info", `<dl class="kv epf-kv"><dt>자세</dt><dd>${esc(w.pose)}</dd><dt>오른손</dt><dd>${esc(w.right_hand)}</dd>
    <dt>왼손</dt><dd>${esc(w.left_hand)}</dd><dt>진행</dt><dd>${esc((w.task_flags || []).join(", ") || "—")}</dd>
    <dt>물체</dt><dd>${objs}</dd><dt>기록</dt><dd class="hint">${esc(rel(e.run_dir || ""))}</dd></dl>
    ${conf}${pend}${last}${det("ep-hist", "최근 기록", `<ol class="events">${hist}</ol>`)}`);
  requestAnimationFrame(drawEpWires);
}

// ── 10.17 policy 탭 tilt 평가(사용자 "policy 부분에 내가 말한 부분들이 시각화 잘되게") ─────────────────
// 이번 실행 계획 검사 · 손 쥐기 → 회차 기록. 값은 서버 tilt_view 가 모은다(읽기만). 성공 · 실패는 색과 글자(✓ · ✗) 둘 다로 보인다.
function tiltValue(v, unit) {
  const one = (x) => (typeof x === "number" ? (Math.abs(x) >= 100 ? x.toFixed(0) : String(Number(x.toFixed(3)))) : esc(x ?? "—"));
  const body = Array.isArray(v) ? v.map((x) => (Array.isArray(x) ? `[${x.map(one).join(", ")}]` : one(x))).join(" · ") : one(v);
  return `${body}${unit ? ` <small>${esc(unit)}</small>` : ""}`;
}
function tiltRunCard(c) {
  const head = `<header><b>${esc(c.node)}</b><span>${esc({ carry: "이송 계획", tilt: "tilt 계획", hand: "손 쥐기" }[c.kind] || c.kind)}</span>
    <span class="badge ${c.ok ? "ok" : "bad"}">${c.ok ? "✓ 통과" : "✗ 걸림"}</span>
    ${c.T_end_s != null ? `<span class="meta">${fmt(c.T_end_s, 1)} s</span>` : ""}${c.tilt_max_deg != null ? `<span class="meta">최대 기울기 ${fmt(c.tilt_max_deg, 1)}°</span>` : ""}</header>`;
  if (c.error) return `<article class="tilt-card bad">${head}<p>${esc(c.error)}</p></article>`;
  if (c.kind === "hand") {
    return `<article class="tilt-card${c.ok ? "" : " bad"}">${head}<dl class="kv"><dt>손</dt><dd>${esc(SIDE_KO[c.side] || c.side)}</dd>
      <dt>결과</dt><dd>${esc(c.phase)}</dd><dt>래치 손가락</dt><dd>${esc(c.latched)}</dd><dt>이유</dt><dd>${esc(c.why || "—")}</dd></dl></article>`;
  }
  const rows = (c.rows || []).map((r) => `<tr class="${r.ok ? "ok" : "bad"}"><td>${esc(r.name)}</td><td>${tiltValue(r.value, r.unit)}</td>
    <td>${r.limit == null ? "—" : tiltValue(r.limit, r.unit)}</td><td>${r.ok ? "✓" : "✗"}</td></tr>`).join("");
  return `<article class="tilt-card${c.ok ? "" : " bad"}">${head}<table class="tilt-tbl"><thead><tr><th>검사</th><th>값</th><th>한계</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table></article>`;
}
function renderTilt(s) {
  // ★10.21 경로 카드(노드 사슬 · 쓸 수 있는지 · 언제)는 실험 목록이 보인다 — 여기는 이번 실행 계획 검사와 회차 기록(접은 칸)
  const t = s.tilt;
  for (const id of ["tilt-run-panel", "tilt-rec-panel"]) $(id).hidden = !t;
  if (!t) return;
  const run = t.run;
  $("tilt-run-panel").hidden = !run;
  if (run) {
    put("tilt-run-meta", `${esc(run.episode)} · ${esc(rel(run.run_dir))}`);
    put("tilt-run", (run.cards || []).length ? run.cards.map(tiltRunCard).join("")
      : `<p class="hint">아직 계획한 궤적 · 손 쥐기가 없다 — 이송 · tilt 노드가 돌면 여기에 검사 값이 뜬다(움직이기 전에 계산한다).</p>`);
  }
  const r = t.records;
  if (!r) return;
  const sum = r.summary || {}, tl = sum.tilt || {}, gr = sum.grasp || {};
  put("tilt-rec-meta", `${esc(r.day)} · logs/eval_real${(t.errors || []).length ? ` · 집계 오류 ${t.errors.length}` : ""}`);
  const tiltRows = (r.tilt || []).slice().reverse().map((x) => `<tr class="${x.success ? "ok" : "bad"}"><td>${esc(x.trial)}</td><td>${esc(x.path)}</td>
    <td>${esc(x.in_rcv)}/${esc(x.n_beads)}</td><td>${esc(x.in_src)}</td><td>${esc(x.spilled)}</td><td>${x.success ? "✓ 성공" : "✗ 실패"}</td>
    <td>${esc(x.grasp_right)} · ${esc(x.grasp_left)}</td><td>${esc(x.video || "—")}</td></tr>`).join("");
  const graspRows = (r.grasp || []).slice().reverse().map((x) => `<tr class="${x.success ? "ok" : "bad"}"><td>${esc(x.trial)}</td><td>${esc(SIDE_KO[x.side] || x.side)}</td>
    <td>${esc(x.result)}</td><td>${x.lift_cm == null ? "—" : fmt(x.lift_cm, 1) + " cm"}</td><td>${x.hold_s == null ? "—" : fmt(x.hold_s, 1) + " s"}</td><td>${x.success ? "✓" : "✗"}</td></tr>`).join("");
  const tiles = [...Object.entries(tl).map(([k, v]) => `<div class="tilt-tile"><small>tilt 경로 ${esc(k)}</small><b>${esc(v.success)}/${esc(v.n)}</b>
      <span>받는 비율 평균 ${fmt(v.fill_mean, 2)} · 흘림 ${esc(v.spilled_total)} 개</span></div>`),
    ...Object.entries(gr).map(([k, v]) => `<div class="tilt-tile"><small>파지 확인 ${esc(SIDE_KO[k] || k)}</small><b>${esc(v.success)}/${esc(v.n)}</b>
      <span>${v.n >= 5 ? (v.success >= 4 ? "4/5 이상 → 경로 A" : "4/5 미만 → 경로 C") : `${5 - v.n} 회 남음`}</span></div>`)].join("");
  // 10.18 sim 비교 기준 — 실기 회차와 같은 판정(끝에 받는 컵 전부)으로 센 sim 값 · 1-C(경로 C 시뮬)
  const sim = (t.sim || []).map((x) => `<div class="tilt-tile sim"><small>sim ${esc(x.kind === "1c" ? "1-C 경로 C" : "기본 궤적")} · ${esc(x.name)}</small>
      <b>${esc(x.ok)}/${esc(x.n)}</b><span>${esc(x.note)}</span></div>`).join("");
  put("tilt-rec", `<div class="tilt-tiles">${tiles || '<p class="hint">오늘 기록이 없다.</p>'}</div>
    ${sim ? `<p class="tilt-rule"><b>sim 비교 기준</b> 같은 판정으로 센 sim 값(logs/sim)</p><div class="tilt-tiles">${sim}</div>` : ""}
    <p class="tilt-rule"><b>tilt 성공</b> ${esc(r.criterion)}</p><p class="tilt-rule"><b>파지</b> ${esc(r.grasp_rule)}</p>
    ${tiltRows ? `<table class="tilt-tbl"><thead><tr><th>회</th><th>경로</th><th>받는 컵</th><th>병에 남음</th><th>흘림</th><th>판정</th><th>파지 우 · 좌</th><th>영상</th></tr></thead><tbody>${tiltRows}</tbody></table>` : ""}
    ${graspRows ? `<table class="tilt-tbl"><thead><tr><th>회</th><th>손</th><th>결과</th><th>들림</th><th>유지</th><th></th></tr></thead><tbody>${graspRows}</tbody></table>` : ""}
    ${det("tilt-rec-cmd", "기록하는 법", `<pre class="tilt-cmd">${esc(r.command)}\npython3 deploy/policy_control/tools/eval_record.py grasp --side right --ok --lift-cm 7 --hold-s 3\npython3 deploy/policy_control/tools/eval_record.py summary</pre>`)}`);
}

// ── ★10.21 policy 탭 실험(사용자 "정책을 찾기 어렵고 실험 진행이 어려움 · 깔끔하고 명확하게") ─────────────────────
// 실험 = 경로 하나(팔마다 또는 두 팔). 목록에서 고르면 그 실험의 정책 사슬 · 컵 · 조건 · 실행 버튼이 한 칸에 모인다.
// 사슬 · 컵 · 쓸 수 있는지는 서버(experiments.py — 에피소드 파일 · 정책 등록부)가, 실행할 수 있는지 · 사유는 관리 작업(ctl_* —
// 파지 · 들기) · 미션 단계(tilt)가 정한다. 여기서는 그것을 그릴 뿐이다. 정책 카드 배치(←→↑↓)는 사용자 "일단 제거".
const XP_KEY = "s2r.xp";
let xpSel = (() => { try { const v = JSON.parse(localStorage.getItem(XP_KEY) || "{}"); return v && typeof v === "object" ? v : {}; } catch { return {}; } })();
let xpPol = "";                 // 펼친 정책 칩(id) — 다시 누르면 닫는다
const XP_STEP = { policy: "정책", handoff: "넘김", script: "스크립트", move: "궤적", confirm: "사람 확인", hand: "손" };
const SIDE_SHORT = { right: "R", left: "L" };
const POL_STATUS = { candidate: "후보", verified: "검증", deployed: "배포", hold: "보류" };
const xpSave = () => { try { localStorage.setItem(XP_KEY, JSON.stringify(xpSel)); } catch { /* 저장이 막혀도 이 화면에서는 유지된다 */ } };

// 실험 한 쪽의 지금 — 실행 중 · 준비됨 · 막힘. 서버가 준 것(작업 enabled · reasons · 단계 can_run · 실행기 phase)만 쓴다.
function xpState(s, side) {
  const runner = (s.episode_runners || []).find((r) => r.runner === side.runner);
  const mine = runner && runner.episode === side.episode ? runner : null;
  const busy = !!(mine && (mine.busy || mine.phase === "RUNNING"));
  const problems = side.problems || [];
  if (side.action) {
    const a = (S.actions || []).find((x) => x.id === side.action) || null;
    const reasons = [...problems, ...(a ? a.reasons || [] : ["이 미션에 실행 작업이 없다"])];
    return { kind: "action", a, runner: mine, live: busy || !!(a && a.job && a.job.active), ok: !!(a && a.enabled) && !problems.length, reasons };
  }
  const row = ((s.mission && s.mission.rows) || []).find((r) => r.id === side.stage) || null;
  const reasons = [...problems, ...(row ? row.reasons || [] : ["이 미션에 단계가 없다"])];
  return { kind: "stage", row, runner: mine, live: busy || !!(row && row.status === "RUNNING"), ok: !!(row && row.can_run) && !problems.length, reasons };
}
const xpTone = (st) => (st.live ? "live" : st.ok ? "ok" : "bad");
const xpWord = (st) => (st.live ? "실행 중" : st.ok ? "준비됨" : "막힘");
const sideName = (k) => SIDE_KO[k] || "두 팔";

// 고른 실험 · 쪽. 고른 적이 없거나 사라졌으면 지금 도는 실험, 그것도 없으면 첫 실험.
function xpPick(s, X) {
  const items = X.items || [];
  const it = items.find((i) => i.id === xpSel.id)
    || items.find((i) => Object.values(i.sides).some((sd) => xpState(s, sd).live)) || items[0];
  if (!it) return [null, "", null];
  const keys = Object.keys(it.sides), key = keys.includes(xpSel.side) ? xpSel.side : keys[0];
  return [it, key, it.sides[key]];
}
// 고르면 '지금 에피소드' 칸도 그 실험의 실행기를 보인다
function xpSyncRunner() {
  const X = S && S.session && S.session.experiments;
  if (!X) return;
  const side = xpPick(S.session, X)[2];
  if (side) epRunner = side.runner;
}

function renderExperiments(s) {
  const X = s.experiments;
  $("xp").hidden = !X;
  if (!X) return;
  const [it, key, side] = xpPick(s, X);
  const groups = [["grasp", "파지 · 들기", "한 팔"], ["tilt", "tilt", "두 팔"]];
  put("xp-list", (X.error ? `<p class="reasons bad">${esc(X.error)}</p>` : "") + groups.map(([g, title, sub]) => {
    const rows = (X.items || []).filter((i) => i.group === g).map((i) => {
      const on = !!it && i.id === it.id;
      const dots = Object.entries(i.sides).map(([k, sd]) => {
        const st = xpState(s, sd);
        return `<span class="xp-dot ${xpTone(st)}" title="${esc(`${sideName(k)} · ${xpWord(st)}${st.reasons.length && !st.ok ? ` — ${st.reasons[0]}` : ""}`)}">${esc(SIDE_SHORT[k] || "R+L")}</span>`;
      }).join("");
      return `<button class="xp-item${on ? " on" : ""}" data-act="xp-pick" data-arg="${esc(i.id)}" aria-pressed="${on}">
        <span class="xp-key">${esc(i.key)}</span><span class="xp-sum">${esc(i.summary)}</span><span class="xp-dots">${dots}</span></button>`;
    }).join("");
    return rows ? `<div class="xp-group"><h2>${esc(title)}<small>${esc(sub)}</small></h2>${rows}</div>` : "";
  }).join(""));
  put("xp-main", it ? xpMain(s, X, it, key, side) : `<p class="hint xp-empty">이 미션에 실험이 없다.</p>`);
}

function xpMain(s, X, it, key, side) {
  const st = xpState(s, side), tone = xpTone(st);
  const sides = Object.entries(it.sides);
  const seg = sides.length > 1 ? `<div class="seg" role="group" aria-label="팔">${sides.map(([k, sd]) =>
      `<button class="seg-b${k === key ? " on" : ""}" data-act="xp-side" data-arg="${esc(k)}" aria-pressed="${k === key}"><span class="lamp ${xpTone(xpState(s, sd))}"></span>${esc(sideName(k))}</button>`).join("")}</div>` : "";
  const head = `<header class="xp-head"><span class="xp-key xp-key-lg">${esc(it.key)}</span>
    <div class="xp-title"><h3>${esc(it.label)} <span>· ${esc(sideName(key))}</span></h3><p>${esc(it.summary)}</p></div>
    ${seg}<span class="badge ${tone}">${esc(xpWord(st))}</span></header>`;
  const open = xpPol && side.steps.some((x) => x.id === xpPol) ? xpPolicy((X.policies || {})[xpPol], xpPol) : "";
  const ready = st.live ? `<div class="xp-ready live">⟳ 실행 중 — 진행 · 확인 · 정지는 아래 <b>지금 에피소드</b></div>`
    : st.ok ? `<div class="xp-ready ok">✓ 실행할 수 있다</div>`
    : `<div class="xp-ready bad"><b>✕ 지금 실행할 수 없다</b><ul>${st.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;
  return head + (side.error ? `<p class="reasons bad">${esc(side.error)}</p>` : "") + xpRail(X, side, st.runner) + open
    + xpFacts(s, it, side) + ready + xpGo(s, it, key, side, st);
}

// 사슬 — 칸 = 궤적 · 사람 확인 · 손 · 정책(누르면 체크포인트 · 계약 · sim 근거) · 넘김(조건). 실행기가 이 실험을 돌리면 지난 칸 ✓ · 지금 칸 강조.
function xpRail(X, side, runner) {
  const idx = runner ? Number(runner.index) : -1, won = !!runner && runner.phase === "SUCCESS";
  const two = !side.action;                        // 두 팔 실험은 칸마다 팔을 적는다
  return `<ol class="xp-rail">${side.steps.map((x) => {
    const at = !runner ? "" : won || x.index < idx ? " done" : x.index === idx ? (runner.confirm ? " wait" : " run") : "";
    const arm = two && x.sides.length ? `<i class="xs-arm">${esc(x.sides.map((k) => SIDE_SHORT[k] || k).join("+"))}</i>` : "";
    if (x.kind === "handoff") return `<li class="xs xs-handoff${at}"><div class="xs-box"><small>넘김</small><span>${esc(x.sub)}</span></div></li>`;
    const p = x.kind === "policy" ? (X.policies || {})[x.id] : null;
    const meta = p ? `${p.ep != null ? `ep ${p.ep} · ` : ""}${POL_STATUS[p.status] || p.status || "—"}` : x.kind === "policy" ? "등록부에 없다" : x.sub;
    const body = `<small>${esc(x.kind === "policy" ? x.sub : XP_STEP[x.kind] || x.kind)}${arm}</small><b>${esc(x.text)}</b><span class="xs-meta">${esc(meta)}</span>`;
    if (x.kind !== "policy") return `<li class="xs xs-${esc(x.kind)}${at}" title="${esc(x.sub)}"><div class="xs-box">${body}</div></li>`;
    const bad = !p || (p.issues || []).length;
    return `<li class="xs xs-policy${at}${xpPol === x.id ? " open" : ""}${bad ? " bad" : ""}"><button class="xs-box" data-act="xp-pol" data-arg="${esc(x.id)}"
      aria-expanded="${xpPol === x.id}" title="${esc(x.id)} — 눌러서 체크포인트 · 계약 · sim 근거">${body}</button></li>`;
  }).join("")}</ol>`;
}

function xpPolicy(p, id) {
  if (!p) return `<div class="xp-pol"><p class="reasons bad">${esc(id)} — 등록부에 없다</p></div>`;
  return `<div class="xp-pol"><header><b>${esc(p.short)}</b><code>${esc(p.id)}</code>
      <span class="badge ${(p.issues || []).length ? "bad" : p.status === "hold" ? "warn" : ""}">${esc(POL_STATUS[p.status] || p.status || "—")}</span>
      <span class="spacer"></span><button class="btn btn-sm btn-ghost" data-act="xp-pol" data-arg="${esc(id)}">닫기</button></header>
    ${p.summary ? `<p>${esc(p.summary)}</p>` : ""}${p.eval ? `<p class="xp-eval"><b>sim 근거</b>${esc(p.eval)}</p>` : ""}
    <dl class="kv"><dt>체크포인트</dt><dd class="mono">${esc(p.checkpoint || "—")}</dd><dt>계약</dt><dd class="mono">${esc(p.contract || "—")}</dd>
      <dt>카드</dt><dd class="mono">${esc(p.path)}/policy.yaml</dd></dl>
    ${(p.issues || []).length ? `<ul class="reasons bad">${p.issues.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}</div>`;
}

function xpFacts(s, it, side) {
  const cups = Object.entries(side.cups || {});
  const cupHtml = cups.map(([role, name]) => {
    const c = fppOf(s, name);
    return `<span class="xp-cup"><span class="lamp ${esc(c.tone)}"></span>${cups.length > 1 ? `<small>${esc(role)}</small>` : ""}<b>${esc(name)}</b>${c.xy ? `<span class="mono">${esc(c.xy)}</span>` : ""}<span class="hint">${esc(c.say)}</span></span>`;
  }).join("");
  const conf = side.confirms ? `확인 노드 ${side.confirms} 개 — 연속 실행도 거기서 멈춘다([다음] 으로 승인)` : "확인 노드 없음 — 끝까지 돈다(멈추려면 ■ 정지)";
  const route = side.action ? `관리 작업 <code>${esc(side.action)}</code>` : `미션 단계 <code>${esc(side.stage)}</code>`;
  return `<dl class="xp-facts"><dt>컵</dt><dd>${cupHtml || "—"}</dd>${it.when ? `<dt>언제</dt><dd>${esc(it.when)}</dd>` : ""}<dt>확인</dt><dd>${esc(conf)}</dd>
    <dt>실행</dt><dd>${route} · 실행기 <code>${esc(side.runner)}</code> · <code>config/episodes/${esc(side.episode)}.yaml</code></dd></dl>`;
}

function xpGo(s, it, key, side, st) {
  const real = s.profile.domain_class === "real", can = holding(), clean = !(side.problems || []).length;
  const name = `${it.label} ${sideName(key)}`;
  if (st.kind === "action") {
    const a = st.a, ok = can && !!a && a.enabled && clean;
    return `<div class="xp-go"><button class="btn ${real ? "btn-real" : "btn-primary"} xp-run" data-act="action-prepare" data-arg="${esc(side.action)}" ${ok ? "" : "disabled"}>▶ ${esc(name)} 실행</button>
      <span class="hint">${can ? "승인 창에서 명령을 보고 실행한다 — 진행 · 정지는 아래 지금 에피소드" : "먼저 조작 권한을 잡을 것"}</span>
      ${a ? actionState(a) + scriptGrip(a) : ""}</div>`;
  }
  const r = st.row;
  if (!r) return "";
  const approve = r.touches_real && !r.approved
    ? `<button class="btn btn-real" data-act="approve" data-arg="${esc(r.id)}" ${can && r.can_approve ? "" : "disabled"}>승인…</button>` : "";
  return `<div class="xp-go">${approve}<button class="btn ${r.touches_real ? "btn-real" : "btn-primary"} xp-run" data-act="run" data-arg="${esc(r.id)}" ${can && r.can_run && clean ? "" : "disabled"}>▶ ${esc(name)} 단계 실행</button>
    <span class="hint">${can ? "단계가 실행기 · 정책 노드를 띄운다 — 그다음 아래 지금 에피소드의 [연속 실행]" : "먼저 조작 권한을 잡을 것"}</span></div>`;
}

// 정책 목록(접은 칸) — 등록부 전부. 실험에 쓰는 것은 그 실험으로 바로 간다.
const shortPolicy = (id) => String(id || "").split("/").pop().replace(/^(right|left|both)_/, "").replace(/^approach_/, "");
function renderPolicyTable(s) {
  const X = s.experiments, cat = (S.workspace && S.workspace.catalog) || [];
  const uses = {};
  for (const it of (X && X.items) || []) for (const sd of Object.values(it.sides)) for (const x of sd.steps) {
    if (x.kind === "policy" && x.id) (uses[x.id] = uses[x.id] || new Map()).set(it.id, it.label);
  }
  const order = { right: 0, left: 1, both: 2 };
  const rows = [...cat].sort((a, b) => String(a.task).localeCompare(String(b.task)) || (order[a.side] ?? 3) - (order[b.side] ?? 3)
      || String(a.id).localeCompare(String(b.id))).map((p) => {
    const ep = /ep_?(\d+)/.exec(String(p.checkpoint || ""));
    const used = [...(uses[p.id] || new Map()).entries()].map(([id, label]) => `<button class="btn btn-sm btn-ghost" data-act="xp-pick" data-arg="${esc(id)}">${esc(label)}</button>`).join("");
    return `<tr class="${(p.issues || []).length ? "bad" : ""}"><td><b>${esc(shortPolicy(p.id))}</b></td><td>${esc(sideName(p.side))}</td><td>${esc(taskLabel(p.task))}</td>
      <td>${ep ? esc(ep[1]) : "—"}</td><td>${esc(POL_STATUS[p.status] || p.status || "—")}</td><td class="pol-uses">${used || '<span class="hint">—</span>'}</td><td class="mono hint">${esc(p.id)}</td></tr>`;
  }).join("");
  put("pol-meta", `${cat.length} 개 · 실험에 쓰는 것 ${cat.filter((p) => uses[p.id]).length}`);
  put("pol-table", rows ? `<table class="tilt-tbl pol-tbl"><thead><tr><th>정책</th><th>팔</th><th>작업</th><th>ep</th><th>상태</th><th>쓰는 실험</th><th>id</th></tr></thead><tbody>${rows}</tbody></table>`
    : `<p class="hint">등록된 정책이 없다.</p>`);
}

// 화살표는 상자가 자리를 잡은 뒤에 잰다. 다음 노드 = 오른쪽 변 → 왼쪽 변(실선), 되돌아가기 = 아래로 돌아 앞 상자 밑으로(주황),
// 같은 노드 다시 = 상자 위 고리(주황). 쓴 피드백은 진하게, 지금 하는 복구는 흐르게.
const EP_ARROW = ["ok", "live", "mute", "warn"].map((k) => `<marker id="epf-ah-${k}" viewBox="0 0 10 10" refX="9" refY="5"
  markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="epf-ah-${k}"/></marker>`).join("");
function drawEpWires() {
  const e = lastEpFlow, root = $("epf"), svg = $("epf-wires");
  if (!e || root.offsetParent === null) return;
  // 위 여백 = 다시 고리 자리, 아래 여백 = 되돌아가기 곡선 수만큼. 재기 전에 정해 둔다(정한 뒤 잰 자리에 그린다)
  const fbs = e.flow.feedback || [], nBack = fbs.filter((f) => f.kind !== "retry").length;
  const padT = `${fbs.some((f) => f.kind === "retry") ? 46 : 12}px`, padB = `${nBack ? Math.round(0.75 * (44 + 22 * (nBack - 1)) + 30) : 14}px`;
  if (root.style.paddingTop !== padT) root.style.paddingTop = padT;
  if (root.style.paddingBottom !== padB) root.style.paddingBottom = padB;
  const r0 = root.getBoundingClientRect(), sx = root.scrollLeft;
  const box = (id) => { const el = document.getElementById(`epf-box-${id}`); return el && el.getBoundingClientRect(); };
  const nodes = e.flow.nodes, out = [];
  for (let i = 0; i + 1 < nodes.length; i++) {
    const a = box(nodes[i].id), b = box(nodes[i + 1].id);
    if (!a || !b) continue;
    const y = a.top - r0.top + a.height / 2, x1 = a.right - r0.left + sx + 2, x2 = b.left - r0.left + sx - 3;
    const sa = epNodeState(e, nodes[i], i), sb = epNodeState(e, nodes[i + 1], i + 1);
    const cls = sb === "running" ? "active" : sa === "done" && sb !== "pending" ? "done" : sa === "done" ? "ready" : "";
    const ah = cls === "active" ? "live" : cls === "done" ? "ok" : "mute";
    out.push(`<path class="ep-wire ${cls}" d="M${x1.toFixed(1)},${y.toFixed(1)} L${x2.toFixed(1)},${y.toFixed(1)}" marker-end="url(#epf-ah-${ah})"/>`);
  }
  let below = 0;
  for (const f of e.flow.feedback || []) {
    const a = box(f.from), b = box(f.to);
    if (!a || !b) continue;
    const r = (e.runs || {})[f.from] || {}, used = (e.attempts || {})[f.from] || 0;
    const mine = r.recovery === (f.kind === "rollback" ? "rollback_retry" : "retry");
    const active = r.state === "recovering" && mine, cls = `ep-fb${used && mine ? " used" : ""}${active ? " active" : ""}`;
    const count = used && mine ? `${used}/${f.max} 사용` : `최대 ${f.max}번`;
    if (f.kind === "retry") {
      const yT = a.top - r0.top, xr = a.right - r0.left + sx - 18, xl = a.left - r0.left + sx + 18;
      out.push(`<path class="${cls}" d="M${xr.toFixed(1)},${yT.toFixed(1)} C${xr.toFixed(1)},${(yT - 30).toFixed(1)} ${xl.toFixed(1)},${(yT - 30).toFixed(1)} ${xl.toFixed(1)},${(yT - 1).toFixed(1)}" marker-end="url(#epf-ah-warn)"/>`);
      out.push(`<text class="ep-fb-label${used && mine ? " used" : ""}" x="${((xr + xl) / 2).toFixed(1)}" y="${(yT - 26).toFixed(1)}" text-anchor="middle">↻ 인지 갱신 후 다시 · ${count}</text>`);
      continue;
    }
    const h = 44 + 22 * below++;
    const yA = a.bottom - r0.top, yB = b.bottom - r0.top, xa = a.left - r0.left + sx + a.width / 2, xb = b.left - r0.left + sx + b.width / 2;
    out.push(`<path class="${cls}" d="M${xa.toFixed(1)},${yA.toFixed(1)} C${xa.toFixed(1)},${(yA + h).toFixed(1)} ${xb.toFixed(1)},${(yB + h).toFixed(1)} ${xb.toFixed(1)},${(yB + 2).toFixed(1)}" marker-end="url(#epf-ah-warn)"/>`);
    out.push(`<text class="ep-fb-label${used && mine ? " used" : ""}" x="${((xa + xb) / 2).toFixed(1)}" y="${(Math.max(yA, yB) + h * 0.75 + 15).toFixed(1)}" text-anchor="middle">↺ ${esc(f.label)} · ${count}</text>`);
  }
  const paths = `<defs>${EP_ARROW}</defs>` + out.join("");
  svg.setAttribute("width", root.scrollWidth);
  svg.setAttribute("height", root.scrollHeight);
  if (paths !== lastEpWires) { lastEpWires = paths; svg.innerHTML = paths; }
}

function episodeModal(action, want) {
  const e = S && S.session ? shownRunner(S.session) : null;
  return prepareAction("episode", { episode_action: action, want, ...(e && e.runner ? { runner: e.runner } : {}) });
}
function shownRunnerName() { const e = S && S.session ? shownRunner(S.session) : null; return (e && e.runner) || ""; }

function renderPolicy(s) {
  const p = s.policy;
  $("policy-panel").hidden = !p;
  if (!p) return;
  const c = p.card || {};
  put("policy-meta", `<span class="badge ${p.issues.length ? "bad" : p.status === "hold" ? "warn" : "ok"}">${esc(p.status)}</span>`);
  const issues = p.issues.length ? `<ul class="reasons bad" style="margin:0 12px 10px">${p.issues.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  put("policy", `<dl class="kv"><dt>id</dt><dd>${esc(p.id)}</dd><dt>task</dt><dd>${esc(c.task)}</dd><dt>side</dt><dd>${esc(c.side)}</dd>
    <dt>체크포인트</dt><dd>${esc(p.checkpoint || "—")}</dd><dt>계약</dt><dd>${esc(p.contract || "—")}</dd></dl>${issues}${c.note ? `<div class="note">${esc(c.note)}</div>` : ""}`);
}

// 로봇 상태 — 자산에서 뽑은 실루엣(링크마다 id) 옆에 관절 표. 판정·채널은 서버(robot_view.py)가 준다.
// 09.23 실기: 손가락이 계약 홈(굽힘 관절 하한 0.0)으로 밀려 꺾였는데 화면 어디에도 그 값이 없었다.
// 렌더(PNG, 실제 메쉬 음영) 위에 실루엣(SVG, 링크마다 id)을 겹친다 — 둘은 같은 투영·같은 창이라 맞아떨어진다.
//: 정적 파일은 /static/ 아래로만 서비스된다(server.py `_static`) — 상대 경로로 부르면 404 다.
// 그림은 로봇 모듈의 art(없으면 그리지 않는다) — DG-5F 그림을 RH56F1 에 덧칠하지 않는다
const ROBOT_ART_DEFAULT = { arms: "robot_arms", right: "robot_hand_right", left: "robot_hand_left" };
const robotArt = () => { const m = S && S.session && S.session.robot_module; return m ? (m.art || {}) : ROBOT_ART_DEFAULT; };
const artCache = {};

async function loadArt(name) {
  if (artCache[name] !== undefined) return artCache[name];
  artCache[name] = "";
  try {
    const r = await fetch(`/static/${name}.svg`);
    artCache[name] = r.ok ? await r.text() : "";
  } catch { artCache[name] = ""; }
  refresh();
  return artCache[name];
}

function jointRows(g, chan) {
  return g.rows.map((row) => {
    const v = row.vals[chan];
    const err = chan === "pos" ? row.err : null;
    return `<tr class="jr j-${esc(row.state)}"><td class="jn mono">${esc(row.joint.replace(/^[rl]_[ah][jl]_/, ""))}</td>
      <td class="jv mono">${v === null || v === undefined ? "—" : fmt(v, 3)}</td>
      <td class="je mono">${err === null ? "" : fmt(err, 3)}</td>
      <td class="jm">${row.state === "missing" ? "미수신" : row.stale ? "오래됨" : row.state === "limit" ? "끝점" : row.state === "off" ? "벗어남" : ""}</td></tr>`;
  }).join("");
}

// effort 칸은 팔과 손이 다른 양이다 — 팔(openarm)은 관절 토크 N·m, 손(DG-5F 드라이버)은 모터 전류 mA 를 넣는다.
// 09.28 사용자: 손 effort 10.000 을 토크로 읽고 "왜 이렇게 하드하게 들어가나" — 실제로는 10 mA 였다.
const EFF_AS = { arm: ["토크", "N·m"], hand: ["모터 전류", "mA"] };
const effAs = (g) => EFF_AS[g.title.endsWith("손") ? "hand" : "arm"];

function jointPanel(g, chan) {
  if (!g) return `<div class="panel jpanel"><div class="empty">—</div></div>`;
  const bad = g.rows.filter((r) => r.state === "limit").length;
  const unit = chan === "eff" ? ` · ${effAs(g)[0]} [${effAs(g)[1]}]` : "";
  return `<div class="panel jpanel"><div class="panel-head"><h2>${esc(g.title)}</h2>
      <span class="meta">${g.seen}/${g.total}${esc(unit)}${bad ? ` · <b class="bad">끝점 ${bad}</b>` : ""}</span></div>
    <table class="joints"><tbody>${jointRows(g, chan)}</tbody></table></div>`;
}

// 링크 id 로 칠한다: r_hj_index_2(관절) → r_hl_index_2(링크). 상태가 없는 링크는 칠하지 않아 렌더가 그대로 보인다.
const ART_TINT = { limit: "rgba(255,93,82,.45)", off: "rgba(227,160,8,.42)" };
const ART_EDGE = { limit: "#ff5d52", off: "#e3a008" };

function artWith(key, groups) {
  const name = robotArt()[key];
  if (!name) return "";
  const svg = artCache[name];
  if (svg === undefined) { loadArt(name); return `<div class="empty">그림 여는 중…</div>`; }
  const rules = [];
  groups.forEach((g) => g.rows.forEach((r) => {
    if (r.state === "ok" || r.state === "missing") return;
    const id = r.joint.replace(/_([ah])j_/, "_$1l_");
    rules.push(`#${id} { fill: ${ART_TINT[r.state]}; stroke: ${ART_EDGE[r.state]}; }`);
  }));
  const img = `<img src="/static/${name}.png" alt="" onerror="this.style.display='none'">`;
  return `${img}${svg ? `<style>${rules.join("\n")}</style>${svg}` : ""}`;
}

// 로봇 모듈 칸 — 어느 로봇 · 어느 정책(고른 것 · 미션 기본값)으로 열었는가, 로봇 설정(09.29 사용자: "robot setting 및 state")

function renderRobot(s) {
  const r = s.robot;
  if (!r || !r.groups.length) return put("robot", `<div class="empty">관절 상태가 아직 없다 — 드라이버가 떠야 보인다.</div>`);
  const chan = r.channels.some((c) => c.key === robotChan) ? robotChan : "pos";
  put("robot-chan", r.channels.map((c) =>
    `<button class="chip${c.key === chan ? " on" : ""}" data-act="robot-chan" data-arg="${esc(c.key)}">${c.key === "eff" ? "토크 · 전류" : esc(c.name)}<span class="hint"> ${c.key === "eff" ? "팔 N·m · 손 mA" : esc(c.unit)}</span></button>`).join(""));
  put("robot-meta", r.stale ? `<span class="warn">오래됨</span>` : `${fmt(r.age_s, 1)} s 전`);
  const by = (t) => r.groups.find((g) => g.title === t);
  const art = (k) => artWith(k, r.groups);
  put("robot", `<div class="rgrid">
      ${jointPanel(by("오른팔"), chan)}<div class="rart"><span class="stack">${art("arms")}</span></div>${jointPanel(by("왼팔"), chan)}
      ${jointPanel(by("오른손"), chan)}
      <div class="rart rhands"><figure><span class="stack">${art("right")}</span><figcaption>오른손</figcaption></figure>
        <figure><span class="stack">${art("left")}</span><figcaption>왼손</figcaption></figure></div>
      ${jointPanel(by("왼손"), chan)}
    </div>
    <div class="hint" style="padding:6px 12px 10px">${chan === "pos" ? "현재 · 목표와의 차이 [rad]" : chan === "eff" ? "팔 = 관절 토크 [N·m] · 손 = 모터 전류 [mA] (DG-5F 드라이버가 effort 칸에 전류를 넣는다)" : `현재 [${esc((r.channels.find((c) => c.key === chan) || {}).unit || "")}]`} · 색은 관절 상태(끝점 빨강 · 벗어남 노랑)</div>`);
}

// ── 모달 ────────────────────────────────────────────────────────────────
function modal(markup, real = false) {
  $("modal-card").className = "modal-card" + (real ? " real" : "");
  $("modal-card").innerHTML = markup;
  $("modal").hidden = false;
}
const closeModal = () => { pendingConfirmation = null; $("modal").hidden = true; };

function approveModal(stageId) { return prepareAction("stage", { stage: stageId }); }

function endModal() {
  const s = S.session, why = s.end_reasons, alive = s.procs.filter((p) => p.alive).length;
  modal(`<h3>run 끝내기</h3><p>띄워 둔 프로세스 ${alive}개를 그룹째 정지하고(SIGTERM → 5 s → SIGKILL) 브리지를 내린다. 기록은 <code>${esc(rel(s.run_dir))}</code> 에 남는다.</p>
    ${why.length ? `<ul class="reasons bad">${why.map((x) => `<li>${esc(x)}</li>`).join("")}</ul><label class="chk" style="margin-top:10px"><input type="checkbox" id="end-force"> 그래도 끝낸다 — pd 가 팔을 잡은 채로 프로세스가 죽는다</label>` : ""}
    <div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button><button class="btn ${why.length ? "btn-real" : "btn-primary"}" data-act="end-go">끝내기</button></div>`, why.length > 0);
}

// ── 동작 ────────────────────────────────────────────────────────────────
const acts = {
  "ep-next"(want) { if (want) episodeModal("next", want); },
  "ep-run"(want) { episodeModal("run", want); },
  async "ep-go"() { return acts["action-execute"](); },
  async "ep-stop"() { await call("POST", "/api/episode/stop", { runner: shownRunnerName() }); refresh(); },
  async "ep-reset"() { await call("POST", "/api/episode", { action: "reset", runner: shownRunnerName() }); refresh(); },
  "ep-runner"(name) { epRunner = name; refresh(); },
  async "lease-take"() {
    const name = ($("op-name")?.value || "").trim();
    if (!name) return toast("운영자 이름을 넣을 것");
    me.name = name;
    me.token = (await call("POST", "/api/lease", { operator: name })).token;
    refresh();
  },
  async "lease-force"() {
    const name = (prompt(`${S.lease.holder} 가 조작 중이다. 가져오면 그쪽 화면의 버튼이 죽는다.\n운영자 이름:`, me.name) || "").trim();
    if (!name) return;
    me.name = name;
    me.token = (await call("POST", "/api/lease", { operator: name, force: true })).token;
    refresh();
  },
  async "lease-drop"() { await call("DELETE", "/api/lease"); me.token = ""; refresh(); },
  async open(id) { await call("POST", "/api/run/open", { profile: id }); refresh(); },
  "pick-robot"(id) { if (pick.robot !== id) { pick.robot = id; pick.policies = {}; } refresh(); },
  async "open-picked"(id) {
    const policies = Object.fromEntries(Object.entries(pick.policies).filter(([, v]) => v));
    await call("POST", "/api/run/open", { profile: id, policies });
    refresh();
  },
  approve(id) { approveModal(id); },
  async "approve-go"() { return acts["action-execute"](); },
  "robot-chan"(key) { robotChan = key; refresh(); },
  async run(id) { return prepareAction("stage", { stage: id }); },
  async restart(id) { return prepareAction("stage", { stage: id, restart: true }); },
  async ack(arg) {
    const [stage, i, ok] = arg.split(":");
    await call("POST", "/api/stage/ack", { stage, index: Number(i), ok: ok === "1" });
    refresh();
  },
  async "abort-stage"(id) { await call("POST", "/api/stage/abort", { stage: id || "" }); refresh(); },
  async quick(arg) {
    const [name, side] = String(arg).split(":");                           // pd 서비스는 팔마다 따로다(09.23)
    await call("POST", `/api/quick/${name}${side ? `/${side}` : ""}`);
    toast("요청을 보냈다 — 결과는 '사건' 에 뜬다", [], true);
    refresh();
  },
  async unit(arg) {
    const [key, on] = [arg.slice(0, arg.lastIndexOf(":")), arg.endsWith(":1")];
    if (!on) return unitModal(key);                                        // 끄기는 한 번 더 묻는다
    await call("POST", "/api/unit", { key, on: true });
    refresh();
  },
  async "unit-off-go"(key) { await call("POST", "/api/unit", { key, on: false }); closeModal(); refresh(); },
  log(key) { logKey = key; $("log-title").textContent = `로그 · ${key}`; $("logdrawer").hidden = false; pollLog(); },
  "end-run"() { endModal(); },
  async "end-go"() { await call("POST", "/api/run/end", { force: !!$("end-force")?.checked }); closeModal(); $("logdrawer").hidden = true; logKey = null; refresh(); },
  "modal-close"() { closeModal(); },
  async copy(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {                                                                // 클립보드 API 가 막힌 브라우저
      const t = Object.assign(document.createElement("textarea"), { value: text });
      document.body.append(t);
      t.select();
      const ok = document.execCommand("copy");
      t.remove();
      if (!ok) return toast("복사하지 못했다 — 명령을 직접 선택해 복사할 것");
    }
    toast("복사했다 — 다른 셸에 붙여넣어 실행할 것", [], true);
  },
  "layout-reset"() { layout = {}; saveLayout(); applyLayout(); drawWires(); },
  "goto-stage"(id) {                                                         // 잠금 줄 · 목록 → 지금 단계면 조작판, 아니면 목록의 그 줄
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    // 창이 여럿이면 "지금 단계"도 여럿이다 — 그 id 가 제 창의 지금 단계면 조작판으로 간다.
    const cur = S && S.session && S.session.mission.rows.find((r) => r.id === id && r.current);
    if (cur) {
      const cp = $("control");
      cp.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "start" });
      cp.classList.remove("flash"); void cp.offsetWidth; cp.classList.add("flash");
      cp.querySelector('[data-act="ack"],[data-act="approve"]:not([disabled]),[data-act="run"]')?.focus({ preventScroll: true });
      return;
    }
    $("all-stages").open = true;
    const li = $(`stage-${id}`);
    if (!li) return;
    li.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "center" });
    flashStage = id; flashUntil = Date.now() + 2000;
    li.classList.remove("flash"); void li.offsetWidth; li.classList.add("flash");
  },
  "show-all"() {
    const d = $("all-stages");
    d.open = true;
    d.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  },
  skip(id) {
    const r = S.session.mission.rows.find((x) => x.id === id);
    modal(`<h3>건너뛰기 — ${esc(id)}</h3><p>${esc(r ? r.title : "")}</p>
      <p>이 단계를 <b>실행하지 않고</b> 완료로 친다. 되돌릴 수 없다 — 다시 하려면 새 run 을 연다.</p>
      <div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button>
      <button class="btn btn-primary" data-act="skip-go" data-arg="${esc(id)}">건너뛴다</button></div>`);
  },
  rewind(id) {
    modal(`<h3>되돌아가기 — ${esc(id)}</h3><p><b>${esc(id)}</b> 단계부터 다시 진행한다. 그 뒤에 끝낸 단계들은 다시 해야 한다.</p>
      <p>떠 있는 프로세스는 그대로 둔다(다시 실행하면 "이미 떠 있음" 으로 넘어간다). 실기 단계는 승인을 다시 받는다.</p>
      <div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button>
      <button class="btn btn-primary" data-act="rewind-go" data-arg="${esc(id)}">되돌아간다</button></div>`);
  },
  async "rewind-go"(id) { await call("POST", "/api/stage/rewind", { stage: id }); closeModal(); refresh(); },
  async "skip-go"(id) { await call("POST", "/api/stage/skip", { stage: id }); closeModal(); refresh(); },
};

// Workspace selections are server state. Polling never starts a policy or replaces the video element.
// 10.17 주소 #policy · #control · #setting 이면 그 탭으로 연다(정책 탭 바로가기 · 화면 확인 도구)
let activeTab = ["setting", "control", "policy"].find((t) => globalThis.location?.hash === `#${t}`) || "setting",
  pendingConfirmation = null, executingConfirmation = false;
function selectTab(tab) {
  if (!["setting", "control", "policy"].includes(tab)) return;
  // 탭을 바꾸면 아래 칸 높이가 바뀐다 — 탭 줄이 화면에서 제자리에 있게 스크롤을 맞춘다(10.15 "control 누르면 스크롤이 올라감")
  const nav = document.querySelector?.(".workspace-tabs");
  const y0 = nav && !nav.closest("[hidden]") ? nav.getBoundingClientRect().top : null;
  const changed = activeTab !== tab;
  activeTab = tab;
  for (const key of ["setting", "control", "policy"]) {
    $("tab-" + key).hidden = key !== tab;
    $("tab-button-" + key).setAttribute?.("aria-selected", String(key === tab));
    $("tab-button-" + key).classList?.toggle("on", key === tab);
  }
  if (changed && y0 !== null) {
    const d = nav.getBoundingClientRect().top - y0;
    if (Math.abs(d) >= 1) window.scrollBy(0, d);
  }
  if (S?.session) { drawEpWires(); drawWires(); }
}
// 작업마다 따로 돈다(10.15) — 버튼 아래에 그 작업의 실행 상태와, 그 작업이 띄운 것이 지금 살아 있는지를 적는다
function actionState(a) {
  const j = a.job, live = a.live || {}, out = [];
  if (j?.active) out.push(`<span class="badge live">⟳ 실행 중${j.stage ? ` · ${esc(j.stage)}` : ""}</span>`);
  else if (j?.outcome) out.push(`<span class="badge ${j.outcome === "DONE" ? "ok" : "bad"}">${j.outcome === "DONE" ? "✓ 완료" : "✗ 실패"}</span>`);
  if ((live.alive || []).length) out.push(`<span class="badge ok" title="${esc(live.alive.join("\n"))}">● 켜짐 ${live.alive.length}</span>`);
  else if (j?.outcome === "DONE" && a.group === "setting" && a.id !== "preflight" && a.id !== "shutdown") out.push(`<span class="badge mute">○ 띄운 프로세스 없음</span>`);
  if ((live.dead || []).length) out.push(`<span class="badge bad" title="${esc(live.dead.join("\n"))}">✗ 죽음 ${live.dead.length}</span>`);
  if (live.pd) out.push(`<span class="badge ${live.pd === "TRACKING" ? "ok" : live.pd === "IDLE" ? "mute" : "warn"}">PD ${esc(live.pd)}${live.pd_err != null ? ` · 오차 ${fmt(live.pd_err, 3)}${live.pd_settled ? " 정착" : ""}` : ""}</span>`);
  const note = !j?.active && j?.outcome === "FAILED" && j.note ? `<small class="bad">${esc(j.note)}</small>` : "";
  return out.length || note ? `<div class="action-state">${out.join("")}</div>${note}` : "";
}
// ★10.19 경로 C 카드 — 스크립트 잡기 단계(접근 정책 → 닫기 → 들기 → 유지)와 이 물체에서 마지막으로 쥔 관절(도)
const GRIP_PHASE = { approach: "접근 정책", close: "스크립트 닫기", lift: "들기", settle: "유지", done: "성공", failed: "실패" };
function scriptGrip(a) {
  const sc = a.live?.script;
  if (!sc) return "";
  const now = sc.phase ? `<span class="badge ${sc.phase === "failed" ? "bad" : sc.phase === "done" ? "ok" : "warn"}">${esc(GRIP_PHASE[sc.phase] || sc.phase)}</span> ${esc(sc.why || "")}` : '<span class="badge mute">대기</span>';
  const g = sc.last_grip;
  const grip = g ? `<p class="hint">마지막 쥔 관절(${esc(g.outcome || "진행 중")}${g.rise_m != null ? ` · 들기 ${fmt(g.rise_m * 100, 1)} cm` : ""}): ${(g.hand_joints || []).map((n, i) => `${esc(String(n).replace(/^[rl]_hj_/, ""))} ${fmt((g.q_meas_rad?.[i] ?? NaN) * 180 / Math.PI, 0)}°`).join(" · ")}</p>` : '<p class="hint">이 물체로 쥔 기록 없음 — 쥐면 logs/grasp_c/grips.jsonl 에 남는다</p>';
  return `<div class="script-grip"><p class="hint">스크립트 잡기: ${esc(sc.steps)}</p><p>${now}</p>${grip}</div>`;
}
function actionButton(a) {
  const reasons = a.reasons || [];
  const can = holding() && a.enabled;
  return `<div class="workspace-action"><button class="btn" data-act="action-prepare" data-arg="${esc(a.id)}" ${can ? "" : "disabled"} title="${esc([a.title, ...reasons].filter(Boolean).join(" · "))}"><b>${esc(a.id)}</b>${a.title && a.title !== a.id ? `<span class="action-desc">${esc(a.title)}</span>` : ""}</button>${actionState(a)}${!can && !a.job?.active ? `<small>${esc(reasons.join(" · ") || "조작 권한 필요")}</small>` : ""}</div>`;
}
async function prepareAction(action, extra = {}) {
  const prepared = await call("POST", "/api/actions/prepare", { action, ...extra });
  pendingConfirmation = prepared.confirmation_id;
  if (extra.restart) prepared.warnings = [...(prepared.warnings || []), ...liveUnits(extra.stage).map(u => `정지 후 다시 연결: ${u.note || u.key} (pid ${u.pid})`)];
  const commands = (prepared.commands || []).map(c => `<li><pre>${esc(typeof c === "string" ? c : c.argv ? shellLine(c.argv) : JSON.stringify(c))}</pre></li>`).join("");
  modal(`<h3>${esc(prepared.title || action)}</h3><p>표시된 명령과 설정을 한 번 실행한다. 설정 변경 시 승인은 무효다.</p><ul>${commands}</ul><ul class="reasons warn">${(prepared.warnings || []).map(w => `<li>${esc(w)}</li>`).join("")}</ul><div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button><button class="btn btn-real" id="action-execute" data-act="action-execute">승인 후 실행</button></div>`, S?.session?.profile.domain_class === "real");
}
// policy 탭의 인지 노드 칸 — 인지(FP++ · locate · 좌표 변환 · 홀더)만. 에피소드 실행기 · 정책 노드는 싣지 않는다(10.15 사용자)
function perceptionNode(b) {
  const name = [b.id, b.key, b.kind].join(" ");
  return /fpp|locate|perception|object_pose|cup_holder/i.test(name) && !/episode|policy|aglt|place|pour|fabric|\bobs\b/i.test(name);
}
function workspaceNode(b) { return /policy|정책|입력|fabric|\bobs\b|fpp|locate|perception|object_pose|cup_holder|episode_runner/i.test([b.id, b.key, b.title, b.kind].join(" ")); }
function baseDiagram(diagram) {
  if (!diagram) return diagram;
  const cols = diagram.cols.map(col => col.filter(b => !workspaceNode(b))).filter(col => col.length);
  const ids = new Set(cols.flat().map(b => b.id));
  return { ...diagram, cols, wires: (diagram.wires || []).filter(e => ids.has(e.from) && ids.has(e.to)) };
}
// 좌표는 소수 넷째 자리(0.1 mm)까지 — 10.15 사용자 "좌표는 0.xxxx 로"
function coord4(v) {
  if (Array.isArray(v)) return `[${v.map(x => typeof x === "number" ? x.toFixed(4) : String(x)).join(", ")}]`;
  if (v && typeof v === "object") return JSON.stringify(v, (k, x) => typeof x === "number" ? Number(x.toFixed(4)) : x);
  return v == null ? "—" : String(v);
}
// ★10.21 인지 상세의 물체 줄 — 좌표 한 칸(좌표계 · 취득 시각은 올리면 보인다), 무효면 이유
function coordShort(result, source) {
  if (!result) return `<span class="coord missing"><small>${esc(source)}</small>미수신</span>`;
  const ok = result.valid === true;
  const when = `좌표계 ${result.frame || result.frame_id || "—"} · 취득 ${result.acquired_at || result.timestamp || result.stamp || "—"} · 출처 ${result.source || source}`;
  return `<span class="coord ${ok ? "ok" : "bad"}" title="${esc(when)}"><small>${esc(source)}</small>${ok ? esc(coord4(result.position ?? result.xyz ?? result.pose ?? null)) : esc(result.error || result.reason || "무효")}</span>`;
}
// ★10.21 policy 탭 인지 노드 — 상자 머리 · 상태 줄 · 떠 있는 유닛만(띄우는 단계 수십 줄은 아래 연결 그림에 있다)
function boxCompact(b) {
  const us = boxUnits(b), alive = us.filter((u) => u.alive), show = alive.length ? alive : us.slice(0, 1);
  const lines = (b.lines || []).map((l) => `<div class="dg-line tone-${esc(l.tone)}">${esc(l.text)}</div>`).join("");
  const rest = us.length - show.length;
  return `<div class="pnode tone-${esc(b.tone)}"><div class="pnode-head"><span class="lamp ${esc(b.tone)}"></span><b>${esc(b.title)}</b>
      <span class="dg-state">${esc(LINK_WORD[b.state] || b.state)}</span>${b.host ? `<span class="dg-host">${esc(b.host)}</span>` : ""}</div>
    ${b.detail ? `<div class="dg-detail">${esc(b.detail)}</div>` : ""}${lines}${show.map((u, i) => unitRow(b, u, true, i === 0)).join("")}
    ${rest > 0 ? `<div class="hint pnode-more">띄우는 단계 ${us.length} 개 중 ${show.length} 개 — 나머지는 아래 연결 그림</div>` : ""}</div>`;
}
// 관리 작업 진행 상황 — 원시 JSON 대신 작업 · 단계 · 결과 한 줄
function managementStatus(m, group) {
  if (!m || (!m.active && !m.outcome)) return "";
  if (m.jobs) {
    const jobs = m.jobs.filter(j => j.active && (!group || j.group === group));
    return jobs.map(j => managementStatus({ ...j, jobs: undefined })).join("");
  }
  const head = m.active ? `실행 중 · ${esc(m.id || "")}${m.stage ? ` · ${esc(m.stage)}` : ""}` : `마지막 작업 ${esc(m.id || "")} · ${esc(m.outcome)}${m.note ? ` — ${esc(m.note)}` : ""}`;
  const steps = (m.steps || []).map(st => `<li>${esc(st.note || st.key || "")} · ${esc(stepLabel(st))}${st.detail ? ` — ${esc(st.detail)}` : ""}</li>`).join("");
  return `<div class="management-status"><b>${head}</b>${steps ? `<ul>${steps}</ul>` : ""}</div>`;
}
// 목 제어 — camera 세션이 보정한 각도로만 간다(10.15 사용자: "camera 세션에 만든 특정 각도로 가야 locate 사용 가능").
// 보정 프로필을 고르면 그 프로필의 각도표가 버튼으로 뜬다. 버튼 = 그 각도 그대로(pan · tilt), [이 각도로 이동] = head_control 승인 창.
const HEAD_PROFILE_PREFERRED = "v9_1014_anchored_z";   // camera 세션 권장(10.14)
let headDraft = null;
function headDefaults(h, profiles) {
  const names = Object.keys(profiles || {});
  const profile = h.calib_profile || (names.includes(HEAD_PROFILE_PREFERRED) ? HEAD_PROFILE_PREFERRED : names[0] || "");
  return { pan: h.pan ?? 0, tilt: h.tilt ?? -20, calib_profile: profile };
}
function headModule(h, action) {
  const profiles = action?.calib_profiles || {};
  const d = headDraft || headDefaults(h, profiles);
  const prof = profiles[d.calib_profile];
  const can = holding() && action && action.enabled;
  const why = action ? (action.reasons || []).join(" · ") : "이 로봇에 head_control 단계가 없다";
  const tilts = prof && !prof.error ? [...new Set(prof.angles.map(a => Math.round(a.tilt)))].sort((a, b) => b - a) : [];
  const hit = (a) => prof && Math.abs(a.pan - Number(d.pan)) <= prof.tolerance && Math.abs(a.tilt - Number(d.tilt)) <= prof.tolerance;
  const rows = tilts.map(t => `<div class="head-row"><span class="head-tilt">tilt ${t}°</span>${prof.angles.filter(a => Math.round(a.tilt) === t).sort((a, b) => a.pan - b.pan)
    .map(a => `<button class="btn btn-sm${hit(a) ? " on" : ""}" data-act="head-preset" data-arg="${a.pan}:${a.tilt}" title="볼트 ${a.n_bolts}개 · 잔차 ${a.resid_mm ?? "—"} mm">pan ${Math.round(a.pan)}</button>`).join("")}</div>`).join("");
  const onTable = !prof || prof.error || !prof.strict || prof.angles.some(hit);
  return `<fieldset class="head-module"><legend>목 제어 — 지금 운용 자세 pan ${esc(h.pan ?? "—")}° · tilt ${esc(h.tilt ?? "—")}° · ${esc(h.calib_profile || "보정 미선택")}</legend>
    <label>보정 프로필 <select data-head-draft="calib_profile">${Object.keys(profiles).map(n => `<option value="${esc(n)}" ${n === d.calib_profile ? "selected" : ""}>${esc(n)}${n === HEAD_PROFILE_PREFERRED ? " (camera 권장)" : ""}</option>`).join("")}</select></label>
    ${prof?.error ? `<small class="bad">${esc(prof.error)}</small>` : ""}
    <div class="head-angles">${rows || '<small>이 프로필에 보정 각도가 없다</small>'}</div>
    <span class="head-target">목표 pan ${fmt(Number(d.pan), 3)}° · tilt ${fmt(Number(d.tilt), 3)}°${onTable ? "" : ' <b class="bad">— 보정 각도 아님(locate · FP++ 좌표 불가)</b>'}</span>
    <details class="head-free"><summary>직접 입력</summary>
      <label>pan [°] <input type="number" step="0.1" min="-90" max="90" data-head-draft="pan" value="${esc(d.pan)}"></label>
      <label>tilt [°] <input type="number" step="0.1" min="-60" max="-10" data-head-draft="tilt" value="${esc(d.tilt)}"></label></details>
    <button class="btn btn-real" data-act="head-move" ${can ? "" : "disabled"}>이 각도로 이동 (head_control)</button>
    <small>보정 각도(±${esc(prof?.tolerance ?? 1)}°)로 이동 → 실측 정착 확인 → 카메라 좌표 변환 노드를 이 프로필로 다시 띄운다. 그 뒤 인지 재추출.</small>
    ${action ? actionState(action) : ""}${!can && why ? `<small>${esc(why)}</small>` : ""}</fieldset>`;
}
function renderWorkspace() {
  const w = S.workspace || {};
  const actions = S.actions || [];
  put("common-controls", actions.filter(a => a.group === "control").map(actionButton).join(""));
  for (const group of ["setting", "control"]) put(group + "-actions", actions.filter(a => a.group === group).map(actionButton).join(""));
  // ★10.21 제어 모듈 에피소드(ctl_*)는 policy 탭 실험 목록이 실행한다 — 여기서 따로 버튼을 늘어놓지 않는다
  put("management", managementStatus(S.management, "setting"));
  put("management-control", managementStatus(S.management, "control"));
  const h = w.head || {};
  const hs = $("head-settings");
  if (!(hs.contains && hs.contains(document.activeElement))) put("head-settings", headModule(h, actions.find(a => a.id === "head_control")));
  put("perception-controls", `<fieldset class="pctl"><legend>인지 · ${esc(h.calib_profile || "보정 미선택")} · pan ${esc(h.pan ?? "—")} / tilt ${esc(h.tilt ?? "—")}</legend>${["fpp", "locate"].map(source => `<label class="chk"><input type="checkbox" data-perception="${source}" ${w.perception?.[source] ? "checked" : ""} ${holding() ? "" : "disabled"}>${source === "fpp" ? "FPP" : "locate"}</label>`).join("")}${(w.objects || []).map(o => `<label class="chk"><input type="checkbox" data-object="${esc(o.id)}" ${o.enabled ? "checked" : ""} ${holding() ? "" : "disabled"}>${esc(o.model || o.id)} · ${esc(o.color || o.marker_id || "마커")}${o.measured === false ? " (프리셋 · 미측정)" : ""}</label>`).join("")}</fieldset>`);
  put("perception-nodes", (S.session?.diagram?.cols || []).flat().filter(perceptionNode).map(boxCompact).join(""));
  put("object-cards", (w.objects || []).filter(o => o.enabled).map(o => `<div class="obj-row"><b>${esc(o.model || o.id)} · ${esc(o.color || o.marker_id || "")}</b>${o.model === "holder" ? '<span class="hint">색 기반 locate 미지원</span>' : ""}<label>정책 입력 <select data-input-source="${esc(o.id)}" ${holding() ? "" : "disabled"}><option value="fpp" ${o.input_source === "fpp" ? "selected" : ""}>FPP</option><option value="locate" ${o.input_source === "locate" ? "selected" : ""} ${o.model === "holder" ? "disabled" : ""}>locate</option></select></label>${coordShort(o.results?.fpp, "FPP")}${coordShort(o.results?.locate, "locate")}<button class="btn btn-sm" data-act="object-perception" data-arg="${esc(o.id)}" ${holding() ? "" : "disabled"}>인지 · 좌표 재추출</button>${actions.filter(a => a.object_id === o.id).map(actionButton).join("")}</div>`).join("") || '<p class="hint obj-none">고른 물체가 없다 — 위에서 고른다</p>');
  selectTab(activeTab);
}
// 10.17 사용자 결정 3: 화면 · 에피소드 · 보고서만 tilt — 작업 공간 키(pouring)와 코드 모듈(pour_*.py)은 그대로
const TASK_LABEL = { pouring: "tilt" };
function taskLabel(task) { return TASK_LABEL[task] || task; }
async function saveWorkspace(patch) {
  pendingConfirmation = null;
  closeModal();
  await call("POST", "/api/workspace", patch);
  await refresh();
}
let videoAttached = false;
function updateVideo(s) {
  const img = $("fpp-video");
  if (!videoAttached) {
    videoAttached = true;
    img.addEventListener("error", () => { if (showVideo) { $("video-retry").hidden = false; $("video-status").textContent = "영상 연결 끊김 · 마지막 화면을 사용하지 말 것"; } });
    img.addEventListener("load", () => { $("video-retry").hidden = true; $("video-status").textContent = "영상 수신 중"; });
  }
  if (!img.getAttribute("src")) img.src = "/api/video";
  const v = s.video || S.video;
  if (v) $("video-retry").hidden = !(v.stale || v.connected === false);
  if (v) $("video-status").textContent = v.stale ? "오래된 영상 · 현재 화면 아님" : v.connected === false ? "영상 연결 끊김" : `영상 수신 중${v.age_s != null ? ` · ${fmt(v.age_s, 1)} s 전` : ""}`;
}
Object.assign(acts, {
  tab: selectTab,
  "video-retry"() { $("fpp-video").removeAttribute("src"); $("fpp-video").src = "/api/video"; $("video-status").textContent = "영상 다시 연결 중"; $("video-retry").hidden = true; },
  "action-prepare": prepareAction,
  async "action-execute"() {
    if (!pendingConfirmation || executingConfirmation) return;
    const confirmation_id = pendingConfirmation;
    pendingConfirmation = null;
    executingConfirmation = true;
    if ($("action-execute")) $("action-execute").disabled = true;
    try { await call("POST", "/api/actions/execute", { confirmation_id }); closeModal(); refresh(); }
    finally { executingConfirmation = false; }
  },
  "head-preset"(arg) {
    const [pan, tilt] = arg.split(":").map(Number), action = (S.actions || []).find(a => a.id === "head_control");
    headDraft = { ...(headDraft || headDefaults(S.workspace?.head || {}, action?.calib_profiles)), pan, tilt };
    html["head-settings"] = null; renderWorkspace();
  },
  "head-move"() {
    const action = (S.actions || []).find(a => a.id === "head_control");
    const d = headDraft || headDefaults(S.workspace?.head || {}, action?.calib_profiles);
    const extra = { pan: Number(d.pan), tilt: Number(d.tilt) };
    if (d.calib_profile) extra.calib_profile = d.calib_profile;
    return prepareAction("head_control", extra);
  },
  "object-perception"(id) { return prepareAction("perception_refresh", {object_ids:[id]}); },
  "xp-pick"(id) { xpSel = { ...xpSel, id }; xpPol = ""; xpSave(); xpSyncRunner(); render(); },
  "xp-side"(side) { xpSel = { ...xpSel, side }; xpSave(); xpSyncRunner(); render(); },
  "xp-pol"(id) { xpPol = xpPol === id ? "" : id; render(); },
  "xp-video"() {
    showVideo = !showVideo;
    if (!showVideo) $("fpp-video").removeAttribute("src");   // 닫으면 스트림을 끊는다(열 때 다시 붙인다)
    render();
  },
});
document.addEventListener("change", e => {
  const t = e.target, d = t.dataset || {};
  let patch;
  if (d.perception) patch = {perception:{...S.workspace.perception,[d.perception]:t.checked}};
  if (d.object || d.inputSource) patch = {objects:S.workspace.objects.map(o => ({id:o.id,enabled:d.object === o.id ? t.checked : o.enabled,input_source:d.inputSource === o.id ? t.value : o.input_source}))};
  if (d.headDraft) {
    const action = (S.actions || []).find(a => a.id === "head_control");
    headDraft = { ...(headDraft || headDefaults(S.workspace.head || {}, action?.calib_profiles)), [d.headDraft]: d.headDraft === "calib_profile" ? t.value : Number(t.value) };
    if (d.headDraft === "calib_profile") { html["head-settings"] = null; t.blur(); renderWorkspace(); }
    return;
  }
  if (patch) saveWorkspace(patch).catch(() => refresh());
});


document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-act]");
  if (!b || b.disabled) return;
  const fn = acts[b.dataset.act];
  if (fn) Promise.resolve(fn(b.dataset.arg)).catch(() => {});
});
document.addEventListener("change", (e) => {
  const side = e.target.dataset && e.target.dataset.pick;
  if (side) { pick.policies[side] = e.target.value; render(); }
});
document.addEventListener("toggle", (e) => {
  const k = e.target.dataset && e.target.dataset.key;
  if (k) e.target.open ? openKeys.add(k) : openKeys.delete(k);
}, true);
$("btn-end").dataset.act = "end-run";
$("log-close").addEventListener("click", () => { $("logdrawer").hidden = true; logKey = null; });
$("modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeModal(); });

async function pollLog() {
  if (!logKey) return;
  try {
    const r = await fetch(`/api/log?key=${encodeURIComponent(logKey)}`);
    const j = await r.json();
    const pre = $("log-body");
    pre.textContent = j.ok ? (j.text || "(비어 있다)") : j.error;
    if ($("log-follow").checked) pre.scrollTop = pre.scrollHeight;
  } catch (e) { /* 끊김은 stale 덮개가 알린다 */ }
}

// ── 흐름 ────────────────────────────────────────────────────────────────
function accept(snapshot) {
  S = snapshot;
  // 서버는 workspace · actions · management 를 run(session) 아래에 싣는다 — 관리 탭 코드는 S.<key> 로 읽는다
  for (const key of ["workspace", "actions", "management"]) S[key] = S.session?.[key] ?? null;
  if (me.token && S.lease.holder !== me.name) me.token = "";   // 만료됐거나 남이 가져갔다
  try {
    render();
    drawError = null;
    lastAt = Date.now();      // 그리기에 성공한 뒤에만 "신선하다"고 친다
  } catch (e) {
    // 반쯤 그린 화면이 살아 있는 값처럼 보이면 안 된다 — 조용히 삼키지 않고 덮는다
    drawError = e;
    console.error(e);
    cover();
  }
}
// 덮개: ① 그리다 실패 ② 아직 한 번도 못 받음 ③ 받다가 끊김. 셋 다 "이 화면을 믿지 말 것".
function cover() {
  const now = Date.now();
  let title = null, why = "";
  if (drawError) {
    title = "화면을 그리지 못했다";
    why = `콘솔 버그다 — 보이는 값은 멈춘 것이다. 정지는 물리 버튼 또는 CLI(episode_ctl / trigger.py). [${String(drawError && drawError.message || drawError)}]`;
  } else if (!lastAt) {
    if (now - bootAt > 2500) { title = "콘솔 서버에서 아직 아무것도 받지 못했다"; why = "서버가 떠 있는지, ssh 터널이 살아 있는지 볼 것."; }
  } else if (now - lastAt >= 2500) {
    title = "콘솔 서버와 끊겼다";
    why = `화면의 값은 ${Math.round((now - lastAt) / 1000)} s 전에 받은 것이다 — 지금의 로봇 상태가 아니다.`;
  }
  $("stale").hidden = !title;
  if (title) { $("stale-title").textContent = title; $("stale-why").textContent = why; }
}
async function refresh() {
  try { accept(await (await fetch("/api/state")).json()); } catch (e) { /* stale 덮개 */ }
}
function connect() {
  const es = new EventSource("/api/stream");
  es.addEventListener("state", (e) => accept(JSON.parse(e.data)));
  es.onerror = () => { es.close(); setTimeout(connect, 1500); };
}
setInterval(cover, 500);
setInterval(() => { if (holding()) call("POST", "/api/lease/renew").catch(() => { me.token = ""; }); }, 10000);
setInterval(pollLog, 1000);
window.addEventListener("resize", () => { if (S && S.session) drawCharts(S.session.metrics); drawWires(); drawEpWires(); });
if (window.ResizeObserver) new ResizeObserver(() => drawWires()).observe($("diagram"));   // 글꼴·표 펼침으로 상자가 밀려도 전선이 따라온다
if (window.ResizeObserver) new ResizeObserver(() => drawEpWires()).observe($("epf"));   // 에피소드 상자가 늘어나도 화살표가 따라온다
refresh().then(connect);
