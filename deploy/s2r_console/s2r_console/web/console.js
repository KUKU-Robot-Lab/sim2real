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
function render() {
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
  renderPolicy(s);
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
  const done = rows.filter((r) => r.done).length;
  const chips = rows.map((r) => `<span class="gc${r.id === lane.next ? " now" : r.done ? " done" : ""}" title="${esc(r.title)}">${r.done ? "✓ " : r.id === lane.next ? "● " : ""}${esc(r.id)}</span>`).join("");
  let body;
  if (!lane.next) body = `<p class="cp-note">이 창의 단계를 모두 끝냈다.</p>`;
  else {
    const cur = rows.find((r) => r.id === lane.next);
    const last = lane.last && lane.last.stage === cur.id ? lane.last : null;
    body = stageCard(s, { ...cur, last_outcome: last ? last.outcome : "", last_note: last ? last.note : "" },
                     s.runners ? s.runners[lane.id] : null, can);
  }
  return `<section class="panel lane${lane.busy ? " lane-busy" : ""}">
    <div class="panel-head"><h2>${esc(lane.title)}</h2><span class="meta">${done}/${rows.length}${lane.busy ? ` · ${esc(lane.busy)} 실행 중` : ""}</span></div>
    <div class="cp-chips">${chips}</div>${body}</section>`;
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
    return put("banner", `<div class="banner-state">대기</div><div class="banner-reasons"><span class="quiet">열린 run 이 없다 — 아래에서 프로파일을 고를 것.</span></div>`);
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
  const ep = s.episode ? `#${esc(s.episode.episode)} ${esc(s.episode.event)}` : "—";
  const m = s.mission;
  put("banner", `<div class="banner-state">${esc(b.state)}</div><div class="banner-reasons">${reasons}</div>
    <div class="pills">${pArmed}${pBridge}
      <div class="pill"><small>에피소드</small><b>${ep}</b></div>
      <div class="pill"><small>미션 · 사이클 ${esc(m.cycle)}</small><b>${esc(m.stage)} ${esc(m.status)}</b></div></div>`);
}

function renderLanding() {
  const can = holding();
  const cards = S.profiles.map((p) => {
    const real = p.domain_class === "real";
    return `<div class="card ${real ? "real" : "fake"}"><div>${domainBadge(p)}</div><h3>${esc(p.title)}</h3>
      <div class="path">미션 ${esc(rel(p.mission))}<br>노드 ${p.status_nodes.map(esc).join(" · ")}${p.policy_dir ? `<br>정책 ${esc(rel(p.policy_dir))}` : ""}</div>
      <div class="actions"><button class="btn ${real ? "btn-real" : "btn-primary"}" data-act="open" data-arg="${esc(p.id)}" ${can ? "" : "disabled"}>run 시작</button>
      ${can ? "" : `<span class="hint">먼저 오른쪽 위에서 조작 권한을 잡을 것</span>`}</div></div>`;
  });
  const broken = Object.entries(S.bad_profiles).map(([name, why]) =>
    `<div class="card broken"><div><span class="badge warn">읽지 못함</span></div><h3>${esc(name)}</h3><div class="path">${esc(why)}</div></div>`);
  put("landing", `<h1>배포 프로파일</h1><p class="lead">프로파일은 미션 · 정책 · DDS 도메인을 한 묶음으로 고정한다. 하나를 열면 그 도메인에 <b>구독 전용</b> 브리지가 붙는다 — 그것만으로는 아무 명령도 나가지 않는다.</p>
    <div class="cards">${cards.join("")}${broken.join("")}</div>`);
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

function unitHtml(b) {
  const u = b.unit;
  if (!u) return "";
  if (u.error) return `<div class="dg-why">⚠ ${esc(u.key)} — ${esc(u.error)}</div>`;
  const can = u.alive ? u.can_off : u.can_on, why = (u.alive ? u.why_off : u.why_on)[0] || "";
  const locked = !holding() ? "조작 권한이 없다" : can ? "" : why;
  const meta = u.alive ? `pid ${esc(u.pid)} · ${fmt(u.age_s, 0)} s` : u.started ? `rc ${esc(u.rc)}` : "—";
  const share = b.shares.length ? `<span class="dg-share" title="같이 켜지고 꺼진다: ${esc(b.shares.join(" · "))}">⛓ ${b.shares.length + 1}개 묶음</span>` : "";
  const manual = u.kind === "manual" ? `<div class="dg-why dg-manual" title="${esc(shellLine(u.argv))}">${copyBtn(shellLine(u.argv))} 운영자 셸에서 직접: <code>${esc(shellLine(u.argv))}</code></div>` : "";
  return `<div class="dg-unit"><button class="sw ${u.alive ? "on" : "off"}" role="switch" data-act="unit" data-arg="${esc(u.key)}:${u.alive ? "0" : "1"}"
      aria-checked="${u.alive}" aria-label="${esc(b.title)} 켜기/끄기" title="${esc(locked || (u.alive ? "끄기" : "켜기"))}"${locked ? " disabled" : ""}></button>
    <span>${esc(u.key)} · ${meta}</span>${share}${u.started ? `<button class="btn btn-sm btn-ghost" data-act="log" data-arg="${esc(u.key)}">로그</button>` : ""}</div>
    ${!can && why ? (u.goto
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
  const D = s.diagram;
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
  const u = S.session.units[key], boxes = S.session.diagram.cols.flat().filter((b) => b.unit && b.unit.key === key);
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
  put("links", s.links.map((b) => {
    const rows = b.rows.map((r) => `<tr class="tone-${esc(r.tone)}"><td><span class="lamp ${esc(r.tone)}"></span>${esc(r.name)}</td>
      <td class="num">${ageText(r.age_ms)}</td><td class="link-state">${esc(LINK_WORD[r.state] || r.state)}</td><td class="link-note">${esc(r.note)}</td></tr>`).join("");
    return `<div class="link tone-${esc(b.tone)}"><div class="link-head"><span class="lamp ${esc(b.tone)}"></span><b>${esc(b.title)}</b>
      <span class="link-state">${esc(LINK_WORD[b.state] || b.state)}</span></div>
      ${b.detail ? `<div class="link-detail">${esc(b.detail)}</div>` : ""}${rows ? `<table>${rows}</table>` : ""}</div>`;
  }).join('<div class="link-arrow">▸</div>'));
}

function renderNodes(s) {
  const br = s.bridge;
  put("bridge-meta", br.enabled ? `ROS 그래프 ${br.graph.length}개 노드${br.bad_lines ? ` · 해석 못 한 줄 ${br.bad_lines}` : ""}` : "브리지 꺼짐");
  const rows = s.nodes.map((n) => {
    const st = n.status;
    if (!st) return `<tr class="stale"><td><span class="lamp"></span><span class="node-name">${esc(n.name)}</span></td><td colspan="5">status 가 아직 오지 않았다</td></tr>`;
    const phase = String(st.phase ?? "");
    const busy = ["running", "TRACKING", "RAMPING"].includes(phase);
    const lamp = n.stale ? "" : st.ok === false ? "bad" : busy ? "live" : "ok";
    const why = (st.reasons || []).length ? `<div class="node-reasons">${st.reasons.map(esc).join("<br>")}</div>` : "";
    const raw = det(`node:${n.name}`, "원본 status", `<pre class="node-json">${esc(JSON.stringify(st, null, 1))}</pre>`);
    return `<tr class="${n.stale ? "stale" : ""}"><td><span class="lamp ${lamp}"></span><span class="node-name">${esc(n.name)}</span>${why}${raw}</td>
      <td><span class="phase">${esc(phase || "—")}</span></td><td class="num">${esc(st.episode ?? "—")}</td><td class="num">${esc(st.seq ?? "—")}</td>
      <td class="num">${fmt(st.proc_ms)}</td><td class="num">${n.stale ? "<b>" + fmt(n.age_s, 1) + " s</b>" : fmt(n.age_s, 1) + " s"}</td></tr>`;
  });
  const graph = br.graph.length ? det("graph", `보이는 ROS 노드 ${br.graph.length}개`, `<pre class="node-json">${br.graph.map(esc).join("\n")}</pre>`) : "";
  put("nodes", `<table><thead><tr><th>노드</th><th>phase</th><th class="num">ep</th><th class="num">seq</th><th class="num">proc ms</th><th class="num">나이</th></tr></thead><tbody>${rows.join("")}</tbody></table><div style="padding:0 12px 10px">${graph}</div>`);
}

function renderMetrics(s) {
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
const ROBOT_ART = { arms: "/static/robot_arms", right: "/static/robot_hand_right", left: "/static/robot_hand_left" };
const artCache = {};

async function loadArt(key) {
  if (artCache[key] !== undefined) return artCache[key];
  artCache[key] = "";
  try {
    const r = await fetch(`${ROBOT_ART[key]}.svg`);
    artCache[key] = r.ok ? await r.text() : "";
  } catch { artCache[key] = ""; }
  refresh();
  return artCache[key];
}

function jointRows(g, chan) {
  return g.rows.map((row) => {
    const v = row.vals[chan];
    const err = chan === "pos" ? row.err : null;
    return `<tr class="jr j-${esc(row.state)}"><td class="jn mono">${esc(row.joint.replace(/^[rl]_[ah][jl]_/, ""))}</td>
      <td class="jv mono">${v === null || v === undefined ? "—" : fmt(v, 3)}</td>
      <td class="je mono">${err === null ? "" : fmt(err, 3)}</td>
      <td class="jm">${row.state === "limit" ? "끝점" : row.state === "off" ? "벗어남" : ""}</td></tr>`;
  }).join("");
}

function jointPanel(g, chan) {
  if (!g) return `<div class="panel jpanel"><div class="empty">—</div></div>`;
  const bad = g.rows.filter((r) => r.state === "limit").length;
  return `<div class="panel jpanel"><div class="panel-head"><h2>${esc(g.title)}</h2>
      <span class="meta">${g.seen}/${g.total}${bad ? ` · <b class="bad">끝점 ${bad}</b>` : ""}</span></div>
    <table class="joints"><tbody>${jointRows(g, chan)}</tbody></table></div>`;
}

// 링크 id 로 칠한다: r_hj_index_2(관절) → r_hl_index_2(링크). 상태가 없는 링크는 칠하지 않아 렌더가 그대로 보인다.
const ART_TINT = { limit: "rgba(255,93,82,.45)", off: "rgba(227,160,8,.42)" };
const ART_EDGE = { limit: "#ff5d52", off: "#e3a008" };

function artWith(key, groups) {
  const svg = artCache[key];
  if (svg === undefined) { loadArt(key); return `<div class="empty">그림 여는 중…</div>`; }
  const rules = [];
  groups.forEach((g) => g.rows.forEach((r) => {
    if (r.state === "ok" || r.state === "missing") return;
    const id = r.joint.replace(/_([ah])j_/, "_$1l_");
    rules.push(`#${id} { fill: ${ART_TINT[r.state]}; stroke: ${ART_EDGE[r.state]}; }`);
  }));
  const img = `<img src="${ROBOT_ART[key]}.png" alt="" onerror="this.style.display='none'">`;
  return `${img}${svg ? `<style>${rules.join("\n")}</style>${svg}` : ""}`;
}

function renderRobot(s) {
  const r = s.robot;
  if (!r || !r.groups.length) return put("robot", `<div class="empty">관절 상태가 아직 없다 — 드라이버가 떠야 보인다.</div>`);
  const chan = S.robotChan && r.channels.some((c) => c.key === S.robotChan) ? S.robotChan : "pos";
  put("robot-chan", r.channels.map((c) =>
    `<button class="chip${c.key === chan ? " on" : ""}" data-act="robot-chan" data-arg="${esc(c.key)}">${esc(c.name)}<span class="hint"> ${esc(c.unit)}</span></button>`).join(""));
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
    <div class="hint" style="padding:6px 12px 10px">${chan === "pos" ? "현재 · 목표와의 차이 [rad]" : `현재 [${esc((r.channels.find((c) => c.key === chan) || {}).unit || "")}]`} · 색은 관절 상태(끝점 빨강 · 벗어남 노랑)</div>`);
}

// ── 모달 ────────────────────────────────────────────────────────────────
function modal(markup, real = false) {
  $("modal-card").className = "modal-card" + (real ? " real" : "");
  $("modal-card").innerHTML = markup;
  $("modal").hidden = false;
}
const closeModal = () => { $("modal").hidden = true; };

function approveModal(stageId) {
  const r = S.session.mission.rows.find((x) => x.id === stageId);
  const cmds = r.commands.map((c) => `<li class="cmd"><span class="cmd-kind ${esc(c.kind)}">${esc(c.kind)}</span><span>${esc(c.note)}<span class="cmd-argv">${esc(shellLine(c.argv))}</span></span><span></span></li>`).join("");
  modal(`<h3>실기 단계 승인 — ${esc(stageId)}</h3>
    <p>${esc(r.title)}</p>
    <p>이 승인은 아래 명령을 <b>한 번</b> 실행하는 것에 대한 것이다. 승인한 뒤 계약이나 체크포인트 파일이 바뀌면 승인은 무효가 된다.</p>
    <ul class="cmds">${cmds}</ul>
    <p style="margin-top:12px">로봇 주변이 비어 있고 물리 비상정지에 손이 닿는지 확인한 뒤, 단계 id <code>${esc(stageId)}</code> 를 그대로 입력할 것.</p>
    <input type="text" id="approve-typed" autocomplete="off" style="width:100%" placeholder="${esc(stageId)}">
    <div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button>
    <button class="btn btn-real" id="approve-go" data-act="approve-go" data-arg="${esc(stageId)}" disabled>승인</button></div>`, true);
  const input = $("approve-typed");
  input.addEventListener("input", () => { $("approve-go").disabled = input.value !== stageId; });
  input.focus();
}

function endModal() {
  const s = S.session, why = s.end_reasons, alive = s.procs.filter((p) => p.alive).length;
  modal(`<h3>run 끝내기</h3><p>띄워 둔 프로세스 ${alive}개를 그룹째 정지하고(SIGTERM → 5 s → SIGKILL) 브리지를 내린다. 기록은 <code>${esc(rel(s.run_dir))}</code> 에 남는다.</p>
    ${why.length ? `<ul class="reasons bad">${why.map((x) => `<li>${esc(x)}</li>`).join("")}</ul><label class="chk" style="margin-top:10px"><input type="checkbox" id="end-force"> 그래도 끝낸다 — pd 가 팔을 잡은 채로 프로세스가 죽는다</label>` : ""}
    <div class="modal-actions"><button class="btn btn-ghost" data-act="modal-close">취소</button><button class="btn ${why.length ? "btn-real" : "btn-primary"}" data-act="end-go">끝내기</button></div>`, why.length > 0);
}

// ── 동작 ────────────────────────────────────────────────────────────────
const acts = {
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
  approve(id) { approveModal(id); },
  async "approve-go"(id) { await call("POST", "/api/approve", { stage: id, typed: $("approve-typed").value }); closeModal(); refresh(); },
  "robot-chan"(key) { S.robotChan = key; refresh(); },
  async run(id) { await call("POST", "/api/stage/run", { stage: id }); refresh(); },
  async restart(id) {
    const live = liveUnits(id);
    if (!confirm(`${id} 가 띄운 ${live.length} 개를 내리고 다시 띄운다.\n\n${live.map((u) => `· ${u.note || u.key} (pid ${u.pid})`).join("\n")}`)) return;
    await call("POST", "/api/stage/run", { stage: id, restart: true });
    refresh();
  },
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

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-act]");
  if (!b || b.disabled) return;
  const fn = acts[b.dataset.act];
  if (fn) Promise.resolve(fn(b.dataset.arg)).catch(() => {});
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
window.addEventListener("resize", () => { if (S && S.session) drawCharts(S.session.metrics); drawWires(); });
if (window.ResizeObserver) new ResizeObserver(() => drawWires()).observe($("diagram"));   // 글꼴·표 펼침으로 상자가 밀려도 전선이 따라온다
refresh().then(connect);
