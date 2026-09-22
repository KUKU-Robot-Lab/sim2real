"use strict";
/* S2R 배포 콘솔 — 화면. 규칙 셋:
 *  1. 상태 지식은 서버에 있다. 여기서는 서버가 준 can_run / tone / reasons 를 그릴 뿐, 스스로 판단하지 않는다.
 *  2. ROS 에서 온 문자열은 전부 esc() 를 거친다.
 *  3. 서버와 끊기면 화면을 덮는다 — 오래된 값을 살아 있는 값처럼 보여 주지 않는다.
 */
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
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
  $("nextstep").hidden = !s;
  if (!s) return renderLanding();
  renderNext(s);
  renderStages(s);
  renderDiagram(s);
  renderLinks(s);
  renderNodes(s);
  renderMetrics(s);
  renderPolicy(s);
  renderProcs(s);
  renderEvents(s);
  put("quick", S.quick.map((q) => `<button class="btn-stop" data-act="quick" data-arg="${esc(q.name)}" title="${esc(q.help)}">■ ${esc(q.label)}</button>`).join(""));
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
let flashStage = null, flashUntil = 0;                                       // 단계로 간 뒤 잠깐 강조 — 다시 그려도 유지
const goBtn = (id) => ` <button class="btn btn-sm" data-act="goto-stage" data-arg="${esc(id)}">${esc(id)} 단계로 가기 ↓</button>`;
function renderNext(s) {
  const m = s.mission, R = s.runner;
  const cur = m.rows.find((r) => r.current);
  let msg, go = "";
  if (!holding()) {
    msg = `<b>먼저 조작 권한을 잡을 것</b> — 오른쪽 위에 이름을 넣고 <b>조작 권한 잡기</b>. 노드는 아래 미션 단계의 <b>▶ 실행</b>으로 켜진다.`;
  } else if (R && R.active) {
    const w = (R.steps || []).find((st) => st.status === "waiting");
    msg = w ? `<b>✋ 확인을 기다린다</b> — ${esc(R.stage)} 단계: 다른 셸에서 명령을 실행한 뒤 카드의 <b>실행했고 정상이다</b>를 누를 것.`
      : `<b>${esc(R.stage)}</b> 단계 실행 중…`;
    go = goBtn(R.stage);
  } else if (cur) {
    const how = cur.touches_real && !cur.approved ? "<b>승인…</b> 뒤 <b>▶ 실행</b>" : "<b>▶ 실행</b>";
    msg = `<b>다음: ${esc(cur.id)}</b> — ${esc(cur.title)}. 카드의 ${how}.`
      + (cur.reasons.length ? ` <span class="warn">막힘: ${esc(cur.reasons[0])}</span>` : "");
    go = goBtn(cur.id);
  } else {
    msg = `모든 단계가 끝났다 — 끝낼 때는 정지 바의 <b>PD 해제</b> → 오른쪽 아래 <b>run 끝내기</b>.`;
  }
  put("nextstep", `<span class="ns-text">${msg}</span>${go}`);   // 문장은 한 덩어리 — flex 간격이 굵은 글자 사이에 끼지 않게
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

function renderStages(s) {
  const can = holding();
  const R = s.runner;
  const m = s.mission;
  put("mission-meta", `${esc(m.name)}`);
  const rows = m.rows.map((r, i) => {
    const running = r.current && m.status === "RUNNING";
    const failed = r.current && (m.status === "FAILED" || m.status === "ABORTED");
    const flash = r.id === flashStage && Date.now() < flashUntil ? "flash" : "";
    const cls = ["stage", r.done ? "done" : "", r.current ? "current" : "", running ? "running" : "", failed ? "failed" : "", r.reasons.length ? "blocked" : "", flash].join(" ");
    const badges = [
      r.touches_real ? `<span class="badge real">실기</span>` : "",
      r.touches_real && r.approved ? `<span class="badge ok">승인됨</span>` : "",
      failed ? `<span class="badge bad">${esc(m.status)}</span>` : "",
      r.reasons.length ? `<span class="badge warn">막힘</span>` : "",
    ].join("");
    const mine = R && R.stage === r.id;
    const steps = mine ? R.steps : null;
    const cmds = r.commands.map((c, k) => {
      const st = steps ? steps[k] : null;
      const chip = st && st.status !== "pending" ? `<span class="chip ${esc(st.status)}">${esc(stepLabel(st))}</span>` : "";
      const kind = { manual: "✋ 수동", background: "⟳ 배경", foreground: "▶ 실행" }[c.kind];
      const logBtn = st && st.kind !== "manual" && st.status !== "pending" && st.status !== "kept" ? `<button class="btn btn-sm btn-ghost" data-act="log" data-arg="${esc(st.key)}">로그</button>` : "";
      const manual = st && st.status === "waiting" ? `<div class="manual-box"><b>다른 셸에서 직접 실행할 것</b> — 콘솔은 이 명령을 실행하지 않는다.
          <pre>${esc(c.argv.join(" "))}</pre>
          <div class="actions"><button class="btn btn-primary btn-sm" data-act="ack" data-arg="${k}:1" ${can ? "" : "disabled"}>실행했고 정상이다 → 다음</button>
          <button class="btn btn-sm" data-act="ack" data-arg="${k}:0" ${can ? "" : "disabled"}>정상이 아니다 → 중단</button></div></div>` : "";
      const detail = st && st.detail ? `<span class="cmd-argv" style="color:var(--warn)">${esc(st.detail)}</span>` : "";
      return `<li class="cmd"><span class="cmd-kind ${esc(c.kind)}">${kind}</span><span>${esc(c.note || c.argv.slice(0, 3).join(" "))}${detail}<span class="cmd-argv">${esc(c.argv.join(" "))}</span></span><span>${chip} ${logBtn}</span>${manual}</li>`;
    }).join("");
    const reasons = r.reasons.length ? `<ul class="reasons">${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
    const stale = r.approval_stale.length ? `<ul class="reasons bad"><li><b>이전 승인이 무효가 됐다</b> — 승인한 뒤 파일이 바뀌었다</li>${r.approval_stale.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
    const note = failed && m.note ? `<ul class="reasons bad"><li>${esc(m.note)}</li></ul>` : "";
    let actions = "";
    if (r.current) {
      if (running) actions = `<button class="btn btn-sm" data-act="abort-stage" ${can ? "" : "disabled"}>■ 이 단계 중단</button><span class="hint">실행 중…</span>`;
      else {
        if (r.touches_real && !r.approved) actions += `<button class="btn btn-real" data-act="approve" data-arg="${esc(r.id)}" ${can && r.can_approve ? "" : "disabled"}>승인…</button>`;
        actions += `<button class="btn ${r.touches_real ? "btn-real" : "btn-primary"}" data-act="run" data-arg="${esc(r.id)}" ${can && r.can_run ? "" : "disabled"}>▶ ${failed ? "다시 실행" : "실행"}</button>`;
        if (!can) actions += `<span class="hint">조작 권한이 없다</span>`;
        else if (r.touches_real && !r.approved && !r.reasons.length) actions += `<span class="hint">실기를 움직이는 단계다 — 승인이 먼저다</span>`;
      }
      actions = `<div class="actions">${actions}</div>`;
    }
    const body = r.current || mine ? `<ul class="cmds">${cmds}</ul>` : det(`cmds:${r.id}`, `명령 ${r.commands.length}개`, `<ul class="cmds">${cmds}</ul>`);
    return `<li id="stage-${esc(r.id)}" class="${cls}"><div class="stage-rail"><span class="dot">${r.done ? "✓" : ""}</span></div><div class="stage-body">
      <div class="stage-line"><span class="stage-id">${i + 1}. ${esc(r.id)}</span>${badges}</div><div class="stage-title">${esc(r.title)}</div>
      ${reasons}${stale}${note}${r.commands.length ? body : ""}${actions}</div></li>`;
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
  const manual = u.kind === "manual" ? `<div class="dg-why dg-manual" title="${esc(u.argv.join(" "))}">운영자 셸에서 직접: <code>${esc(u.argv.join(" "))}</code></div>` : "";
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
  return `<div class="dg-box tone-${esc(b.tone)}" id="dg-box-${esc(b.id)}"><div class="dg-head"><span class="lamp ${esc(b.tone)}"></span><b>${esc(b.title)}</b>
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

function renderProcs(s) {
  const alive = s.procs.filter((p) => p.alive).length;
  put("procs-meta", `${alive} / ${s.procs.length} 살아 있음`);
  if (!s.procs.length) return put("procs", `<div class="empty">아직 띄운 프로세스가 없다.</div><div style="padding:0 12px 10px"><button class="btn btn-sm btn-ghost" data-act="log" data-arg="bridge">브리지 로그</button></div>`);
  const rows = s.procs.map((p) => `<tr><td><span class="lamp ${p.alive ? "live" : p.rc === 0 ? "ok" : "bad"}"></span><span class="mono">${esc(p.key)}</span><div class="hint">${esc(p.note)}</div></td>
    <td class="num">${p.alive ? `${fmt(p.age_s, 0)} s` : `rc ${esc(p.rc)}`}</td><td><button class="btn btn-sm btn-ghost" data-act="log" data-arg="${esc(p.key)}">로그</button></td></tr>`);
  put("procs", `<table><tbody>${rows.join("")}</tbody></table><div style="padding:6px 12px 10px"><button class="btn btn-sm btn-ghost" data-act="log" data-arg="bridge">브리지 로그</button></div>`);
}

function renderEvents(s) {
  const rows = s.events.slice().reverse().map((e) =>
    `<li class="ev-${esc(e.kind)}"><time>${new Date(e.t * 1000).toLocaleTimeString("ko-KR", { hour12: false })}</time><span class="ev-dot"></span><span>${esc(e.text)}</span></li>`);
  put("events", rows.join("") || `<li><time></time><span></span><span class="hint">아직 사건이 없다.</span></li>`);
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
  const cmds = r.commands.map((c) => `<li class="cmd"><span class="cmd-kind ${esc(c.kind)}">${esc(c.kind)}</span><span>${esc(c.note)}<span class="cmd-argv">${esc(c.argv.join(" "))}</span></span><span></span></li>`).join("");
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
  async run(id) { await call("POST", "/api/stage/run", { stage: id }); refresh(); },
  async ack(arg) { const [i, ok] = arg.split(":"); await call("POST", "/api/stage/ack", { index: Number(i), ok: ok === "1" }); refresh(); },
  async "abort-stage"() { await call("POST", "/api/stage/abort"); refresh(); },
  async quick(name) { await call("POST", `/api/quick/${name}`); toast("정지 요청을 보냈다 — 결과는 '사건' 에 뜬다", [], true); refresh(); },
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
  "goto-stage"(id) {                                                         // 잠금 줄·다음 할 일 → 그 단계 카드
    const li = $(`stage-${id}`);
    if (!li) return;
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    li.scrollIntoView({ behavior: calm ? "auto" : "smooth", block: "center" });
    flashStage = id; flashUntil = Date.now() + 2000;
    li.classList.remove("flash"); void li.offsetWidth; li.classList.add("flash");
    li.querySelector('[data-act="ack"],[data-act="approve"]:not([disabled]),[data-act="run"]')?.focus({ preventScroll: true });
  },
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
