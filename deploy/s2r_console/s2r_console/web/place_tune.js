// ★11.18 고정 놓기 튜닝 탭(#place) — console.js 의 전역(call · prepareAction · acts · toast · esc · fmt · S · TABS · $)을 쓴다.
// 로봇을 움직이는 버튼(tilt_now · 하강 · 마저 놓기 · 오프셋 저장)은 기존 승인 창(prepareAction)을 그대로 탄다.
// jog 는 '이 탭에서 jog 켜기' 승인 한 번 뒤 클릭마다 prepare → execute(창 없음). 탭을 떠나거나 pd 가 TRACKING 을 벗어나면 서버가 끈다.
(() => {
  TABS.push("place");
  const PT_CUPS = { blue: "파랑 · 오른팔", pink: "분홍 · 오른팔", shaker: "shaker · 왼팔" };
  const AX = [["x", "mm"], ["y", "mm"], ["z", "mm"], ["yaw", "°"]];
  let ptCup = sessionStorage.getItem("s2r.place_tune.cup") || "blue", ptView = null, ptBusy = false;
  const stamp = () => {
    const d = new Date(), p = (n, w = 2) => String(n).padStart(w, "0");
    return `${p(d.getFullYear() % 100)}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}_${p(d.getMilliseconds(), 3)}`;
  };
  const stage = (kind) => `place_fixed_${ptCup}_${kind}_${ptCup === "shaker" ? "left" : "right"}`;

  async function ptLoad() {
    try {   // call() 은 GET 에도 body 를 실어 fetch 가 TypeError 를 낸다 — GET 은 fetch 로 직접
      const j = await (await fetch(`/api/place_tune?cup=${encodeURIComponent(ptCup)}`)).json();
      if (j.ok === false) toast(j.error || "place_tune 조회 실패", j.reasons);
      ptView = j.ok === false ? null : j;
    } catch (e) { toast("콘솔 서버에 닿지 않는다", [String(e)]); ptView = null; }
    ptRender();
  }
  function ptRender() {
    const root = $("pt-root");
    if (!root) return;
    const v = ptView, can = !!(S && S.session) && !ptBusy;
    const tracking = v && v.pd_phase === "TRACKING", off = v ? v.offset || {} : {};
    const cupBtns = Object.entries(PT_CUPS).map(([k, t]) =>
      `<button class="btn btn-sm ${k === ptCup ? "on" : ""}" data-act="pt-cup" data-arg="${k}" aria-pressed="${k === ptCup}">${esc(t)}</button>`).join("");
    const jog = AX.map(([a, u], i) => `<div class="pt-axis"><span class="pt-ax">${a}</span>${
      (u === "°" ? [-1, 1] : [-5, -1, 1, 5]).map((s) => {
        const d = [0, 0, 0, 0]; d[i] = s;
        return `<button class="btn btn-sm" data-act="pt-jog" data-arg="${d.join(",")}" ${can && v && v.armed && tracking ? "" : "disabled"}>${s > 0 ? "+" : ""}${s}${u}</button>`;
      }).join("")}</div>`).join("");
    const offTxt = off.error ? `<span class="bad">${esc(off.error)}</span>`
      : `x ${fmt(off.x, 4)} · y ${fmt(off.y, 4)} · z ${fmt(off.z, 4)} m · yaw ${fmt(off.yaw_deg, 1)}°`;
    const cum = v ? v.cum : [0, 0, 0, 0];
    const recent = (v && v.recent || []).map((r) => {
      const ok = r.rc === 0, tag = ok ? "ok" : r.rc === 2 ? "warn" : "bad";
      const o = r.offset && r.offset.ok ? ` · 저장 Δ ${esc(JSON.stringify(r.offset.delta))}` : "";
      const needSave = r.mode === "resume" && (r.failed || []).some((k) => k.startsWith("resume_"))
        ? ' <span class="hint warn">하강 끝 계획과 멀다 — 오프셋 저장을 먼저 할 것</span>' : "";
      return `<li><span class="badge ${tag}">${esc(r.mode)} rc ${esc(r.rc)}</span>${r.execute ? "" : ' <span class="badge mute">드라이런</span>'}
        ${r.jog ? `<span class="mono">${esc(r.jog)}</span>` : ""}${(r.failed || []).length ? ` <span class="bad">${esc((r.reasons || r.failed).join(" · "))}</span>` : ""}${o}${needSave}${r.error ? ` <span class="bad">${esc(r.error)}</span>` : ""}
        <div class="mono pt-path">${esc(r.dir)}</div></li>`;
    }).join("");
    root.innerHTML = `
      <section class="ev-block" aria-labelledby="pt-h-cup"><h2 id="pt-h-cup" class="ev-h">컵</h2>
        <div class="pt-row">${cupBtns}</div>
        <dl class="pt-kv"><dt>pd</dt><dd><span class="badge ${tracking ? "ok" : "warn"}">${esc(v ? v.pd_phase || "—" : "—")}</span></dd>
          <dt>offset</dt><dd class="mono">${offTxt} <small class="hint">${esc(v ? v.config : "")}</small></dd></dl></section>
      <section class="ev-block" aria-labelledby="pt-h-run"><h2 id="pt-h-run" class="ev-h">순서</h2>
        <div class="pt-row">
          <button class="btn btn-real" data-act="pt-stage" data-arg="tilt_now" ${can ? "" : "disabled"}>1 · tilt_now</button>
          <button class="btn btn-real" data-act="pt-stage" data-arg="descend" ${can ? "" : "disabled"}>2 · 하강까지(쥔 채)</button>
          <button class="btn btn-real" data-act="pt-save" ${can && tracking ? "" : "disabled"}>4 · 오프셋 저장</button>
          <button class="btn btn-real" data-act="pt-resume" ${can && tracking ? "" : "disabled"}>5 · 마저 놓기</button>${v && v.unsaved_jog ? ` <span class="hint warn">${esc(v.unsaved_msg)}</span>` : ""}
        </div>
        ${v && !tracking ? `<p class="hint warn">4 · 5 · jog 는 그 팔 pd 가 TRACKING 일 때만 켜진다 — 지금 pd ${esc(v.pd_phase || "상태 없음")}.</p>` : ""}
        <p class="hint">하강·마저 놓기·저장은 매번 승인 창을 띄운다. 저장은 로봇을 움직이지 않는다(현재 자세 ↔ 하강 끝 계획 차이를 yaml 에 더하고 _bak 에 백업).</p></section>
      <section class="ev-block" aria-labelledby="pt-h-jog"><h2 id="pt-h-jog" class="ev-h">3 · jog (홀더 기준)</h2>
        <div class="pt-row">${v && v.armed
          ? `<span class="badge live">jog 켜짐 · ${esc(v.armed_left_s)} s 남음</span><button class="btn btn-sm" data-act="pt-arm" data-arg="off">jog 끄기</button>`
          : `<button class="btn btn-real" data-act="pt-arm" data-arg="on" ${can && tracking ? "" : "disabled"}>이 탭에서 jog 켜기</button>`}</div>
        ${jog}
        <p class="mono">하강 끝 계획에서 Δ x ${fmt(cum[0], 0)} · y ${fmt(cum[1], 0)} · z ${fmt(cum[2], 0)} mm · yaw ${fmt(cum[3], 0)}°
          <small class="hint">한 번 ≤ ${v ? v.limits.step_mm : 10} mm · ${v ? v.limits.step_deg : 3}°, 하강 끝에서 ≤ ${v ? v.limits.cum_mm : 40} mm · ${v ? v.limits.cum_deg : 15}°, 관절 ≤ 0.1 rad/s, 손은 쥔 값 유지</small></p></section>
      <section class="ev-block" aria-labelledby="pt-h-res"><h2 id="pt-h-res" class="ev-h">최근 결과</h2>
        <ul class="pt-recent">${recent || '<li class="hint">아직 없다</li>'}</ul></section>`;
  }
  async function ptArmOff() {
    if (!(ptView && ptView.armed)) return;
    try { await call("POST", "/api/place_tune/arm", { cup: ptCup, on: false }); }
    catch (e) { toast("jog 끄기 실패 — 서버가 2 분 · 폴링 끊김 10 s 뒤 끈다", [String(e)]); }
  }
  // jog 한 번이 끝날 때까지(그 stamp 의 result.json 이 최근 결과에 보일 때까지) 기다린다 — 실행은 202 로 먼저 돌아온다
  async function ptWaitDone(st, limitMs = 90000) {
    const t0 = Date.now(), tail = `_tune_jog_${st.slice(7)}`;
    while (Date.now() - t0 < limitMs) {
      await new Promise((r) => setTimeout(r, 500));
      await ptLoad();
      const r = ((ptView && ptView.recent) || []).find((x) => String(x.dir || "").endsWith(tail));
      if (r) { if (r.rc !== 0) toast(`jog rc ${r.rc}`, r.reasons || r.failed || []); return; }
    }
    toast("jog 결과가 90 s 안에 안 보인다 — 실행 로그를 볼 것", []);
  }

  const tabOrig = acts.tab;
  Object.assign(acts, {
    tab(v) { if (v !== "place") ptArmOff(); tabOrig(v); if (v === "place") ptLoad(); },
    async "pt-cup"(v) { await ptArmOff(); ptCup = v; sessionStorage.setItem("s2r.place_tune.cup", v); ptLoad(); },
    "pt-stage"(v) { return prepareAction(v === "tilt_now" ? "tilt_now" : stage("descend")); },
    "pt-save"() { return prepareAction(stage("descend"), { tune: "save", stamp: stamp() }); },
    "pt-resume"() { return prepareAction(stage("descend"), { tune: "resume", stamp: stamp() }); },
    async "pt-arm"(v) {
      if (v === "off") { await ptArmOff(); return ptLoad(); }
      if (!confirm(`${PT_CUPS[ptCup]} — 이 탭에서 jog 를 켠다. 켜진 동안 jog 버튼은 확인 없이 팔을 움직인다(한 번 ≤ 10 mm · 3°). 탭을 떠나거나 pd 가 TRACKING 을 벗어나면 꺼진다.`)) return;
      try { await call("POST", "/api/place_tune/arm", { cup: ptCup, on: true }); } finally { ptLoad(); }
    },
    async "pt-jog"(v) {
      if (ptBusy) return;
      ptBusy = true; ptRender();
      try {
        const st = stamp();
        const prepared = await call("POST", "/api/actions/prepare", { action: stage("descend"), tune: "jog", jog: v, stamp: st });
        if (prepared.enabled === false) { toast("jog 를 지금 할 수 없다", prepared.reasons || []); return; }
        await call("POST", "/api/actions/execute", { confirmation_id: prepared.confirmation_id });
        await ptWaitDone(st);
      } catch (e) { toast("jog 실패", [String(e)]); }
      finally { ptBusy = false; ptLoad(); }
    },
  });
  // 화면 갱신: 탭이 열려 있으면 1 s 마다(jog 중에도 — 서버는 폴링이 10 s 끊기면 jog 를 끈다)
  setInterval(() => { if (activeTab === "place") ptLoad(); }, 1000);
  // 창을 닫으면 끄기를 못 보낸다(X-S2R-Console 헤더 필요) — 서버가 폴링 끊김 10 s · ARM_TTL_S(120 s) · pd 이탈로 끈다
  if (globalThis.location?.hash === "#place") acts.tab("place");
})();
