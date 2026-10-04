// 상황판 화면 확인 — 헤드리스 크롬을 CDP 로 몰아 화면 · 콘솔 오류 · 패널 상태를 본다(Node 22, 의존성 없음).
//   node deploy/s2r_console/tools/screenshot_cdp.mjs http://127.0.0.1:8091/ out.png [wait_ms] [css 선택자 — 그 요소만]
// chrome --screenshot 는 상황판(/api/stream SSE)에서 JS 가 그린 화면을 못 담는다(10.04 — 빈 화면). CDP 로 그린 뒤 찍는다.
import { spawn } from "node:child_process";
const [url, png, waitMs = "6000", selector = ""] = process.argv.slice(2);
const port = 9333 + Math.floor(Math.random() * 500);
const chrome = spawn("google-chrome", ["--headless=new", "--disable-gpu", "--no-sandbox", `--remote-debugging-port=${port}`,
  "--window-size=1600,2600", "--user-data-dir=/tmp/cdp-prof-" + port, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); } }
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map(); const logs = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === "Runtime.consoleAPICalled") logs.push(`[console.${m.params.type}] ` + m.params.args.map((a) => a.value ?? a.description).join(" "));
  if (m.method === "Runtime.exceptionThrown") logs.push("[exception] " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  if (m.method === "Log.entryAdded") logs.push(`[log.${m.params.entry.level}] ${m.params.entry.text}`);
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send("Runtime.enable"); await send("Log.enable"); await send("Page.enable");
await send("Page.navigate", { url });
await sleep(Number(waitMs));
const r = await send("Runtime.evaluate", { expression: "JSON.stringify({len: document.body.innerText.length, landing: document.getElementById('landing')?.hidden, work: document.getElementById('work')?.hidden, ep: document.getElementById('episode-panel')?.hidden, S: typeof S !== 'undefined' && S ? Object.keys(S) : null})", returnByValue: true });
console.log("state", r.result?.result?.value);
let clip;
if (selector) {
  const b = await send("Runtime.evaluate", { returnByValue: true, expression:
    `(() => { const e = document.querySelector(${JSON.stringify(selector)}); if (!e) return null; e.scrollIntoView();
      const r = e.getBoundingClientRect(); return {x: r.x + scrollX, y: r.y + scrollY, width: r.width, height: r.height}; })()` });
  const r = b.result?.result?.value;
  if (r && r.width > 0) clip = { ...r, scale: 1 };
  else console.log(`selector ${selector}: 없음 · 크기 0`);
}
const shot = await send("Page.captureScreenshot", clip ? { format: "png", clip, captureBeyondViewport: true }
                                                     : { format: "png", captureBeyondViewport: false });
const fs = await import("node:fs"); fs.writeFileSync(png, Buffer.from(shot.result.data, "base64"));
console.log(logs.slice(0, 20).join("\n"));
ws.close(); chrome.kill();
process.exit(0);
