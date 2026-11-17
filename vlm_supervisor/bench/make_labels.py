"""정답 라벨 만들기 — python3 make_labels.py stages.jsonl > labels_real_1117.csv (stages.jsonl = 단계별 물체 · 손 특징, 분석 도구 산출).
손 힘 · 각도는 라벨에 쓰지 않는다(센서 규칙과 순환 방지) — 물체 FP++ live_pose, 러너 물체 신호, 운영자 정지, 카메라 확인만."""
import csv, json, sys

CAM = {  # 카메라 bag head_mask/depth 단계 시작 · 끝 프레임을 눈으로 본 것(11.17)
    ("261116", "100859_grasp_s_left", "grasp_lift"): ("ok", "카메라: 컵이 들려 카메라 앞으로 옴"),
    ("261116", "101519_grasp_s_pink_right", "grasp_lift"): ("fail", "카메라: 컵 제자리, 손 안 닿음"),
}


def kind(state):
    s = state.lower()
    if any(k in s for k in ("place", "release", "open")):
        return "release"
    if any(k in s for k in ("grasp", "pick")):
        return "grasp"
    if any(k in s for k in ("tilt", "carry", "home_r_after", "lift")):
        return "hold"
    return None


def sig_all(r):
    out = {}
    for s in r["signals"].values():
        if isinstance(s, dict):
            out.update(s)
    return out


rows = []
for l in open(sys.argv[1]):
    r = json.loads(l)
    k = kind(r["state"])
    if "feats" not in r or k is None:
        continue
    sg = sig_all(r)
    stop_after = any(a.startswith(("stop:operator", "safe_stop")) for a in r["after"][:2])
    for side, f in r["feats"].items():
        held = (r["lho"] if side == "left" else r["rho"]) or "EMPTY"
        ob = f.get("obj") or {}
        z0, z1, dp = ob.get("z0"), ob.get("z1"), ob.get("d_palm1")
        lab, ev = "unsure", []
        if (r["day"], r["ep"], r["state"]) in CAM:
            lab, e = CAM[(r["day"], r["ep"], r["state"])]
            ev.append(e)
        if k == "grasp":
            if z0 is not None and z1 is not None and dp is not None:
                rise = z1 - z0
                ev.append(f"물체 {ob['name']} {rise * 100:+.1f} cm, 손바닥 {dp * 100:.0f} cm")
                if lab == "unsure":
                    if rise >= 0.03 and dp <= 0.12:
                        lab = "ok"
                    elif rise < 0.01:
                        lab = "fail"
                        if r["result"] == "COMPLETED":
                            ev.append("러너는 성공(정책 관측 상승) — 물체 · 손 힘이 반대")
            elif "lift_hold" in (r["reason"] or "") and lab == "unsure":
                lab = "ok"; ev.append("러너 lift_hold(정책 관측 컵 상승)")
        elif k == "hold":
            if held == "EMPTY":
                continue
            want = {"SRC_BLUE": "blue", "SRC_PINK": "pink", "SHAKER": "shaker", "CUP": ""}.get(held, held.lower())
            if dp is not None and want not in ob.get("name", ""):
                ev.append(f"쥔 물체 {held} live 없음(최근접 {ob['name']} {dp * 100:.0f} cm)")
            elif dp is not None:
                ev.append(f"쥔 물체 {held} 최근접 {ob['name']} 손바닥 {dp * 100:.0f} cm")
                if lab == "unsure":
                    lab = "ok" if dp <= 0.10 else "fail" if dp >= 0.20 else "unsure"
        else:  # release
            cih = sg.get("cup_in_holder")
            live = cih.get("live") if isinstance(cih, dict) else cih
            reason = (r["reason"] or "")
            if live is True:
                lab = "ok"; ev.append("러너 cup_in_holder live=True")
            elif live is False:
                lab = "fail"; ev.append("러너 cup_in_holder live=False")
            if "opened before latch" in reason or r["result"] == "place_failed":
                if lab == "ok":
                    lab = "unsure"
                elif lab == "unsure":
                    lab = "fail"
                ev.append("러너 " + reason[:40])
            if dp is not None:
                ev.append(f"물체 손바닥 {dp * 100:.0f} cm z {z1}")
        if r["result"] not in ("COMPLETED",):
            ev.append(f"러너 {r['result']}")
        if stop_after:
            ev.append("직후 정지")
        rows.append({"day": r["day"], "ep": r["ep"], "state": r["state"], "side": side, "kind": k,
                     "label": lab, "runner": r["result"], "evidence": " · ".join(ev)})

w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
