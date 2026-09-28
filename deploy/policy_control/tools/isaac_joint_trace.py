"""joint family 학습 env 에서 정책을 짧게 돌려 **관절 순서 · 관측 · 행동 · 상태** 궤적을 남긴다(Isaac, 로봇 없음).

    cd ~/rl_ws/hdgp && ../IsaacLab/isaaclab.sh -p ~/rl_ws/sim2real/deploy/policy_control/tools/isaac_joint_trace.py \
        --run ~/rl_ws/sim2real/deploy/policies/right_m15_e800 --num_envs 2 --steps 240 \
        --out ~/rl_ws/sim2real/deploy/policies/right_m15_e800/trace.npz

hdgp 는 고치지 않는다 — env 를 그대로 만들고 **학습 전용 관측 노이즈 · 지연만 끈다**(그래야 env 관측 = 배포가 만드는
깨끗한 관측이다). 남기는 것:
  meta   joint_names(시뮬레이터 순서) · 손 관측 순서 · 손 행동(프로필) 순서 · 행동 한계 · 팔 한계 · 시작 자세
  매 스텝 obs(133, env 관측) · action(26, 정책 원출력) · 전 관절 q/qd · 물체 · 목표 자세(env-local) ·
         팔 q* · 손 q*(행동 적용 뒤) · episode_length_buf
배포 쪽 test_joint_trace_parity 가 이 파일로 관측 · 디코더를 대조한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True, help="deploy/policies/<id> (params/ · nn/ · joint_contract.json)")
parser.add_argument("--task", default="", help="기본 = run 의 agent.yaml config.name")
parser.add_argument("--num_envs", type=int, default=1, help="LSTM 이면 1 — 로더가 은닉 상태를 하나만 든다")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", type=Path, required=True)
# 실기 대조(09.28 사용자: "실제값하고 sim값을 비교 분석") — 실기 기록(joint_recorder.py npz)과 같은 조건으로
parser.add_argument("--cup-xy", default="", help="컵 스폰을 이 자리(env-local = 로봇 base)에 고정 'x,y' — 무작위 폭 0")
parser.add_argument("--keep-train-delays", action="store_true",
                    help="학습 때 행동 · 관측 · 물체 지연을 그대로 둔다(기본은 배포 관측과 맞추려고 끈다)")
parser.add_argument("--replay-actions", type=Path, default=None,
                    help="실기 기록 npz 의 정책 행동을 순서대로 넣는다(정책 대신) — 같은 명령에 sim 이 어떻게 반응하는가")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import numpy as np           # noqa: E402
import torch                 # noqa: E402
import yaml                  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
import openarm.agnostic.tasks.manip_stages.config  # noqa: E402,F401  (등록 ImportError 를 드러낸다)

SIM2REAL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SIM2REAL / "scripts"))
from policy_loader import RLGamesActorPolicy, RLGamesLstmActorPolicy  # noqa: E402

run = args.run.resolve()
agent_yaml = run / "params" / "agent.yaml"
agent = yaml.safe_load(agent_yaml.read_text())
contract = json.loads((run / "joint_contract.json").read_text())
task = args.task or str(agent["params"]["config"]["name"])

cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
# 학습 때 설정으로 되돌린다 — play.py 와 같은 도구(hdgp scripts/tools/run_cfg_restore.py). 레지스트리 기본값은
# 그 뒤 바뀌었을 수 있다(09.28: 기본값으로 띄우자 리셋 가드가 손바닥–컵 380.6 mm 로 거부했다).
HDGP = Path.cwd()
sys.path.insert(0, str(HDGP / "scripts" / "tools"))
from run_cfg_restore import apply_logged_env_cfg, load_run_yaml, rebase_logged_paths  # noqa: E402
apply_logged_env_cfg(cfg, rebase_logged_paths(load_run_yaml(str(run / "params" / "env.yaml")),
                                              workspace_root=str(HDGP.parent)))
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
# 런 폴더에 보상 사본(reward/compute_reward.py)이 있으면 그것을 쓴다 — 다른 호스트의 hdgp 에 그 라운드 파일이 없을 수 있다
# (09.28 vision-3090: cup_grasp_l/iter_01 없음). trace 는 궤적만 남기므로 보상 값은 결과에 영향이 없다.
if (run / "reward" / "compute_reward.py").is_file() and hasattr(cfg, "reward_code_path"):
    cfg.reward_code_path = str(run / "reward" / "compute_reward.py")
# 부팅 가드 `start_palm_dist_band_m` 은 갓 리셋한 env 들의 **평균** 손바닥–컵 거리를 본다. env 1 개면 무작위 스폰
# 한 번이 그대로 평균이라 가장자리(09.28: 380.6 mm, 상한 380)에 걸린다 — 물리가 아니라 부팅 점검이라 넓힌다.
if hasattr(cfg, "start_palm_dist_band_m") and args.num_envs < 64:
    cfg.start_palm_dist_band_m = (0.0, 10.0)
# 학습 전용 관측 교란을 끈다 — 배포는 이것 없이 관측을 만든다(노이즈는 0, 지연 큐 길이 1 = 지연 0)
_off = [("obs_noise_qpos", 0.0), ("obs_noise_qvel", 0.0), ("obs_noise_body", 0.0), ("obs_noise_object", 0.0),
        ("obs_object_xyz_std", 0.0), ("obs_object_rot_deg", 0.0)]
if not args.keep_train_delays:
    _off += [("obs_delay_steps", 1), ("action_delay_steps", 1), ("object_delay_steps", 1)]
for k, v in _off:
    if hasattr(cfg, k):
        setattr(cfg, k, v)
if args.cup_xy:
    _x, _y = (float(v) for v in args.cup_xy.split(","))
    cfg.object_spawn_center_override = (_x, _y)
    cfg.spawn_range = 0.0
    if hasattr(cfg, "respawn_on_fail"):
        cfg.respawn_on_fail = False
if contract["recurrent"] and args.num_envs != 1:
    raise SystemExit("recurrent policy: --num_envs 1 (policy_loader keeps a single hidden state)")
env = gym.make(task, cfg=cfg).unwrapped
obs_dict, _ = env.reset()
N, dev = env.num_envs, env.device

ckpt = run / "nn" / Path(contract["checkpoint"]).name
cls = RLGamesLstmActorPolicy if contract["recurrent"] else RLGamesActorPolicy
policy = cls(str(agent_yaml), str(ckpt), obs_dim=int(contract["obs_dim"]), action_dim=int(contract["action_dim"]),
             device=str(dev), action_clip=None)
if getattr(policy, "load_report", None) is not None and policy.load_report.missing_keys:
    raise SystemExit(f"checkpoint did not fill {len(policy.load_report.missing_keys)} weights")
clip = float(agent["params"]["env"]["clip_observations"])

names = list(env.robot.data.joint_names)
meta = {
    "task": task, "joint_names": names,
    "hand_obs_order": [names[int(i)] for i in env._hand_ids_t.tolist()],
    "hand_action_order": [names[int(i)] for i in env._syn_ids],
    "arm_joints": [names[int(i)] for i in env._arm_ids_t.tolist()],
    "act_lo": env._act_lo.tolist(), "act_hi": env._act_hi.tolist(),
    "arm_lo": env._arm_lo.reshape(-1, env._arm_lo.shape[-1])[0].tolist() if env._arm_lo.ndim > 1 else env._arm_lo.tolist(),
    "arm_hi": env._arm_hi.reshape(-1, env._arm_hi.shape[-1])[0].tolist() if env._arm_hi.ndim > 1 else env._arm_hi.tolist(),
    "hand_reset_q": env._hand_reset_q.tolist(),
    "palm_body": env.robot.data.body_names[env.palm_idx],
    "tip_bodies": [env.robot.data.body_names[int(i)] for i in env._tip_ids_t.tolist()],
    "k_arm": float(env.cfg.k_arm), "arm_ema": float(env.cfg.arm_ema), "hand_ema": float(env.cfg.hand_ema),
    "num_envs": N, "steps": args.steps, "seed": args.seed, "clip_observations": clip,
}
rec = {k: [] for k in ("obs", "action", "q", "qd", "obj_pos", "obj_quat", "goal_pos", "goal_quat", "arm_qstar",
                       "hand_qstar", "palm_pos", "palm_quat", "tips", "ep_len", "tau")}
meta.update(cup_xy=args.cup_xy, keep_train_delays=bool(args.keep_train_delays),
            delays={k: getattr(cfg, k, None) for k in ("action_delay_steps", "obs_delay_steps", "object_delay_steps")},
            replay_actions=str(args.replay_actions or ""))
replay = None
if args.replay_actions is not None:
    # 실기 기록의 정책 행동 — joint_node 가 running 인 구간을 seq 순서로(정책이 낸 그대로, 클립 전 원출력)
    _d = np.load(args.replay_actions)
    _run = [float(t) for t, s in zip(_d["jn_t"], _d["jn_json"]) if json.loads(str(s)).get("phase") == "running"]
    _keep = (_d["act_t"] >= min(_run)) & (_d["act_t"] <= max(_run))
    _order = np.argsort(_d["act_seq"][_keep])
    replay = torch.as_tensor(_d["act"][_keep][_order], dtype=torch.float32, device=dev)
    meta["replay_rows"] = int(replay.shape[0])
    args.steps = min(args.steps, int(replay.shape[0]))


def snap(obs: torch.Tensor) -> None:
    rec["obs"].append(obs.cpu().numpy())
    rec["q"].append(env.robot.data.joint_pos.cpu().numpy())
    rec["qd"].append(env.robot.data.joint_vel.cpu().numpy())
    rec["obj_pos"].append((env.object.data.root_pos_w - env.scene.env_origins).cpu().numpy())
    rec["obj_quat"].append(env.object.data.root_quat_w.cpu().numpy())
    rec["goal_pos"].append(env.goal_pos.cpu().numpy())
    rec["goal_quat"].append(env.goal_quat.cpu().numpy())
    rec["arm_qstar"].append(env._arm_q_target.cpu().numpy())
    rec["hand_qstar"].append(env._syn_target.cpu().numpy())
    rec["palm_pos"].append((env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins).cpu().numpy())
    rec["palm_quat"].append(env.robot.data.body_quat_w[:, env.palm_idx].cpu().numpy())
    rec["tips"].append((env.robot.data.body_pos_w[:, env._tip_ids_t] - env.scene.env_origins.unsqueeze(1)).cpu().numpy())
    rec["ep_len"].append(env.episode_length_buf.cpu().numpy())
    rec["tau"].append(env.robot.data.applied_torque.cpu().numpy())


if hasattr(policy, "reset_states"):
    policy.reset_states()
obs = obs_dict["policy"]
with torch.inference_mode():
    for t in range(args.steps):
        snap(obs)
        o = obs.clamp(-clip, clip)
        a = policy.get_action(o) if replay is None else replay[t].unsqueeze(0).expand(N, -1)
        rec["action"].append(a.cpu().numpy())
        obs_dict, _, term, trunc, _ = env.step(a)
        obs = obs_dict["policy"]
        if bool((term | trunc).any()):
            # 에피소드가 끝나면 env 는 자동 리셋한다 — 다음 기록 행의 ep_len 이 0 이다. LSTM 도 학습처럼 0 으로.
            why = {k: bool(v[0]) for k, v in (("terminated", term), ("truncated", trunc))}
            print(f"[trace] episode ended at step {t} {why}", flush=True)
            rec.setdefault("ended", []).append((t, int(bool(term[0])), int(bool(trunc[0]))))
            if hasattr(policy, "reset_states"):
                policy.reset_states()

ended = rec.pop("ended", [])
meta["episode_ends"] = ended
out = {k: np.stack(v) for k, v in rec.items() if v}
T = min(len(rec["obs"]), len(rec["action"]))
out = {k: v[:T] for k, v in out.items()}
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, **out)
Path(str(args.out).replace(".npz", "_meta.json")).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
print(f"[trace] {task} · {N} env · {T} steps · hand obs order {meta['hand_obs_order']}", flush=True)
print(f"[trace] → {args.out}", flush=True)
app.close()
