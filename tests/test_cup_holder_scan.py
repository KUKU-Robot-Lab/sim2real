"""컵홀더 목 스캔(scripts/calib/cup_holder_scan.py): 자세 고르기, 보정 순서, 원형 중앙값, 거절, dry run. 로봇·카메라 없음."""
import math
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

pytest.importorskip("cv2")
pytest.importorskip("scipy")

import calib_profile as cp  # noqa: E402
import cup_holder_scan as S  # noqa: E402
from cup_holder_pose import Estimate, load_cfg  # noqa: E402
from head_fk_chain import t_base_neck_from_encoder  # noqa: E402

PROFILE = cp.load_profile(S.DEFAULT_PROFILE)
CFG = load_cfg()
K, SIZE = S.nominal_camera()
Z_FK = S.fk_table_z(PROFILE) - CFG.base_bottom_m
#: sim(rh_place_r) 배치. 클램프가 상판 끝에 닿아 x 가 같다
TRUTH = {0: (0.38, 0.083, Z_FK, 0.0), 1: (0.38, -0.034, Z_FK, 0.0), 2: (0.38, -0.153, Z_FK, 0.0)}


@dataclass(frozen=True)
class Reading:
    pan_deg: float
    tilt_deg: float


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    k: np.ndarray
    stamp_s: float = 0.0


class Head:
    """명령 + offset 을 실측으로 돌려준다(off_at 이 주어지면 그 순번만). 이동·복귀를 기록한다."""

    def __init__(self, offset: float = 0.0, off_at: int | None = None, fail: bool = False):
        self.moves, self.homed = [], 0
        self.offset, self.off_at, self.fail = offset, off_at, fail

    def move(self, pan, tilt):
        if self.fail:
            raise AssertionError("dry run 에서 목을 움직였다")
        off = self.offset if self.off_at in (None, len(self.moves)) else 0.0
        self.moves.append((pan, tilt))
        return Reading(pan + off, tilt)

    def go_home(self):
        if self.fail:
            raise AssertionError("dry run 에서 목을 움직였다")
        self.homed += 1
        return Reading(0.0, -20.0)


class Camera:
    def __init__(self):
        self.shots = 0

    def capture(self, n_frames=1):
        self.shots += n_frames
        return [Frame(np.zeros((4, 4, 3), np.uint8), K) for _ in range(n_frames)]


def fake_estimate(truth):
    """영상 대신 T_bc 로 보이는 홀더만 참값(FK 프레임)으로 돌려준다."""
    def est(gray, K_, T_bc, cfg, **kw):
        xy = np.array([p[:2] for p in truth.values()])
        uv, depth = S.marker_corners_px(cfg, K_, T_bc, xy, Z_FK)
        vis = S.visible(uv, depth, SIZE)
        poses = {i: np.array(p, float) for (i, p), v in zip(truth.items(), vis) if v}
        return Estimate(det={}, src={i: "aruco" for i in poses}, free={}, poses=poses, rms={i: 0.5 for i in poses},
                        hits={}, missing=[i for i in truth if i not in poses], shared=cfg.shared)
    return est


# ── 자세 고르기 ──────────────────────────────────────────────────────────
def test_default_plan_uses_only_profile_angles_in_pan_order():
    plan = S.plan_scan(PROFILE, CFG, K, SIZE)
    assert len(plan.poses) >= 3
    assert all(cp.lookup(PROFILE, *p) is not None for p in plan.poses)
    assert all(p[1] == -20.0 for p in plan.poses)
    assert list(plan.poses) == sorted(plan.poses)
    # 닿을 수 있는 만큼(최대 MIN_VIEWS) 다 덮는다, 클램프 줄은 모두 두 번 이상
    assert (plan.views >= np.minimum(plan.reachable, S.MIN_VIEWS)).all()
    row = S.clamp_row(S.holder_grid())
    assert (plan.views[row] >= 2).all()


def test_plan_skips_angles_that_see_no_table_and_keeps_to_the_tilt_row():
    e = cp.AngleEntry
    prof = cp.CalibProfile(name="t", neck_to_camera=PROFILE.neck_to_camera,
                           entries=(e(0.13, -20.0, (0.0, 0.0), n_bolts=40), e(170.2, -20.0, (0.0, 0.0), n_bolts=40),
                                    e(0.13, -30.0, (0.0, 0.0), n_bolts=8)))
    assert S.plan_scan(prof, CFG, K, SIZE).poses == ((0.0, -20.0),)
    both = S.plan_scan(prof, CFG, K, SIZE, tilt_deg=None).poses
    assert (170.0, -20.0) not in both and set(both) <= {(0.0, -20.0), (0.0, -30.0)}


def test_check_poses_refuses_uncalibrated_and_upward_tilt():
    S.check_poses(PROFILE, [(15.0, -20.0), (-30.0, -25.0)])
    with pytest.raises(cp.UncalibratedAngle):
        S.check_poses(PROFILE, [(10.0, -20.0)])
    with pytest.raises(ValueError, match="한계"):
        S.check_poses(PROFILE, [(0.0, -5.0)])
    with pytest.raises(ValueError):
        S.check_poses(PROFILE, [])


def test_travel_counts_the_return_home():
    assert S.travel_deg([(-45.0, -20.0), (45.0, -20.0)]) == (180.0, 0.0)


# ── 보정 ─────────────────────────────────────────────────────────────────
def test_camera_is_fk_times_profile_hand_eye():
    T = S.T_base_cam(PROFILE, 15.117, -20.0)
    assert np.allclose(T, t_base_neck_from_encoder(15.117, -20.0) @ PROFILE.neck_to_camera, atol=1e-9)


def test_correction_follows_object_pose_node_order():
    """+ bias -> anchor -> snap_z(PoseConverter.convert). 회전이 커서 bias 를 anchor 뒤에 더하면 값이 달라진다."""
    prof = cp.CalibProfile(name="t", neck_to_camera=PROFILE.neck_to_camera,
                           entries=(cp.AngleEntry(15.0, -20.0, (0.01, -0.005)),),
                           anchor=cp.FrameAnchor(rot_deg=3.0, trans_xy_m=(0.01, -0.02), dz_m=-0.003),
                           known_z=cp.KnownZ(0.205, 0.015))
    pose = np.array([0.38, 0.05, 0.236, 0.1])
    out = S.correct(prof, CFG, pose, 15.2, -20.1)
    # object_pose_node 의 줄을 그대로 따라 한 값
    pos = cp.anchor_position(prof, pose[:3] + cp.bias_for(prof, 15.2, -20.1))
    pos[2] = cp.snap_z(prof, pos[2], -CFG.base_bottom_m)
    assert np.allclose(out[:3], pos, atol=1e-12)
    # 손 계산: bias 먼저, 그다음 3 deg 회전 + 이동, z 는 상판 + 30 mm 로
    c, s = math.cos(math.radians(3.0)), math.sin(math.radians(3.0))
    x, y = 0.38 + 0.01, 0.05 - 0.005
    assert np.allclose(out[:3], [c * x - s * y + 0.01, s * x + c * y - 0.02, 0.235], atol=1e-12)
    wrong = cp.anchor_position(prof, pose[:3]) + cp.bias_for(prof, 15.2, -20.1)
    assert np.linalg.norm(out[:2] - wrong[:2]) > 4e-4
    assert out[3] == pytest.approx(0.1 + math.radians(3.0))


def test_fk_table_height_lands_on_known_table_after_anchor():
    out = S.correct(PROFILE, CFG, (0.38, 0.0, Z_FK, 0.0), 0.13, -20.0)
    assert out[2] == pytest.approx(0.235, abs=1e-12)
    assert cp.anchor_position(PROFILE, np.array([0.38, 0.0, Z_FK]))[2] == pytest.approx(0.235, abs=1e-12)
    assert out[3] == pytest.approx(math.radians(PROFILE.anchor.rot_deg))


def test_correction_refuses_uncalibrated_angle_even_if_not_strict():
    loose = cp.CalibProfile(name="t", neck_to_camera=PROFILE.neck_to_camera,
                            entries=(cp.AngleEntry(0.0, -20.0, (0.0, 0.0)),), strict=False)
    with pytest.raises(cp.UncalibratedAngle):
        S.correct(loose, CFG, (0.38, 0.0, 0.236, 0.0), 2.0, -20.0)


# ── 모으기 ───────────────────────────────────────────────────────────────
def test_circular_median_crosses_pi():
    m = S.circular_median(np.radians([179.0, -179.0, 178.0]))
    assert abs(S.wrap(m - math.radians(179.0))) < 1e-9
    assert S.circular_median([0.1, 0.2, 0.9]) == pytest.approx(0.2)


def _obs(i, shot, x, y, yaw=0.0):
    return S.Obs(i, shot, np.array([x, y, 0.238, yaw]), "aruco", 0.5)


def test_aggregate_drops_outlier_and_takes_median():
    obs = [_obs(0, 0, 0.380, 0.080), _obs(0, 1, 0.381, 0.082), _obs(0, 2, 0.382, 0.081), _obs(0, 3, 0.380, 0.120),
           _obs(0, 4, 0.381, 0.081, yaw=math.radians(20.0))]
    r = S.aggregate(obs, 0)
    assert (r.n_obs, r.n_used, r.n_shots) == (5, 3, 3)
    assert np.allclose(r.pose[:2], [0.381, 0.081])


def test_aggregate_refuses_too_few_poses():
    r = S.aggregate([_obs(1, 0, 0.38, 0.0), _obs(1, 0, 0.38, 0.0005)], 1)
    assert r.pose is None and "자세 1 곳" in r.reason
    assert S.aggregate([], 2).reason == "한 번도 못 봤다"


def test_shared_axis_is_aligned_or_refused():
    out, spread, errs = S.apply_shared({0: np.array([0.380, 0.1, 0.2, 0.0]), 1: np.array([0.384, 0.0, 0.2, 0.0])}, ("x",))
    assert not errs and out[0][0] == out[1][0] == pytest.approx(0.382) and spread["x"] == pytest.approx(0.004)
    _, _, errs = S.apply_shared({0: np.array([0.38, 0.1, 0.2, 0.0]), 1: np.array([0.40, 0.0, 0.2, 0.0])}, ("x",))
    assert errs


# ── 실행 흐름(가짜 목·카메라·추정) ───────────────────────────────────────
def test_dry_run_does_not_move(tmp_path, capsys):
    out = tmp_path / "holders.yaml"
    head = Head(fail=True)
    assert S.main(["--out", str(out)], head=head, camera=Camera()) == 0
    assert head.moves == [] and head.homed == 0 and not out.exists()
    assert "움직이지 않았다" in capsys.readouterr().out


def test_execute_writes_three_holders_in_fpp_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "estimate", fake_estimate(TRUTH))
    out, head, cam = tmp_path / "holders.yaml", Head(), Camera()
    assert S.main(["--execute", "--nice", "0", "--out", str(out)], head=head, camera=cam) == 0
    assert head.moves == list(S.plan_scan(PROFILE, CFG, K, SIZE).poses) and head.homed == 1
    holders = yaml.safe_load(out.read_text())["holders"]
    assert [h["marker_id"] for h in holders.values()] == [0, 1, 2]
    for h in holders.values():
        want = S.to_fpp_frame(PROFILE, CFG, TRUTH[h["marker_id"]])
        assert np.allclose(h["position"], want[:3], atol=0.004)
        assert h["position"][2] == pytest.approx(0.235)
        assert h["yaw_rad"] == pytest.approx(math.radians(PROFILE.anchor.rot_deg), abs=1e-3)
    # 공유 축 x 는 FK 프레임에서 같게 맞춘 뒤 anchor 로 돌린 값
    assert len({h["position"][0] for h in holders.values()}) == 3


def test_partial_result_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "estimate", fake_estimate({i: TRUTH[i] for i in (0, 1)}))
    out, head = tmp_path / "holders.yaml", Head()
    assert S.main(["--execute", "--nice", "0", "--out", str(out)], head=head, camera=Camera()) == 1
    assert not out.exists() and head.homed == 1


def test_single_pose_is_too_few_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "estimate", fake_estimate(TRUTH))
    out = tmp_path / "holders.yaml"
    assert S.main(["--execute", "--nice", "0", "--poses", "0,-20", "--frames", "3", "--out", str(out)],
                  head=Head(), camera=Camera()) == 1
    assert not out.exists()


def test_uncalibrated_measured_angle_drops_that_pose(monkeypatch):
    monkeypatch.setattr(S, "estimate", fake_estimate(TRUTH))
    poses = list(S.plan_scan(PROFILE, CFG, K, SIZE).poses)
    cam = Camera()
    log = S.scan(poses, Head(offset=1.5, off_at=2), cam, PROFILE, CFG, log=lambda *_: None)
    assert len(log.skipped) == 1 and "보정 표 밖" in log.skipped[0]
    assert 2 not in {o.shot for o in log.obs} and cam.shots == len(poses) - 1


def test_all_poses_uncalibrated_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "estimate", fake_estimate(TRUTH))
    out, head = tmp_path / "holders.yaml", Head(offset=2.0)
    assert S.main(["--execute", "--nice", "0", "--out", str(out)], head=head, camera=Camera()) == 1
    assert not out.exists() and head.homed == 1


def test_head_failure_returns_home_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "estimate", fake_estimate(TRUTH))

    class Broken(Head):
        def move(self, pan, tilt):
            raise RuntimeError("정착 실패")

    out, head = tmp_path / "holders.yaml", Broken()
    assert S.main(["--execute", "--nice", "0", "--out", str(out)], head=head, camera=Camera()) == 2
    assert not out.exists() and head.homed == 1


def test_cli_poses_outside_profile_are_refused_before_moving(tmp_path):
    head = Head()
    assert S.main(["--execute", "--nice", "0", "--poses", "10,-20", "--out", str(tmp_path / "h.yaml")],
                  head=head, camera=Camera()) == 2
    assert head.moves == [] and head.homed == 0


def test_board_markers_lying_on_the_table_are_told_apart_from_standing_holder_markers():
    """★10.20 실기 스캔: 상판의 ChArUco 보정 보드가 홀더와 같은 사전 · 같은 id 0~2 라 홀더 id 가 두세 번 보였다.
    스캔 자세(tilt −20 줄) · 홀더가 놓일 수 있는 모든 칸 · yaw ±10° 의 서 있는 홀더 마커는 누운 것으로 보이지 않고,
    상판 위 같은 칸에 누운 22 mm 마커(보드)는 어느 방향이든 누운 것으로 보인다."""
    from cup_holder_pose import T_planar, on_table_square

    table = S.fk_table_z(PROFILE)
    h = CFG.marker_size / 2
    sq = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    n_stand = n_lie = 0
    for pan in (-45.0, -30.0, 0.0, 15.0, 30.0, 45.0):
        T_bc = S.T_base_cam(PROFILE, pan, -20.0)
        for x, y in S.holder_grid():
            for yaw in (-10.0, 0.0, 10.0):
                T = T_planar(x, y, Z_FK, math.radians(yaw))
                uv, depth = S.project(T_bc, K, CFG.corners_stl @ T[:3, :3].T + T[:3, 3])
                if S.visible(uv[None], depth[None], SIZE).all():
                    n_stand += 1
                    assert not on_table_square(uv, K, T_bc, table), (pan, x, y, yaw)
                F = T_planar(x, y, table, math.radians(yaw * 4))
                uv, depth = S.project(T_bc, K, sq @ F[:3, :3].T + F[:3, 3])
                if S.visible(uv[None], depth[None], SIZE).all():
                    n_lie += 1
                    assert on_table_square(uv, K, T_bc, table), (pan, x, y, yaw)
    assert n_stand > 500 and n_lie > 500
