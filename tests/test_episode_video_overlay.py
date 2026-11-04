"""episode_video_overlay — 에피소드 bag(arm · camera)의 머리 카메라 영상에 FP++ 자세 · 손바닥 · 넘김을 겹친다(11.04 사용자).
순수 로직(구간 · 넘김 · 목 FK 보정 · 투영 · 관측 칸)과, 가짜 bag 두 개로 PNG · mp4 가 나오는지 본다."""
from __future__ import annotations

import json

import numpy as np
import pytest

import episode_video_overlay as V

EXT = V.DEFAULT_EXTRINSICS


def test_node_segment_takes_the_last_run_and_names_the_nodes_it_saw():
    rows = [(1, {"node": "grasp_blue"}), (2, {"node": "grasp_shaker"}), (3, {"node": "grasp_shaker"}),
            (4, {"node": "carry"}), (5, {"node": "grasp_shaker"}), (6, {"node": "grasp_shaker"})]
    assert V.node_segment(rows, "grasp_shaker") == (5, 6)
    with pytest.raises(V.OverlayError, match="grasp_blue"):
        V.node_segment(rows, "nope")


def test_handoff_is_the_approach_to_policy_switch_once_per_episode():
    st = lambda ep, ph: {"episode": ep, "approach_grasp": {"phase": ph}}   # noqa: E731
    rows = [(1, st(1, "approach")), (2, st(1, "approach")), (3, st(1, "policy")), (4, st(1, "policy")),
            (5, st(2, "policy")),                         # 새 에피소드가 policy 로 시작 = 넘김 아님(앞 approach 가 없다)
            (6, st(3, "approach")), (7, st(3, "policy"))]
    assert V.handoff_events(rows, "left") == [(3, "HANDOFF L"), (7, "HANDOFF L")]


def test_head_fk_correction_is_identity_at_the_calibration_pose_and_moves_the_camera_otherwise():
    T_cal = np.asarray(V._base_from_camera(EXT), float)
    q_cal = V.calib_head_urdf_deg(EXT)
    assert np.allclose(V.head_corrected(T_cal, q_cal, q_cal), T_cal, atol=1e-9)
    assert np.allclose(V.head_corrected(T_cal, q_cal, None), T_cal)
    moved = V.head_corrected(T_cal, q_cal, (q_cal[0] + 5.0, q_cal[1]))
    ang = np.degrees(np.arccos((np.trace(moved[:3, :3].T @ T_cal[:3, :3]) - 1) / 2))
    assert 4.0 < ang < 6.0                              # 목 pan 5° → 카메라도 약 5° 돈다


def test_a_point_on_the_optical_axis_lands_on_the_principal_point():
    T = np.asarray(V._base_from_camera(EXT), float)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    p = T[:3, 3] + 0.6 * T[:3, 2]                       # 광학축(카메라 z) 위 0.6 m
    behind = T[:3, 3] - 0.6 * T[:3, 2]
    uv = V.project(K, np.zeros(5), T, np.vstack([p, behind]))
    assert np.allclose(uv[0], [640, 360], atol=1e-6) and np.isnan(uv[1]).all()


def test_palm_cup_and_tips_come_from_the_rh_aglt_actor_layout():
    sl = V.obs_slices()
    o = np.zeros(94)
    o[sl["palm_pos"]] = [0.1, 0.3, 0.45]
    o[sl["cup_rel_palm"]] = [0.05, 0.0, -0.1]
    o[sl["tips_rel_cup"]] = np.tile([0.0, 0.01, 0.0], 5)
    h = V.palm_from_obs(o, sl)
    assert np.allclose(h["palm"], [0.1, 0.3, 0.45]) and np.allclose(h["cup"], [0.15, 0.3, 0.35])
    assert np.allclose(h["tips"], np.tile([0.15, 0.31, 0.35], (5, 1)))
    assert V.palm_from_obs(np.zeros(40), sl) is None   # 다른 정책의 짧은 관측은 안 쓴다


# ---------------------------------------------------------------------------- 가짜 bag 으로 끝까지
def _bag(path, rows):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    made = set()
    for topic, typ, t, msg in sorted(rows, key=lambda r: r[2]):
        if topic not in made:
            w.create_topic(rosbag2_py.TopicMetadata(name=topic, type=typ, serialization_format="cdr"))
            made.add(topic)
        w.write(topic, serialize_message(msg), t)
    del w


def _stamp(msg, t):
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(t, 1_000_000_000)
    return msg


def test_overlay_writes_one_png_per_frame_and_an_mp4(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
    from std_msgs.msg import Float64MultiArray, String

    S = 1_000_000_000
    base = 1_793_760_000 * S
    T = np.asarray(V._base_from_camera(EXT), float)
    shaker = T[:3, 3] + 0.6 * T[:3, 2]                  # 화면 가운데
    q_cal = V.calib_head_urdf_deg(EXT)
    arm, cam = [], []
    for k in range(20):                                 # 실행기 2 Hz: 0~4 s grasp_blue, 5~9 s grasp_shaker
        t = base + k * S // 2
        node = "grasp_shaker" if k >= 10 else "grasp_blue"
        arm.append(("/policy_control/status/episode_runner", "std_msgs/msg/String", t, String(data=json.dumps({"node": node}))))
    for k in range(60):                                 # aglt 10 Hz, 6.0 s 에 넘김
        t = base + 5 * S + k * S // 10
        s = {"episode": 1, "phase": "running", "approach_grasp": {"phase": "policy" if k >= 10 else "approach"}}
        arm.append(("/policy_control/status/rh_aglt_node_left", "std_msgs/msg/String", t, String(data=json.dumps(s))))
        obs = np.zeros(94)
        obs[V.obs_slices()["palm_pos"]] = shaker + [0.0, 0.05, 0.0]
        arm.append(("/policy_control/left/obs", "std_msgs/msg/Float64MultiArray", t, Float64MultiArray(data=obs.tolist())))
        p = _stamp(PoseStamped(), t - 50_000_000)
        p.header.frame_id = "base_link"
        p.pose.position.x, p.pose.position.y, p.pose.position.z = (float(v) for v in shaker)
        p.pose.orientation.w = 1.0
        arm.append(("/objects/shaker_c_orange/live_pose", "geometry_msgs/msg/PoseStamped", t, p))
    ok, jpg = cv2.imencode(".jpg", np.full((720, 1280, 3), 80, np.uint8))
    info = CameraInfo(width=1280, height=720, k=[900.0, 0.0, 640.0, 0.0, 900.0, 360.0, 0.0, 0.0, 1.0], d=[0.0] * 5)
    cam.append(("/camera/camera/color/camera_info", "sensor_msgs/msg/CameraInfo", base, info))
    for k in range(100):                                # 영상 10 Hz 0~10 s
        t = base + k * S // 10
        img = _stamp(CompressedImage(format="rgb8; jpeg compressed bgr8", data=jpg.tobytes()), t - 30_000_000)
        cam.append(("/camera/camera/color/image_raw/compressed", "sensor_msgs/msg/CompressedImage", t + 30_000_000, img))
        js = JointState(name=["head_j_pan", "head_j_tilt"], position=[float(np.radians(v)) for v in q_cal])
        cam.append(("/head/joint_states", "sensor_msgs/msg/JointState", t, js))
    _bag(tmp_path / "ep" / "arm", arm)
    _bag(tmp_path / "ep" / "camera", cam)

    assert V.main([str(tmp_path / "ep"), "--node", "grasp_shaker", "--pad", "0.5"]) == 0
    out = tmp_path / "ep" / "overlay_grasp_shaker"
    pngs = sorted((out / "frames").glob("*.png"))
    assert 50 <= len(pngs) <= 62 and (out / "overlay.mp4").stat().st_size > 0
    # 넘김 시각 프레임에는 빨간 글자가 있고, 화면 가운데 근처에 물체 축(빨강 · 초록 · 파랑)이 그려진다
    late = cv2.imread(str(pngs[len(pngs) // 2]))
    cx = late[300:420, 580:700]
    assert (cx != 80).any()
    assert V.main([str(tmp_path / "ep"), "--node", "nope"]) == 2
