/* RH56F1 손가락별 어드민턴스 — 마스터 주기(500 Hz)마다, 레지스터 단위. 2026-10-06.
 *
 * 기존 위치 제어(/hand_<s>/angle_set → CMD_ANGLE)는 그대로다. 이 제어는 따로 연 입력
 * /hand_<s>/angle_target(CMD_ANGLE_ADM)으로 받은 축에만 걸린다 — 원격조작 · 정책이 같은 로직을 쓴다.
 * 정본 계약 · Python 참조 구현: robot_control components/rh56f1.yaml (control.admittance) · robot_control.rh56f1_hand
 * — 마스터는 시작 때 계약 값을 --adm 으로 받는다. 아래 ADM_DEFAULTS 는 계약과 같아야 한다(sim2real 테스트가 대조).
 *
 * 레지스터는 닫을수록 작아진다(네 손가락 1740 → 900, 엄지 굽힘 1350 → 1100). 축마다 매 주기:
 *   f   = max(힘 - 쉼 값 - deadband, 0)            (손끝 촉각이 조용하면 / proximal_scale: 1 번 링크 접촉)
 *   y  += a (f / k + max(f - f_max, 0) / k_over - y),   a = dt / (tau + dt)   (tau: 접촉 중 · 풀린 뒤)
 *         단 접촉 중 |f - k y| < hold_band 면 y 그대로(손이 3~5 칸씩 움직여 생기는 출렁임을 막는다)
 *   cmd = 목표 + y                                       (y ≥ 0: 힘만큼 연다)
 *   f > rate_on 이 되면 cmd 는 그때의 실제 각도에서 다시 출발해 초당 rate (1 - f / f_max) 칸까지만 닫힌다(여는 쪽은
 *   바로) — 힘과 y 가 다 빠질 때까지. 10.06: 실제 각도 대비 앞서기 상한(11 칸)은 굳은 컵에서 들어간 깊이와 상관없이
 *   ~350 g 에 묶였다(손가락을 지난 명령 1 칸 ≈ 55 g).
 * 빈 공간에서는 f ≈ 0 이라 목표를 그대로 따른다. 접촉하면 쥐는 힘 ≈ k x (목표가 접촉점을 지난 칸 수).
 * 10.06 실측: 모드 0 위치 오프셋만으로는 물체 · 링크마다 3~55 g/칸, 1.1~1.85 kg 포화
 * (motion_acq docs/RH56F1_HAND_TUNING.md). 손 펌웨어는 모드 0 그대로다(OP 중 모드 전환은 손이 받지 않는다).
 */
#ifndef RH56F1_ADMITTANCE_H
#define RH56F1_ADMITTANCE_H
#include <math.h>

typedef struct {
  double k_g_per_reg;      /* 쥐는 힘 / 목표가 접촉점을 지난 칸 수 */
  double deadband_g;
  double tau_contact_s;
  double tau_release_s;
  double f_max_g;          /* 넘으면 k_over 로 훨씬 빨리 연다(부드러운 상한) */
  double k_over_g_per_reg;
  double rate_on_g;        /* deadband 위 이 힘부터 닫히는 속도를 묶는다(빈 공간 빠른 닫기 50~60 g 는 접촉 아님) */
  double rate_reg_s;       /* 접촉 중 닫히는 속도(칸/s) x (1 - f / f_max) */
  double max_offset_reg;
  double proximal_scale;   /* 0~1, 손끝 촉각 < tip_on 이면 f /= proximal_scale */
  double tip_on_counts;    /* 손끝 법선 촉각(0.01 N 단위) */
  double hold_band_g;      /* |f - k y| 가 이 안이면 y 를 그대로 둔다(손 위치 분해능 사이를 오가며 출렁이지 않게) */
  int joints[6];           /* 슬롯 순서(새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전) — 1 = 이 제어를 쓸 수 있다 */
} adm_params_t;

typedef struct {
  double y[6];             /* 레지스터, ≥ 0 */
  double f[6];             /* 쓴 힘(g) */
  double bias[6];          /* 쉼 값(g) */
  double lim[6];           /* 속도 제한 중 지난 주기의 cmd */
  int limiting[6];
} adm_state_t;

/* 기본값 — 10.06 오른손 검지 단독 시험 뒤(컵 1 칸 ≈ 100 g). 실기 튜닝 전. */
static const adm_params_t ADM_DEFAULTS = {
    .k_g_per_reg = 3.6,           /* ≈ 2000 g/rad (네 손가락 550 칸/rad) */
    .deadband_g = 40.0,
    .tau_contact_s = 0.3,
    .tau_release_s = 0.15,
    .f_max_g = 800.0,
    .k_over_g_per_reg = 0.36,     /* ≈ 200 g/rad */
    .rate_on_g = 60.0,
    .rate_reg_s = 165.0,          /* ≈ 0.3 rad/s */
    .max_offset_reg = 880.0,      /* 손가락 전 범위 */
    .proximal_scale = 1.0,
    .tip_on_counts = 20.0,        /* 0.2 N */
    .hold_band_g = 100.0,         /* 10.06 오른손 검지 + 컵: 손이 3~5 칸씩 움직여 80 <-> 300 g 를 오갔다 */
    .joints = {1, 1, 1, 1, 1, 0}, /* 엄지 회전: 하중 때 힘 부호가 반대(10.06) — 위치 제어만 */
};

/* 한 축 한 주기. target · actual = 레지스터, force = g, tip = 손끝 법선(0.01 N, 모르면 -1). 반환 = 보낼 레지스터. */
static inline double adm_step(const adm_params_t *p, adm_state_t *s, int i, double dt, double target, double actual,
                              double force, double tip) {
  double f = force - s->bias[i] - p->deadband_g;
  if (f < 0) f = 0;
  if (f > 0 && tip >= 0 && tip < p->tip_on_counts) f /= p->proximal_scale;
  double goal = f / p->k_g_per_reg + (f > p->f_max_g ? (f - p->f_max_g) / p->k_over_g_per_reg : 0.0);
  if (goal > p->max_offset_reg) goal = p->max_offset_reg;
  if (f > 0 && fabs(f - p->k_g_per_reg * s->y[i]) < p->hold_band_g && f <= p->f_max_g) goal = s->y[i];
  double tau = f > 0 ? p->tau_contact_s : p->tau_release_s;
  double a = tau <= 0 ? 1.0 : dt / (tau + dt);
  s->y[i] += a * (goal - s->y[i]);
  if (s->y[i] < 0) s->y[i] = 0;
  s->f[i] = f;
  double cmd = target + s->y[i];
  if (!s->limiting[i] && f > p->rate_on_g) { s->limiting[i] = 1; s->lim[i] = actual; }
  if (s->limiting[i]) {
    double rate = 1.0 - f / p->f_max_g;
    if (rate < 0) rate = 0;
    double lim = s->lim[i] - p->rate_reg_s * rate * dt;
    if (cmd < lim) cmd = lim;
    s->lim[i] = cmd;
    if (f <= 0 && s->y[i] < 1.0) s->limiting[i] = 0;
  }
  return cmd;
}

#endif
