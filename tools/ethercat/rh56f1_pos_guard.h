/* RH56F1 위치(angle_set) 축 보호 — 마스터 rh56f1_ecat_master.c 가 매 주기 부른다.
 *
 * ★10.14 사용자: "힘 1,300 g 을 넘거나 전류 750 mA 를 넘으면 s2r 할 때 멈추게 하는 게 맞다".
 * 배경: 위치 경로(angle_set)에는 어드민턴스의 current_hold(650 mA, angle_target 축 전용)가 없다. 10.06 실측 모드 0 은
 * FORCESET 을 무시하고 접촉 뒤 1.1~1.85 kg · 1 A 넘게 누른다(motion_acq docs/RH56F1_HAND_TUNING.md).
 *
 * 법칙(축마다): 원시 force_actual > force_g 또는 current_actual > current_ma 가 hold_s 이상 이어지면 그 축을 지금
 * 각도에 고정한다(더 밀지 않는다 — 리드 스크루가 전류 없이 쥔 자리를 붙든다). 고정하는 순간 목표가 민 쪽(dir)을 기억하고,
 * 목표가 고정 각도에서 반대쪽(놓는 쪽)으로 가면 풀어 목표를 그대로 보낸다. 민 쪽 목표는 계속 고정 각도로 바꾼다.
 * 10.13 rg3 bag 7 개(어드민턴스 경로)에서 두 문턱을 0.3 s 넘게 넘은 구간은 0 회 — 빈 공간 오작동 근거 없음.
 * force_g · current_ma 0 = 그 문턱 끔, 둘 다 0 = 보호 없음(10.14 전 동작).
 */
#ifndef RH56F1_POS_GUARD_H
#define RH56F1_POS_GUARD_H

typedef struct {
  double force_g;     /* 원시 force_actual 문턱 [g] (0 = 안 씀) */
  double current_ma;  /* current_actual 문턱 [mA] (0 = 안 씀) */
  double hold_s;      /* 이만큼 이어져야 고정(순간 충돌 전류는 넘긴다) */
} pos_guard_params_t;

typedef struct {
  double over_s[6];   /* 문턱을 넘은 채 이어진 시간 */
  int pinned[6];      /* 고정 중 */
  int pin[6];         /* 고정 각도(레지스터) */
  int dir[6];         /* 고정할 때 목표가 민 쪽: -1 = 레지스터 ↓(네 손가락 · 엄지 굽힘은 닫힘) · +1 */
  long trips[6];      /* 고정한 횟수(로그) */
} pos_guard_state_t;

static inline int pos_guard_on(const pos_guard_params_t *p) { return p->force_g > 0.0 || p->current_ma > 0.0; }

/* 축을 위치 제어로 새로 받거나 명령 전(hold)이면 부른다 */
static inline void pos_guard_reset(pos_guard_state_t *s, int i) {
  s->over_s[i] = 0.0;
  s->pinned[i] = 0;
}

/* 보낼 각도를 돌려준다. 고정이 새로 걸린 주기에는 *tripped = 1. */
static inline int pos_guard_step(const pos_guard_params_t *p, pos_guard_state_t *s, int i, double dt, int target, int actual,
                                 double force_g, double current_ma, int *tripped) {
  if (tripped) *tripped = 0;
  if (!pos_guard_on(p)) return target;
  int over = (p->force_g > 0.0 && force_g > p->force_g) || (p->current_ma > 0.0 && current_ma > p->current_ma);
  s->over_s[i] = over ? s->over_s[i] + dt : 0.0;
  if (!s->pinned[i] && s->over_s[i] >= p->hold_s) {
    s->pinned[i] = 1;
    s->pin[i] = actual;
    s->dir[i] = target > actual ? 1 : -1;   /* 같으면 닫힘 쪽으로 본다 */
    s->trips[i]++;
    if (tripped) *tripped = 1;
  }
  if (!s->pinned[i]) return target;
  if ((target - s->pin[i]) * s->dir[i] < 0) {   /* 놓는 쪽으로 — 풀고 목표 그대로 */
    pos_guard_reset(s, i);
    return target;
  }
  return s->pin[i];
}

#endif
