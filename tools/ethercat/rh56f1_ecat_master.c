/* RH56F1 손 EtherCAT 마스터 (SOEM v1.4.0) — 손 하나 · NIC 하나 · 프로세스 하나. 2026-10-02.
 *
 *   rh56f1_ecat_master --ifname enp6s0 --master-sock /tmp/x_m.sock --node-sock /tmp/x_n.sock
 *                      [--hz 1000] [--state-hz 100] [--speed 2000] [--force 600] [--enable-value 1]
 *                      [--hb-timeout-ms 500] [--no-op]
 *
 * ROS 는 이 프로세스에 없다 — setcap(cap_net_raw) 실행 파일은 LD_LIBRARY_PATH 를 무시해서 ROS 라이브러리를 못 읽는다.
 * ROS 쪽은 policy_control/rh56f1_ecat_node.py 가 이 프로세스를 자식으로 띄우고 유닉스 데이터그램 소켓으로 주고받는다
 * (형식은 policy_control/rh56f1_ecat.py 와 같아야 한다: 상태 RHS1 · 명령 RHC1).
 *
 * 안전 규칙
 *  · 손이 첫 각도 명령을 받기 전(hold)에는 매 주기 ANGLESET = 지금 ANGLEACT(범위로 자름) · ENABLE_SET = 0 — 손이 제자리.
 *  · 명령의 -1 은 그 축 직전 목표를 유지(직전이 없으면 지금 각도). 범위 밖 값은 매뉴얼 범위로 자른다.
 *  · 노드 소식(명령 · 하트비트)이 hb-timeout 넘게 없으면 hold 로 되돌아간다. 3 s 넘게 없거나 부모가 죽으면 끝낸다.
 *  · SIGTERM/SIGINT: hold 로 50 주기 → INIT → 종료.
 *  · --no-op: SAFE_OP 에 머문다(손은 출력을 쓰지 않는다) — 상태만 읽는 점검용.
 *  · 손 보호 설정(SDO 0x2000, 매뉴얼 표 52 · 2.5.7 · 2.5.21) — PREOP 에서, OP 전에:
 *      --clear-error           0x2000:03 = 1 (막힘 · 과전류 · 이상 · 통신 고장을 지운다. 과열은 식어야 풀린다)
 *      --current-limit a,..,f  0x2000:07~0C 손가락별 전류 보호(mA). 넘으면 손이 그 손가락을 세운다(상태 5)
 *      --finger-mode a,..,f    0x2000:1B~20 0 속도 · 힘 보호 · 1 힘 폐루프 · 2 임피던스
 *      --force-calibrate       0x2000:06 = 1 힘 센서 영점 보정(빈손, 6 s 동안 손가락이 움직인다)
 *    값 순서는 PDO 와 같다(새끼 · 약지 · 중지 · 검지 · 엄지 굽힘 · 엄지 회전), -1 = 그 축은 쓰지 않는다.
 *    쓰고 나면 다시 읽어 같아야 OP 로 간다(다르거나 못 쓰면 끝낸다 — 보호 없이 손을 움직이지 않는다).
 *    쓰든 안 쓰든 시작마다 읽어 한 줄로 찍는다: "[master] SDO {json}" (노드가 ecat_status 의 sdo 로 낸다).
 *    10.06 왼손 컵 쥐기: 손가락마다 1.1~1.4 A · 합 6.5 A(매뉴얼 최대 쥠 4.0 A) 뒤 손이 먹통이 됐다 — 이 설정이 그 대책.
 *
 * AL 0x1E 우회: 펌웨어 PDO 매핑 항목이 UINT16 이라 SOEM complete access 를 끈다(tools/ethercat/ecat_rh56f1.c 주석).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <fcntl.h>
#include <getopt.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include "ethercat.h"
#include "rh56f1_admittance.h"

#define N_IN 76
#define N_OUT 19
#define EXPECT_ID 0x9252
#define STATE_MAGIC 0x31534852u /* "RHS1" */
#define CMD_MAGIC 0x31434852u   /* "RHC1" */
#define CMD_HEARTBEAT 0
#define CMD_ANGLE 1
#define CMD_FORCE 2
#define CMD_SPEED 3
#define CMD_ENABLE 4
#define CMD_ANGLE_ADM 6  /* /hand_<s>/angle_target — 각도 목표 + 손가락별 어드민턴스(rh56f1_admittance.h) */
#define CMD_MODE 5     /* 손가락 동작 모드(0x2000:1B~20) — OP 중에 SDO 스레드가 쓴다 · -1 = 그대로 */
/* 입력 · 출력 PDO 안 위치 (매뉴얼 표 50) */
#define IN_ANGLE 6
#define IN_FORCE 12
#define IN_CURRENT 18
#define IN_TOUCH 42   /* 손가락 5 개 x (법선 · 접선 · 방향 · 근접 2) — 새끼부터, 법선 = 0.01 N */
#define OUT_ENABLE 0
#define OUT_ANGLE 1
#define OUT_FORCE 7
#define OUT_SPEED 13
#define FLAG_OP 0x01
#define FLAG_ENABLED 0x02
#define FLAG_COMMANDED 0x04
#define FLAG_NODE_OK 0x08
#define FLAG_STOPPING 0x10

#pragma pack(push, 1)
typedef struct {
  uint32_t magic, seq;
  uint64_t t_ns;
  uint16_t al_state, al_code, flags, wkc_bad;
  uint32_t cycles;
  uint16_t rtt_max_us, late_max_us;
  int16_t in[N_IN];
  int16_t out[N_OUT];
} state_msg;
typedef struct {
  uint32_t magic;
  uint16_t kind, n;
  int32_t v[6];
} cmd_msg;
#pragma pack(pop)
_Static_assert(sizeof(state_msg) == 222, "state_msg 크기 — policy_control/rh56f1_ecat.py STATE_SIZE 와 같게");
_Static_assert(sizeof(cmd_msg) == 32, "cmd_msg 크기 — rh56f1_ecat.py CMD_SIZE 와 같게");

/* 매뉴얼 2.5.11 각도 범위: 새끼 · 약지 · 중지 · 검지 900~1740, 엄지 굽힘 1100~1350, 엄지 회전 600~1800 */
static const int ANG_LO[6] = {900, 900, 900, 900, 1100, 600};
static const int ANG_HI[6] = {1740, 1740, 1740, 1740, 1350, 1800};
#define FORCE_MAX 1000 /* g, 매뉴얼 2.5.12 */
#define SPEED_MAX 4000 /* 매뉴얼 2.5.13 */

static volatile sig_atomic_t g_stop = 0;
static char IOmap[4096];

static void on_signal(int s) { (void)s; g_stop = 1; }
static int clampi(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static uint64_t mono_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

typedef struct {
  int commanded;           /* 첫 각도 명령을 받았는가 */
  int hold_enable;         /* --op-enable: 명령 전(hold)에도 ENABLE_SET = enable_value (목표 = 지금 각도) */
  int enable_value;        /* 명령 중 ENABLE_SET 값 */
  int16_t target[6], force[6], speed[6];
  int adm_on[6];           /* 이 축의 목표가 angle_target(어드민턴스)로 왔는가 — angle_set 이 오면 0 */
  int16_t sent[6];         /* 실제로 보낸 각도(어드민턴스를 거친 값) */
} ctl_t;

static adm_params_t g_adm;
static adm_state_t g_adm_state;

static void hold(ctl_t *c, const int16_t *in) {
  for (int i = 0; i < 6; i++) c->sent[i] = c->target[i] = (int16_t)clampi(in[IN_ANGLE + i], ANG_LO[i], ANG_HI[i]);
}

static void write_outputs(ctl_t *c, int16_t *out) {
  out[OUT_ENABLE] = (int16_t)((c->commanded || c->hold_enable) ? c->enable_value : 0);
  for (int i = 0; i < 6; i++) {
    out[OUT_ANGLE + i] = c->sent[i];
    out[OUT_FORCE + i] = c->force[i];
    out[OUT_SPEED + i] = c->speed[i];
  }
}

/* 매 주기 보낼 각도: angle_set 축은 목표 그대로, angle_target 축은 어드민턴스를 거친다. 쉼 값(힘 영점)은
 * 명령 전(hold) 동안 지수 평균으로 잡는다(10.06 영점 보정 뒤 대부분 ±10 g). */
static void control(ctl_t *c, const int16_t *in, double dt) {
  for (int i = 0; i < 6; i++) {
    double force = in[IN_FORCE + i];
    if (!c->commanded) {
      g_adm_state.bias[i] += 0.01 * (force - g_adm_state.bias[i]);
      g_adm_state.y[i] = 0;
      g_adm_state.limiting[i] = g_adm_state.pinned[i] = 0;
      c->sent[i] = c->target[i];
      continue;
    }
    if (!c->adm_on[i]) { c->sent[i] = c->target[i]; continue; }
    int16_t tip_raw = i < 5 ? in[IN_TOUCH + i * 5] : -1;   /* 0xFFFF(안 읽힘) = -1 */
    double cmd = adm_step(&g_adm, &g_adm_state, i, dt, c->target[i], in[IN_ANGLE + i], force, tip_raw < 0 ? -1.0 : tip_raw,
                          (double)in[IN_CURRENT + i]);
    c->sent[i] = (int16_t)clampi((int)(cmd + 0.5), ANG_LO[i], ANG_HI[i]);
  }
}

static void request_modes(const int32_t v[6]);

static void apply_cmd(ctl_t *c, const cmd_msg *m, const int16_t *in) {
  switch (m->kind) {
    case CMD_ANGLE:
    case CMD_ANGLE_ADM:
      for (int i = 0; i < 6; i++) {
        int v = m->v[i];
        if (v < 0) v = c->commanded ? c->target[i] : in[IN_ANGLE + i];   /* -1 = 그 축은 둔다 */
        else {
          int adm = m->kind == CMD_ANGLE_ADM && g_adm.joints[i];
          if (!adm) g_adm_state.y[i] = 0, g_adm_state.limiting[i] = g_adm_state.pinned[i] = 0;   /* 위치 제어로 돌아온 축: 보정 없이 목표 그대로 */
          c->adm_on[i] = adm;
        }
        c->target[i] = (int16_t)clampi(v, ANG_LO[i], ANG_HI[i]);
      }
      if (!c->commanded) printf("[master] 첫 각도 명령 — ENABLE_SET %d\n", c->enable_value);
      c->commanded = 1;
      break;
    case CMD_FORCE:
      for (int i = 0; i < 6; i++) if (m->v[i] >= 0) c->force[i] = (int16_t)clampi(m->v[i], 0, FORCE_MAX);
      break;
    case CMD_SPEED:
      for (int i = 0; i < 6; i++) if (m->v[i] >= 0) c->speed[i] = (int16_t)clampi(m->v[i], 0, SPEED_MAX);
      break;
    case CMD_ENABLE:
      c->enable_value = m->v[0];
      break;
    case CMD_MODE:
      request_modes(m->v);
      break;
    default:
      break;
  }
}

/* AL 상태 · 코드를 slave 1 에서 직접 읽고 쓴다. ec_slave[].state 를 요청 값으로 덮지 않는다
 * (10.02: 덮어쓴 값이 상태로 나가 OP 로 오판했고, 0.1 s 마다 OP 를 다시 요청해 전이가 끝나지 못했다). */
static uint16 read_al(uint16 *code) {
  uint16 st = 0, cd = 0;
  if (ec_FPRD(ec_slave[1].configadr, ECT_REG_ALSTAT, sizeof(st), &st, EC_TIMEOUTRET) <= 0) return 0xFFFF;
  ec_FPRD(ec_slave[1].configadr, ECT_REG_ALSTATCODE, sizeof(cd), &cd, EC_TIMEOUTRET);
  *code = etohs(cd);
  return etohs(st);
}

static void request_al(uint16 state) {
  uint16 v = htoes(state);
  ec_FPWR(ec_slave[1].configadr, ECT_REG_ALCTL, sizeof(v), &v, EC_TIMEOUTRET);
}

/* OP 가 안 될 때 ESC 레지스터를 본다(ET1100/ESC 데이터시트 주소) — 읽기만 */
static void dump_esc(void) {
  static const struct { uint16 adr, len; const char *name; } R[] = {
      {0x0110, 2, "DL status"},       {0x0130, 2, "AL status"},       {0x0134, 2, "AL code"},
      {0x0220, 4, "AL event"},        {0x0810, 8, "SM2(start,len,ctl,st,act,pdi)"},
      {0x0818, 8, "SM3(start,len,ctl,st,act,pdi)"}, {0x0400, 2, "WD divider"}, {0x0410, 2, "WD PDI time"},
      {0x0420, 2, "WD SM time"},      {0x0440, 2, "WD SM status"},    {0x0442, 2, "WD SM counter"},
      {0x0981, 1, "DC activation"},   {0x0980, 1, "DC cyclic unit"}};
  uint8 buf[8];
  printf("[master] ESC 레지스터:");
  for (unsigned i = 0; i < sizeof(R) / sizeof(R[0]); i++) {
    memset(buf, 0, sizeof(buf));
    int w = ec_FPRD(ec_slave[1].configadr, R[i].adr, R[i].len, buf, EC_TIMEOUTRET);
    printf(" | %s 0x%04x=", R[i].name, R[i].adr);
    if (w <= 0) { printf("(못 읽음)"); continue; }
    for (int k = 0; k < R[i].len; k++) printf("%02x", buf[k]);
  }
  printf("\n");
}

static int open_sock(const char *path) {
  int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) return -1;
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  strncpy(a.sun_path, path, sizeof(a.sun_path) - 1);
  unlink(path);
  if (bind(fd, (struct sockaddr *)&a, sizeof(a)) < 0) { close(fd); return -1; }
  return fd;
}

/* -- 손 보호 설정 (SDO 0x2000) ---------------------------------------------------------------- */
#define SDO_IDX 0x2000
#define SDO_CLEAR_ERROR 0x03
#define SDO_FORCE_CALIB 0x06    /* 1 = 손 힘 센서 영점 보정(6 s, 손가락을 폈다 굽힌다 · 빈손이어야 한다) */
#define SDO_CURRENT_LIMIT 0x07   /* ~0x0C, mA */
#define SDO_DEFAULT_SPEED 0x0D   /* ~0x12 */
#define SDO_DEFAULT_FORCE 0x13   /* ~0x18, g */
#define SDO_FINGER_MODE 0x1B     /* ~0x20 */
#define CURRENT_LIMIT_MIN 100    /* 이보다 낮으면 빈손 동작(≤ 320 mA, 10.06)도 멈춘다 */
#define CURRENT_LIMIT_MAX 1500   /* 매뉴얼 2.5.7 */

/* "a,b,c,d,e,f" 또는 값 하나(여섯 축 모두) → v[6]. 범위 밖 · 형식 오류면 0 */
static int parse6(const char *txt, int lo, int hi, int v[6]) {
  int n = 0;
  const char *p = txt;
  while (*p && n < 6) {
    char *end;
    long x = strtol(p, &end, 10);
    if (end == p || (x != -1 && (x < lo || x > hi))) return 0;
    v[n++] = (int)x;
    if (*end == ',') p = end + 1; else if (*end == '\0') p = end; else return 0;
  }
  if (*p) return 0;
  if (n == 1) for (int i = 1; i < 6; i++) v[i] = v[0];
  else if (n != 6) return 0;
  return 1;
}

static int sdo_read16(uint8 sub, int *out) {
  int16 v = 0;
  int sz = sizeof(v);
  if (ec_SDOread(1, SDO_IDX, sub, FALSE, &sz, &v, EC_TIMEOUTRXM) > 0 && sz == (int)sizeof(v)) { *out = v; return 1; }
  *out = INT_MIN;
  return 0;
}

static int sdo_write16(uint8 sub, int value) {
  int16 v = (int16)value;
  return ec_SDOwrite(1, SDO_IDX, sub, FALSE, sizeof(v), &v, EC_TIMEOUTRXM) > 0;
}

static void drain_ec_errors(void) {
  while (EcatError) printf("[master] SDO 오류: %s", ec_elist2string());
}

static int json6(char *buf, size_t n, const char *key, const int v[6]) {
  int k = snprintf(buf, n, "\"%s\": [", key);
  for (int i = 0; i < 6; i++)
    k += (v[i] == INT_MIN) ? snprintf(buf + k, n - k, "%snull", i ? ", " : "") : snprintf(buf + k, n - k, "%s%d", i ? ", " : "", v[i]);
  return k + snprintf(buf + k, n - k, "]");
}

/* 쓰고(요청한 것만) 다시 읽어 찍는다. 요청한 값이 그대로 읽히지 않으면 -1 */
static int sdo_setup(int clear_error, const int cur[6], const int mode[6], int force_calib) {
  int bad = 0;
  /* ★10.06 첫 실기: ec_config_init 직후에는 손이 아직 PREOP 로 넘어가는 중이라 메일박스가 안 열려
   * SDO 가 모두 바로 실패했다(SOEM 은 그때 오류도 남기지 않는다). PREOP 를 기다린 뒤 쓴다. */
  ec_statecheck(1, EC_STATE_PRE_OP, EC_TIMEOUTSTATE);
  ec_readstate();
  printf("[master] SDO 전 상태 AL 0x%02x · 메일박스 %d B · 프로토콜 0x%02x(CoE %s)\n", ec_slave[1].state,
         ec_slave[1].mbx_l, ec_slave[1].mbx_proto, (ec_slave[1].mbx_proto & ECT_MBXPROT_COE) ? "있음" : "없음");
  if (clear_error) {
    int ok = sdo_write16(SDO_CLEAR_ERROR, 1);
    printf("[master] 오류 지우기(0x2000:03) %s\n", ok ? "ok" : "✗ 실패");
    bad |= !ok;
  }
  if (force_calib) {   /* 매뉴얼 2.5.6: 빈손에서 6 s — 다섯 손가락을 펴고, 네 손가락 굽힘 · 폄, 엄지 굽힘 · 폄 */
    int ok = sdo_write16(SDO_FORCE_CALIB, 1);
    printf("[master] 힘 센서 영점 보정(0x2000:06) %s — 7 s 기다린다(손이 움직인다)\n", ok ? "시작" : "✗ 실패");
    bad |= !ok;
    if (ok) for (int k = 0; k < 70 && !g_stop; k++) usleep(100000);
  }
  for (int i = 0; i < 6; i++) {
    if (cur[i] >= 0 && !sdo_write16((uint8)(SDO_CURRENT_LIMIT + i), cur[i])) { printf("[master] ✗ 전류 한계 %d 쓰기 실패\n", i); bad = 1; }
    if (mode[i] >= 0 && !sdo_write16((uint8)(SDO_FINGER_MODE + i), mode[i])) { printf("[master] ✗ 동작 모드 %d 쓰기 실패\n", i); bad = 1; }
  }
  int r_cur[6], r_mode[6], r_speed[6], r_force[6];
  for (int i = 0; i < 6; i++) {
    sdo_read16((uint8)(SDO_CURRENT_LIMIT + i), &r_cur[i]);
    sdo_read16((uint8)(SDO_FINGER_MODE + i), &r_mode[i]);
    sdo_read16((uint8)(SDO_DEFAULT_SPEED + i), &r_speed[i]);
    sdo_read16((uint8)(SDO_DEFAULT_FORCE + i), &r_force[i]);
    if ((cur[i] >= 0 && r_cur[i] != cur[i]) || (mode[i] >= 0 && r_mode[i] != mode[i])) bad = 1;
  }
  drain_ec_errors();
  char line[512];
  int k = snprintf(line, sizeof(line), "{");
  k += json6(line + k, sizeof(line) - k, "current_limit_ma", r_cur);
  k += snprintf(line + k, sizeof(line) - k, ", ");
  k += json6(line + k, sizeof(line) - k, "finger_mode", r_mode);
  k += snprintf(line + k, sizeof(line) - k, ", ");
  k += json6(line + k, sizeof(line) - k, "default_speed", r_speed);
  k += snprintf(line + k, sizeof(line) - k, ", ");
  k += json6(line + k, sizeof(line) - k, "default_force_g", r_force);
  snprintf(line + k, sizeof(line) - k, "}");
  printf("[master] SDO %s\n", line);
  if (bad) printf("[master] ✗ 손 보호 설정이 요청과 다르다(쓰기 실패 또는 다시 읽은 값이 다름) — OP 로 가지 않는다\n");
  return bad ? -1 : 0;
}

/* -- 실행 중 손가락 모드 전환 (10.06: 빈 공간 = 모드 0 위치, 쥐는 동안 = 모드 1 힘 폐루프) ---------- *
 * SDO 는 메일박스로 수 ms 걸려 1 kHz 루프를 막으면 안 되므로 따로 스레드가 쓴다(SOEM 포트는 송수신 ·
 * 인덱스 뮤텍스로 보호된다). 결과는 "[master] MODE {json}" 한 줄 — 노드가 /hand_<s>/finger_mode 로 낸다. */
typedef struct {
  pthread_mutex_t mu;
  pthread_cond_t cv;
  int pending[6], have, quit;
  int applied[6];
} mode_box_t;
static mode_box_t g_mode = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, {-1, -1, -1, -1, -1, -1}, 0, 0,
                            {-1, -1, -1, -1, -1, -1}};

static void print_modes(const int v[6], double ms, int ok) {
  char line[160];
  int k = snprintf(line, sizeof(line), "{");
  k += json6(line + k, sizeof(line) - k, "finger_mode", v);
  snprintf(line + k, sizeof(line) - k, ", \"ms\": %.1f, \"ok\": %s}", ms, ok ? "true" : "false");
  printf("[master] MODE %s\n", line);
}

static void *mode_worker(void *arg) {
  (void)arg;
  for (;;) {
    int want[6];
    pthread_mutex_lock(&g_mode.mu);
    while (!g_mode.have && !g_mode.quit) pthread_cond_wait(&g_mode.cv, &g_mode.mu);
    if (g_mode.quit) { pthread_mutex_unlock(&g_mode.mu); return NULL; }
    memcpy(want, g_mode.pending, sizeof(want));
    g_mode.have = 0;
    for (int i = 0; i < 6; i++) g_mode.pending[i] = -1;
    pthread_mutex_unlock(&g_mode.mu);
    uint64_t t0 = mono_ns();
    int ok = 1, now[6];
    for (int i = 0; i < 6; i++) {
      if (want[i] < 0) continue;
      if (!sdo_write16((uint8)(SDO_FINGER_MODE + i), want[i])) ok = 0;
    }
    for (int i = 0; i < 6; i++) {
      sdo_read16((uint8)(SDO_FINGER_MODE + i), &now[i]);
      if (want[i] >= 0 && now[i] != want[i]) ok = 0;
    }
    drain_ec_errors();
    pthread_mutex_lock(&g_mode.mu);
    memcpy(g_mode.applied, now, sizeof(now));
    pthread_mutex_unlock(&g_mode.mu);
    print_modes(now, (mono_ns() - t0) / 1e6, ok);
  }
}

static pthread_t g_mode_thread;
static int g_mode_thread_ok = 0;

static void stop_mode_worker(void) {
  if (!g_mode_thread_ok) return;
  pthread_mutex_lock(&g_mode.mu);
  g_mode.quit = 1;
  pthread_cond_signal(&g_mode.cv);
  pthread_mutex_unlock(&g_mode.mu);
  pthread_join(g_mode_thread, NULL);   /* SDO 가 진행 중이면 그 한 번(≤ 수 ms, 최악 EC_TIMEOUTRXM)을 마친다 */
  g_mode_thread_ok = 0;
}

static void request_modes(const int32_t v[6]) {
  pthread_mutex_lock(&g_mode.mu);
  for (int i = 0; i < 6; i++)
    if (v[i] >= 0 && v[i] <= 2) g_mode.pending[i] = v[i];   /* 몰린 요청은 합쳐 쓴다 */
  g_mode.have = 1;
  pthread_cond_signal(&g_mode.cv);
  pthread_mutex_unlock(&g_mode.mu);
}

static int parse_adm(const char *txt, adm_params_t *p) {
  double v[19];
  int n = 0;
  const char *q = txt;
  while (*q && n < 19) {
    char *end;
    v[n++] = strtod(q, &end);
    if (end == q) return 0;
    if (*end == ',') q = end + 1; else if (*end == '\0') q = end; else return 0;
  }
  if (n != 19 || *q) return 0;
  adm_params_t a = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12], {0}};
  for (int i = 0; i < 6; i++) a.joints[i] = v[13 + i] != 0;
  if (a.k_g_per_reg <= 0 || a.k_over_g_per_reg <= 0 || a.f_max_g <= 0 || a.max_offset_reg <= 0 ||
      a.deadband_g < 0 || a.tau_contact_s < 0 || a.tau_release_s < 0 || a.rate_on_g < 0 || a.rate_reg_s <= 0 ||
      a.proximal_scale <= 0 || a.proximal_scale > 1 || a.hold_band_g < 0 || a.current_hold_ma <= 0) return 0;
  *p = a;
  return 1;
}

static void usage(const char *p) {
  fprintf(stderr, "usage: %s --ifname IF --master-sock P --node-sock P [--hz 1000] [--state-hz 100] [--speed 2000] "
                  "[--force 600] [--enable-value 1] [--hb-timeout-ms 500] [--no-op] [--op-enable] [--sync-type N] [--op-timeout-ms 3000] [--hz-op 1000] "
                  "[--clear-error] [--current-limit mA[,x6]] [--finger-mode m[,x6]] [--force-calibrate] [--adm 19 values]\n", p);
}

int main(int argc, char **argv) {
  const char *ifname = NULL, *msock = NULL, *nsock = NULL;
  double hz = 1000, state_hz = 100;
  int speed = 2000, force = 600, enable_value = 1, hb_timeout_ms = 500, no_op = 0;
  int op_enable = 0, sync_type = -1, op_timeout_ms = 3000;   /* OP 실험 손잡이(기본 끔) */
  g_adm = ADM_DEFAULTS;
  int force_calib = 0, clear_error = 0, current_limit[6] = {-1, -1, -1, -1, -1, -1}, finger_mode[6] = {-1, -1, -1, -1, -1, -1};
  double hz_op = 0;   /* >0: OP 에 들어간 뒤 이 주기로(10.03 — 1 kHz 로는 OP 전이가 안 되지만 들어간 뒤는 확인 대상) */
  static struct option opts[] = {{"ifname", 1, 0, 'i'}, {"master-sock", 1, 0, 'm'}, {"node-sock", 1, 0, 'n'},
                                 {"hz", 1, 0, 'h'},     {"state-hz", 1, 0, 's'},    {"speed", 1, 0, 'v'},
                                 {"force", 1, 0, 'f'},  {"enable-value", 1, 0, 'e'}, {"hb-timeout-ms", 1, 0, 't'},
                                 {"no-op", 0, 0, 'o'},  {"op-enable", 0, 0, 'E'}, {"sync-type", 1, 0, 'y'},
                                 {"op-timeout-ms", 1, 0, 'T'}, {"hz-op", 1, 0, 'H'},
                                 {"clear-error", 0, 0, 'C'}, {"force-calibrate", 0, 0, 'K'}, {"adm", 1, 0, 'A'}, {"current-limit", 1, 0, 'L'}, {"finger-mode", 1, 0, 'M'},
                                 {0, 0, 0, 0}};
  for (int c; (c = getopt_long(argc, argv, "", opts, NULL)) != -1;) {
    switch (c) {
      case 'i': ifname = optarg; break;
      case 'm': msock = optarg; break;
      case 'n': nsock = optarg; break;
      case 'h': hz = atof(optarg); break;
      case 's': state_hz = atof(optarg); break;
      case 'v': speed = atoi(optarg); break;
      case 'f': force = atoi(optarg); break;
      case 'e': enable_value = atoi(optarg); break;
      case 't': hb_timeout_ms = atoi(optarg); break;
      case 'o': no_op = 1; break;
      case 'E': op_enable = 1; break;
      case 'y': sync_type = atoi(optarg); break;
      case 'T': op_timeout_ms = atoi(optarg); break;
      case 'H': hz_op = atof(optarg); break;
      case 'C': clear_error = 1; break;
      case 'K': force_calib = 1; break;
      case 'A':
        if (!parse_adm(optarg, &g_adm)) {
          fprintf(stderr, "--adm: 숫자 13 개(k deadband tau_c tau_r f_max k_over rate_on rate max_off prox tip_on hold_band current_hold) + joints 6 개\n");
          return 2;
        }
        break;
      case 'L':
        if (!parse6(optarg, CURRENT_LIMIT_MIN, CURRENT_LIMIT_MAX, current_limit)) {
          fprintf(stderr, "--current-limit: %d~%d mA(또는 -1), 하나 또는 여섯 개\n", CURRENT_LIMIT_MIN, CURRENT_LIMIT_MAX);
          return 2;
        }
        break;
      case 'M':
        if (!parse6(optarg, 0, 2, finger_mode)) { fprintf(stderr, "--finger-mode: 0~2(또는 -1), 하나 또는 여섯 개\n"); return 2; }
        break;
      default: usage(argv[0]); return 2;
    }
  }
  if (!ifname || !msock || !nsock || hz < 50 || hz > 4000 || state_hz <= 0 || state_hz > hz ||
      (hz_op != 0 && (hz_op < state_hz || hz_op > 4000))) { usage(argv[0]); return 2; }
  setvbuf(stdout, NULL, _IOLBF, 0);
  prctl(PR_SET_PDEATHSIG, SIGTERM);   /* 노드가 죽으면 같이 끝난다 */
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);
  struct sched_param sp = {.sched_priority = 80};
  if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) printf("[master] SCHED_FIFO 못 씀(%s) — 보통 우선순위로 돈다\n", strerror(errno));
  mlockall(MCL_CURRENT | MCL_FUTURE);

  int fd = open_sock(msock);
  if (fd < 0) { printf("[master] ✗ 소켓 %s: %s\n", msock, strerror(errno)); return 1; }
  struct sockaddr_un node = {.sun_family = AF_UNIX};
  strncpy(node.sun_path, nsock, sizeof(node.sun_path) - 1);

  int rc = 1;
  if (!ec_init(ifname)) { printf("[master] ✗ ec_init(%s) — cap_net_raw(setcap) · 인터페이스 이름 확인\n", ifname); goto out_sock; }
  if (ec_config_init(FALSE) != 1) { printf("[master] ✗ slave %d 개(1 이어야 한다 — 손 하나 · NIC 하나)\n", ec_slavecount); goto out_ec; }
  if (ec_slave[1].eep_id != EXPECT_ID) { printf("[master] ✗ slave ID 0x%x ≠ RH56F1 0x%x\n", ec_slave[1].eep_id, EXPECT_ID); goto out_ec; }
  ec_slave[1].CoEdetails &= ~ECT_COEDET_SDOCA;
  if (sync_type >= 0) {   /* 0x1C32/0x1C33:01 Synchronization Type — PREOP 에서 SDO 로 */
    uint16 v = (uint16)sync_type;
    int w1 = ec_SDOwrite(1, 0x1C32, 0x01, FALSE, sizeof(v), &v, EC_TIMEOUTRXM);
    int w2 = ec_SDOwrite(1, 0x1C33, 0x01, FALSE, sizeof(v), &v, EC_TIMEOUTRXM);
    printf("[master] sync type %d 쓰기 1C32 %s · 1C33 %s\n", sync_type, w1 > 0 ? "ok" : "실패", w2 > 0 ? "ok" : "실패");
  }
  if (sdo_setup(clear_error, current_limit, finger_mode, force_calib) != 0) goto out_ec;   /* PREOP: OP 전에 손 보호 설정 */
  printf("[master] ADM {\"k_g_per_reg\": %g, \"deadband_g\": %g, \"tau_contact_s\": %g, \"tau_release_s\": %g, "
         "\"f_max_g\": %g, \"k_over_g_per_reg\": %g, \"rate_on_g\": %g, \"rate_reg_s\": %g, \"max_offset_reg\": %g, \"proximal_scale\": %g, "
         "\"tip_on_counts\": %g, \"hold_band_g\": %g, \"current_hold_ma\": %g, \"joints\": [%d, %d, %d, %d, %d, %d]} — angle_target 축에만\n",
         g_adm.k_g_per_reg, g_adm.deadband_g, g_adm.tau_contact_s, g_adm.tau_release_s, g_adm.f_max_g,
         g_adm.k_over_g_per_reg, g_adm.rate_on_g, g_adm.rate_reg_s, g_adm.max_offset_reg, g_adm.proximal_scale, g_adm.tip_on_counts,
         g_adm.hold_band_g, g_adm.current_hold_ma,
         g_adm.joints[0], g_adm.joints[1], g_adm.joints[2], g_adm.joints[3], g_adm.joints[4], g_adm.joints[5]);
  g_mode_thread_ok = pthread_create(&g_mode_thread, NULL, mode_worker, NULL) == 0;
  if (!g_mode_thread_ok) printf("[master] ⚠ 모드 전환 스레드를 못 띄움 — 실행 중 모드 전환 불가\n");
  ec_config_map(&IOmap);
  ec_configdc();   /* SOEM simple_test 와 같은 순서(매뉴얼: DC 동기 모드는 없다 — SYNC 는 켜지 않는다) */
  if (ec_slave[1].Obytes != N_OUT * 2 || ec_slave[1].Ibytes != N_IN * 2) {
    printf("[master] ✗ PDO 크기 출력 %d · 입력 %d B (기대 %d · %d)\n", ec_slave[1].Obytes, ec_slave[1].Ibytes, N_OUT * 2, N_IN * 2);
    goto out_init;
  }
  ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);
  ec_readstate();   /* statecheck(0) 은 ec_slave[0] 만 갱신한다 — slave 1 은 다시 읽어야 한다(10.02 'AL 0x0000' 오판) */
  if (ec_slave[1].state != EC_STATE_SAFE_OP) {
    printf("[master] ✗ SAFE_OP 실패 AL 0x%04x %s\n", ec_slave[1].ALstatuscode, ec_ALstatuscode2string(ec_slave[1].ALstatuscode));
    goto out_init;
  }
  int16_t *in = (int16_t *)ec_slave[1].inputs, *out = (int16_t *)ec_slave[1].outputs;
  int expected = ec_group[0].outputsWKC * 2 + ec_group[0].inputsWKC;
  ctl_t c = {.commanded = 0, .hold_enable = op_enable, .enable_value = enable_value};
  for (int i = 0; i < 6; i++) { c.force[i] = (int16_t)clampi(force, 0, FORCE_MAX); c.speed[i] = (int16_t)clampi(speed, 0, SPEED_MAX); }
  memset(out, 0, N_OUT * 2);
  /* 입력이 차도록 몇 주기 돌린 뒤 hold 목표를 잡는다 */
  for (int i = 0; i < 20; i++) { ec_send_processdata(); ec_receive_processdata(EC_TIMEOUTRET); usleep(1000); }
  hold(&c, in);
  write_outputs(&c, out);
  printf("[master] SAFE_OP · %s · 출력 %d B · 입력 %d B · %.0f Hz · 상태 %.0f Hz · 속도 %d · 힘 %d · %s\n", ifname,
         ec_slave[1].Obytes, ec_slave[1].Ibytes, hz, state_hz, speed, force, no_op ? "--no-op(SAFE_OP 유지)" : "OP 요청");
  uint16 al_code = 0, al_now = read_al(&al_code);
  uint64_t op_req_t = 0;
  if (!no_op) { request_al(EC_STATE_OPERATIONAL); op_req_t = mono_ns(); }

  long period = (long)(1e9 / hz);
  int state_every = (int)(hz / state_hz + 0.5);
  double hz_now = hz;
  struct timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  uint64_t last_node = mono_ns(), started = last_node;
  uint32_t cycles = 0, seq = 0;
  uint16_t wkc_bad = 0, rtt_max = 0, late_max = 0;
  int was_op = 0, node_ok = 0, stop_left = -1, consecutive_bad = 0;
  for (;;) {
    next.tv_nsec += period;
    while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    uint64_t t0 = mono_ns();
    uint64_t want = (uint64_t)next.tv_sec * 1000000000ull + next.tv_nsec;
    if (t0 > want) { uint64_t l = (t0 - want) / 1000; if (l > late_max) late_max = l > 65535 ? 65535 : (uint16_t)l; }

    /* 노드 명령 */
    cmd_msg m;
    ssize_t r;
    while ((r = recv(fd, &m, sizeof(m), 0)) > 0) {
      if (r != sizeof(m) || m.magic != CMD_MAGIC) continue;
      last_node = t0;
      if (!node_ok) { node_ok = 1; printf("[master] 노드 연결\n"); }
      if (stop_left < 0) apply_cmd(&c, &m, in);
    }
    uint64_t silent_ms = (t0 - last_node) / 1000000ull;
    if (node_ok && silent_ms > (uint64_t)hb_timeout_ms) {
      node_ok = 0;
      if (c.commanded) printf("[master] ⚠ 노드 소식 %llu ms 없음 — hold\n", (unsigned long long)silent_ms);
      c.commanded = 0;
    }
    if (silent_ms > 3000 && (t0 - started) > 3000000000ull && stop_left < 0) {
      printf("[master] 노드 소식 3 s 없음 — 끝낸다\n");
      g_stop = 1;
    }
    if (g_stop && stop_left < 0) {
      stop_left = 50;
      c.commanded = 0;
      static const int32_t all_position[6] = {0, 0, 0, 0, 0, 0};
      request_modes(all_position);   /* 힘 폐루프로 쥔 손가락도 위치 모드(목표 = 지금 각도)로 두고 끝낸다 */
      printf("[master] 정지 — 모드 0 · hold 후 INIT\n");
    }
    if (!c.commanded) hold(&c, in);
    control(&c, in, 1.0 / hz_now);
    write_outputs(&c, out);

    ec_send_processdata();
    int wkc = ec_receive_processdata(EC_TIMEOUTRET);
    uint64_t t1 = mono_ns();
    uint64_t rtt = (t1 - t0) / 1000;
    if (rtt > rtt_max) rtt_max = rtt > 65535 ? 65535 : (uint16_t)rtt;
    if (wkc < expected) {
      wkc_bad++;
      if (++consecutive_bad == 100) printf("[master] ⚠ WKC %d < %d 가 100 주기 연속 — 케이블 · 손 전원\n", wkc, expected);
    } else {
      consecutive_bad = 0;
    }
    cycles++;

    if (cycles % 100 == 0) {   /* 상태 확인 · OP 요청(한 번 요청하고 기다린다 — 오류 · 3 s 초과 때만 다시) */
      uint16 code = 0, now = read_al(&code);
      if (now != 0xFFFF) {
        if (now != al_now || code != al_code)
          printf("[master] AL 0x%02x → 0x%02x · code 0x%04x %s\n", al_now, now, code, code ? ec_ALstatuscode2string(code) : "");
        al_now = now;
        al_code = code;
      }
      int op = (al_now & 0x0F) == EC_STATE_OPERATIONAL && !(al_now & EC_STATE_ERROR);
      if (op != was_op) {
        printf("[master] %s\n", op ? "OP" : "OP 아님");
        was_op = op;
        double want_hz = (op && hz_op > 0) ? hz_op : hz;   /* OP 안에서만 빠른 주기 — OP 를 잃으면 전이용 주기로 */
        if (want_hz != hz_now) {
          hz_now = want_hz;
          period = (long)(1e9 / hz_now);
          state_every = (int)(hz_now / state_hz + 0.5);
          printf("[master] 주기 %.0f Hz\n", hz_now);
        }
      }
      if (!no_op && !op && stop_left < 0 && al_now != 0xFFFF) {
        if (al_now & EC_STATE_ERROR) {
          request_al((al_now & 0x0F) | EC_STATE_ACK);
          op_req_t = 0;
        } else if (op_req_t == 0 || t0 - op_req_t > (uint64_t)op_timeout_ms * 1000000ull) {
          if (op_req_t) {
            printf("[master] ⚠ OP 요청 %d ms 지남(AL 0x%02x · code 0x%04x) — 다시 요청\n", op_timeout_ms, al_now, al_code);
            dump_esc();
          }
          request_al(EC_STATE_OPERATIONAL);
          op_req_t = t0;
        }
      }
    }
    if (cycles % state_every == 0) {
      state_msg s = {.magic = STATE_MAGIC, .seq = seq++, .t_ns = t1, .al_state = al_now,
                     .al_code = al_code, .wkc_bad = wkc_bad, .cycles = cycles,
                     .rtt_max_us = rtt_max, .late_max_us = late_max};
      s.flags = (was_op ? FLAG_OP : 0) | (out[OUT_ENABLE] ? FLAG_ENABLED : 0) | (c.commanded ? FLAG_COMMANDED : 0) |
                (node_ok ? FLAG_NODE_OK : 0) | (stop_left >= 0 ? FLAG_STOPPING : 0);
      memcpy(s.in, in, sizeof(s.in));
      memcpy(s.out, out, sizeof(s.out));
      sendto(fd, &s, sizeof(s), MSG_DONTWAIT, (struct sockaddr *)&node, sizeof(node));
      wkc_bad = rtt_max = late_max = 0;
    }
    if (stop_left >= 0 && --stop_left <= 0) break;
  }
  rc = 0;
out_init:
  stop_mode_worker();
  ec_slave[0].state = EC_STATE_INIT;
  ec_writestate(0);
  ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);
  printf("[master] INIT · 끝\n");
out_ec:
  ec_close();
out_sock:
  close(fd);
  unlink(msock);
  return rc;
}
