/* RH56F1 손 EtherCAT 점검 도구 (SOEM v1.4.0) — 2026-10-02.
 *
 *   ecat_rh56f1 rtt    <ifname> [n=2000]             BRD 왕복 시간(상태 전이 없음)
 *   ecat_rh56f1 safeop <ifname> [seconds=3] [hz=1000] PREOP → SAFE_OP 후 입력 PDO 를 주기적으로 읽는다
 *
 * ★SAFE_OP 까지만 간다. OP 는 요청하지 않는다 — 출력 PDO(ANGLESET 등)는 SAFE_OP 에서 손이 쓰지 않고, 이 도구는 출력
 *   영역을 0 으로 둔 채 건드리지 않는다. 끝나면 INIT 으로 되돌린다. 손은 움직이지 않는다.
 *
 * AL 0x1E(Invalid input configuration)의 원인과 우회:
 *   손 펌웨어의 PDO 매핑 객체(0x1601 RxPDO 19 항목 · 0x1A00 TxPDO 76 항목)의 항목이 표준 UINT32
 *   (index<<16 | sub<<8 | bitlen) 가 아니라 UINT16(sub<<8 | bitlen, 예 0x0110)이다. SOEM 은 CoE complete access(SDOCA)로
 *   매핑을 읽을 때 4 바이트 항목으로 해석해 크기를 잘못 계산하고(출력 144 · 입력 608 bit), 그 길이로 SM2/SM3 를 쓴다.
 *   손이 기대하는 길이는 매뉴얼 표 50 과 같은 출력 19 × INT16 = 38 B · 입력 76 × INT16 = 152 B 다 → 입력 길이 불일치 = 0x1E.
 *   SDOCA 를 끄면 SOEM 이 항목을 하나씩 읽고(2 바이트 값의 하위 바이트 = bitlen 16) 크기가 38 / 152 B 로 맞는다.
 *
 * raw socket 이라 cap_net_raw 가 필요하다:  sudo setcap cap_net_raw,cap_net_admin=ep ./ecat_rh56f1   (운영자)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ethercat.h"

#define OUT_WORDS 19
#define IN_WORDS 76

static char IOmap[4096];

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static void stats(const char *label, double *v, int n) {
  if (n <= 0) { printf("  %s: 없음\n", label); return; }
  qsort(v, n, sizeof(double), cmp);
  double s = 0;
  for (int i = 0; i < n; i++) s += v[i];
  printf("  %s us: min %.1f · p50 %.1f · p90 %.1f · p99 %.1f · max %.1f · 평균 %.1f (n %d)\n", label, v[0], v[n / 2],
         v[(int)(n * 0.9)], v[(int)(n * 0.99)], v[n - 1], s / n, n);
}

static int cmd_rtt(const char *ifname, int n) {
  if (!ec_init(ifname)) { fprintf(stderr, "ec_init(%s) 실패 — cap_net_raw 가 있는가\n", ifname); return 1; }
  double *us = calloc(n, sizeof(double));
  int ok = 0, lost = 0, wkc_max = 0;
  uint16 al = 0;
  for (int i = 0; i < n; i++) {
    double t0 = now_us();
    int wkc = ec_BRD(0x0000, ECT_REG_ALSTAT, sizeof(al), &al, EC_TIMEOUTRET);
    double t1 = now_us();
    if (wkc <= 0) { lost++; continue; }
    us[ok++] = t1 - t0;
    if (wkc > wkc_max) wkc_max = wkc;
  }
  ec_close();
  printf("%s: BRD n %d · 응답 %d · 잃음 %d · slave %d · AL status 0x%04x\n", ifname, n, ok, lost, wkc_max, al);
  stats("왕복", us, ok);
  free(us);
  return ok ? 0 : 1;
}

static const char *FINGER[6] = {"새끼", "약지", "중지", "검지", "엄지굽힘", "엄지회전"};

static void print_inputs(const int16 *w) {
  /* 매뉴얼 표 50: 0x6000:01~06 POSACT · 07~0C ANGLEACT · 0D~12 FORCEACT · 13~18 CURACT · 19~1E ERROR · 1F~24 STATUS · 25~2A TEMP */
  const char *grp[7] = {"POSACT", "ANGLEACT", "FORCEACT", "CURACT", "ERROR", "STATUS", "TEMP"};
  for (int g = 0; g < 7; g++) {
    printf("    %-8s", grp[g]);
    for (int f = 0; f < 6; f++) printf(" %s %6d", FINGER[f], w[g * 6 + f]);
    printf("\n");
  }
  printf("    촉각 · 기타(0x2B~0x4C):");
  for (int i = 42; i < IN_WORDS; i++) printf(" %d", w[i]);
  printf("\n");
}

static int cmd_safeop(const char *ifname, double seconds, double hz) {
  if (!ec_init(ifname)) { fprintf(stderr, "ec_init(%s) 실패 — cap_net_raw 가 있는가\n", ifname); return 1; }
  int rc = 1;
  if (ec_config_init(FALSE) <= 0) { printf("%s: slave 없음\n", ifname); goto out; }
  for (int s = 1; s <= ec_slavecount; s++) ec_slave[s].CoEdetails &= ~ECT_COEDET_SDOCA;   /* ★위 주석 */
  int used = ec_config_map(&IOmap);
  ec_configdc();   /* 매뉴얼: DC 모드는 지원하지 않는다 — SYNC 를 켜지 않고 시간만 읽는다 */
  printf("%s: slave %d · %s · 출력 %d B · 입력 %d B (기대 %d / %d) · IOmap %d B\n", ifname, ec_slavecount,
         ec_slave[1].name, ec_slave[1].Obytes, ec_slave[1].Ibytes, OUT_WORDS * 2, IN_WORDS * 2, used);
  ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);
  ec_readstate();
  if (ec_slave[1].state != EC_STATE_SAFE_OP) {
    printf("  ✗ SAFE_OP 실패: state 0x%02x · AL 0x%04x %s\n", ec_slave[1].state, ec_slave[1].ALstatuscode,
           ec_ALstatuscode2string(ec_slave[1].ALstatuscode));
    goto back;
  }
  printf("  ✓ SAFE_OP (OP 는 요청하지 않는다 — 출력 미적용)\n");
  if (ec_slave[1].Ibytes < IN_WORDS * 2) { printf("  ✗ 입력 %d B < %d B\n", ec_slave[1].Ibytes, IN_WORDS * 2); goto back; }

  int n = (int)(seconds * hz);
  if (n < 1) n = 1;
  double *rtt = calloc(n, sizeof(double)), *per = calloc(n, sizeof(double));
  int ok = 0, bad = 0, nper = 0, expected = ec_group[0].outputsWKC * 2 + ec_group[0].inputsWKC;
  int16 first[IN_WORDS], last[IN_WORDS];
  long period_ns = (long)(1e9 / hz);
  struct timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  double prev = 0;
  for (int i = 0; i < n; i++) {
    next.tv_nsec += period_ns;
    while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    double t0 = now_us();
    ec_send_processdata();
    int wkc = ec_receive_processdata(EC_TIMEOUTRET);
    double t1 = now_us();
    if (prev > 0) per[nper++] = t0 - prev;
    prev = t0;
    if (wkc <= 0) { bad++; continue; }
    rtt[ok++] = t1 - t0;
    memcpy(last, ec_slave[1].inputs, sizeof(last));
    if (ok == 1) memcpy(first, last, sizeof(first));
  }
  printf("  주기 %.0f Hz · %d 회 · 응답 %d · 잃음 %d · wkc 기대 %d\n", hz, n, ok, bad, expected);
  stats("PDO 왕복", rtt, ok);
  stats("주기 간격", per, nper);
  if (ok) {
    printf("  첫 입력:\n");
    print_inputs(first);
    printf("  마지막 입력:\n");
    print_inputs(last);
    rc = 0;
  }
  free(rtt);
  free(per);
back:
  ec_slave[0].state = EC_STATE_INIT;
  ec_writestate(0);
  ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);
  ec_readstate();
  printf("  끝: state 0x%02x\n", ec_slave[1].state);
out:
  ec_close();
  return rc;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s rtt <ifname> [n] | safeop <ifname> [seconds] [hz]\n", argv[0]);
    return 2;
  }
  if (!strcmp(argv[1], "rtt")) return cmd_rtt(argv[2], argc > 3 ? atoi(argv[3]) : 2000);
  if (!strcmp(argv[1], "safeop"))
    return cmd_safeop(argv[2], argc > 3 ? atof(argv[3]) : 3.0, argc > 4 ? atof(argv[4]) : 1000.0);
  fprintf(stderr, "unknown command %s\n", argv[1]);
  return 2;
}
