/* EtherCAT 프레임 왕복 시간 측정 — 상태 전이 없이 읽기만 한다(RH56F1 손 EtherCAT 점검, 2026-10-02).
 *
 *   ecat_rtt <ifname> [n=2000]
 *
 * BRD(브로드캐스트 읽기)로 ESC 의 AL status(0x0130)를 n 번 읽어 왕복 시간 분포와 응답 slave 수(wkc)를 낸다.
 * ec_config_init 을 부르지 않으므로 슬레이브 상태(INIT/PREOP …)를 바꾸지 않고, 프로세스 데이터(손 지령)도 보내지 않는다.
 * raw socket 이라 cap_net_raw 가 필요하다:  sudo setcap cap_net_raw,cap_net_admin=ep ./ecat_rtt   (운영자)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ethercat.h"

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <ifname> [n]\n", argv[0]); return 2; }
  int n = argc > 2 ? atoi(argv[2]) : 2000;
  if (n < 10) n = 10;
  if (!ec_init(argv[1])) { fprintf(stderr, "ec_init(%s) 실패 — cap_net_raw 가 있는가\n", argv[1]); return 1; }
  double *us = calloc(n, sizeof(double));
  int ok = 0, wkc_min = 1 << 30, wkc_max = 0, lost = 0;
  uint16 al = 0, al_first = 0;
  for (int i = 0; i < n; i++) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int wkc = ec_BRD(0x0000, 0x0130, sizeof(al), &al, EC_TIMEOUTRET);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (wkc <= 0) { lost++; continue; }
    if (ok == 0) al_first = al;
    us[ok++] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    if (wkc < wkc_min) wkc_min = wkc;
    if (wkc > wkc_max) wkc_max = wkc;
  }
  ec_close();
  if (!ok) { printf("%s: 응답 없음 (%d 번 모두 timeout)\n", argv[1], n); free(us); return 1; }
  qsort(us, ok, sizeof(double), cmp);
  double sum = 0;
  for (int i = 0; i < ok; i++) sum += us[i];
  printf("%s: n %d · 응답 %d · 잃음 %d · slave(wkc) %d~%d · AL status 0x%04x\n", argv[1], n, ok, lost, wkc_min, wkc_max,
         al_first);
  printf("  왕복 us: min %.1f · p50 %.1f · p90 %.1f · p99 %.1f · max %.1f · 평균 %.1f\n", us[0], us[ok / 2],
         us[(int)(ok * 0.9)], us[(int)(ok * 0.99)], us[ok - 1], sum / ok);
  free(us);
  return 0;
}
