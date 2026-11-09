# RH56F1 정지 추종 분석

## 관측 사실

수신 시각과 원본 timestamp는 events.jsonl에 보존한다.

## 계산 결과

```json
{
  "pass": false,
  "eligible": false,
  "commands_sent": 0,
  "trials": 1,
  "excluded": []
}
```

## 원인 가설

중력·마찰의 개별 기여도는 이 기록만으로 확정하지 않는다.

## 합격 여부

미통과 또는 판정 제외

FK 추종 차이는 모델상의 계산이며 실제 공간의 1 mm 정확도를 증명하지 않는다. applied.effort는 feedforward 토크다.
