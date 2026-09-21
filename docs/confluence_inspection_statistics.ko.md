# Inspection Document → Inspection Statistics 변환 (Data Drift 분석용)

## 목적

- Data Agent가 AI 검사 모델의 **data drift**(모델 입력 데이터나 모델 동작이 시간에 따라 달라지는 현상)를 안정적으로 분석할 수 있도록 document 변환 pipeline 구축
- 원본 inspection document 대신, **기간 단위로 미리 집계된 통계 collection**(`inspectionStatistics`) 생성
- MCP 서버(Data Agent가 호출하는 tool) 쪽의 계산을 제거하고, 통계는 Data Service가 저장 시점에 계산

## 현재 상태

- Data Service는 검사 결과 document를 MongoDB `inspections` collection에 저장 중
- 현재 data drift tool은 이 `inspections` collection에서 원본 document를 직접 읽어서, prediction 하나하나를 펼친 뒤 tool 내부에서 모든 통계를 계산한다
- 하나의 inspection document에는 여러 검사 모델의 결과(`inspectionResults.aiResults`)가 포함될 수 있고, prediction 수에 따라 document가 매우 커질 수 있다

## 문제점

현재 방식으로 drift를 분석하면 아래 문제가 생긴다.

- **조회 비용**: 한 번의 분석에 수천~수만 개의 원본 document를 읽고 펼쳐야 한다. 분석 기간이 길어질수록 비례해서 느려진다.
- **기간 제한과 sampling**: 비용 때문에 분석 기간을 최대 30일로 제한하고, record 수가 15,000개를 넘으면 sampling을 해야 했다. 장기 추세를 보기 어렵고, sampling은 결과의 정확도를 떨어뜨린다.
- **복잡한 로직**: bucket 자동 선택, 작은 bucket 병합, 여러 통계 검정이 tool 안에 들어가 있어 유지보수가 어렵고, LLM이 결과를 읽기에도 필드가 너무 많다.
- **원본 document 의존**: inspection document는 약 30일 후 만료되므로, 그 이전 데이터와는 비교 자체가 불가능하다.

## Solution

### 1. 변환 방식

- Data Service가 inspection document를 **기간(period) 단위로 집계**해서 별도 collection `inspectionStatistics`에 저장
- 기간 종류(`granularity`)는 4가지: `hourly`(1시간), `shift`(12시간, 00시~12시 / 12시~24시), `daily`(1일), `weekly`(월요일~월요일). 경계는 모두 UTC 기준
- 변환 단위: **(모델 이름 × 모델 버전 × task × mode × gbm × process × productId × 기간 종류 × 기간 시작 시각)** 당 statistics document 1개
- 하나의 document 안에는 equipment별 통계와, 모든 equipment를 합친 `total` 통계가 함께 들어간다
- 해당 기간에 inspection document가 하나도 없으면 statistics document도 만들지 않는다
- 대상 task는 `cls`(classification)와 `det`(detection). `seg`는 drift 분석 대상이 아니므로 제외
- 보관 기간: 약 1년 (원본은 30일이므로, 장기 비교는 이 collection으로만 가능해진다)

### 2. 무엇을 저장하는가

Drift 분석에 필요한 것은 "prediction 하나하나"가 아니라 **"이 기간에 모델이 어떻게 답했는지의 분포"**다. 그래서 개별 prediction은 저장하지 않고, 아래 숫자만 저장한다. 모두 **더할 수 있는 값**(count, sum, histogram)이라서, 시간별 document를 더하면 일별·주별 값이 정확히 나온다.

| 항목 | 설명 |
|---|---|
| 데이터 수 | inspection document 수, prediction 수(이미지 수), detection의 경우 bounding box 수 |
| Class별 개수 | 예: OK 2,190개, NG 170개 → class 비율 |
| Confidence 분포 | confidence 값을 고정된 10개 구간으로 나누어, 각 구간에 몇 개가 들어갔는지 세어 둔 **histogram**. 전체와 class별로 각각 저장 |
| Confidence 요약 | 합계, 제곱합(→ 평균과 표준편차 계산용), 최소/최대, 분위수(p05~p95) |
| Threshold 관련 개수 | confidence가 threshold 미만인 prediction 수, threshold 근처(±0.05)인 prediction 수 |
| 이미지당 box 수 (det) | box가 0개, 1개, 2개, ... 인 이미지가 각각 몇 개인지의 histogram, box가 없는 이미지 수 |
| 추론 시간 | elapsed time의 개수와 합계 |
| 실행 환경 | backend, threshold 값 (설정 변경 감지용) |
| 품질 | confidence가 없거나 파싱에 실패한 prediction 수 |

**Confidence histogram이 핵심인 이유.** 두 기간의 histogram을 나란히 놓으면 "모델이 예전보다 덜 확신한다"는 것이 구간별 개수 차이로 바로 보인다. 그리고 histogram은 더할 수 있으므로, 어떤 기간 조합이든 정확한 분포를 다시 만들 수 있다. 평균 하나만 저장하면 이런 비교가 불가능하다.

### 3. Statistics Document Schema (classification 예시, 축약)

```json
{
  "modelName": "sidetopsideu8000",
  "modelVersion": "sidetopsideu8000_260316_260316081905",
  "task": "cls",
  "mode": "production",
  "gbm": "SEHC",
  "process": "Side",
  "productId": "12HJ3NGL601106K",
  "granularity": "hourly",
  "startDate": "2026-06-20T00:00:00Z",
  "endDate":   "2026-06-20T01:00:00Z",
  "classes": ["OK", "NG"],
  "localTimezone": "Asia/Bangkok",
  "bins": {
    "confidenceEdges": [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0],
    "nearThresholdMargin": 0.05,
    "boxesPerImageMax": 5
  },
  "computedAt": "2026-06-20T01:05:12Z",
  "equipmentCount": 2,
  "total": { "...equipment 항목과 같은 구조, 모든 equipment의 합..." },
  "equipments": [
    {
      "equipmentId": "SEHC_Side_VM07",
      "location": "VM07",
      "backend": "ts",
      "threshold": 0.5,
      "inspectionCount": 118,
      "predictionCount": 2360,
      "quality": { "missingConfidenceCount": 0, "parseErrorCount": 0 },
      "elapsedTime": { "count": 2360, "sum": 472.0, "mean": 0.2, "min": 0.1, "max": 0.31, "p50": 0.2, "p90": 0.24 },
      "classCounts": { "OK": 2190, "NG": 170 },
      "confidence": {
        "count": 2360, "sum": 2276.4, "sumSq": 2199.1, "mean": 0.9646, "std": 0.041,
        "min": 0.5012, "max": 0.9999,
        "quantiles": { "p05": 0.88, "p10": 0.91, "p25": 0.95, "p50": 0.975, "p75": 0.99, "p90": 0.995, "p95": 0.998 },
        "histogram": [0, 0, 0, 0, 3, 12, 41, 210, 620, 1474],
        "belowThresholdCount": 3,
        "nearThresholdCount": 15
      },
      "perClass": {
        "OK": { "...confidence와 같은 구조, OK로 예측된 prediction만..." },
        "NG": { "...NG로 예측된 prediction만..." }
      }
    },
    { "equipmentId": "SEHC_Side_VM08", "location": "VM08", "..." : "..." }
  ]
}
```

- `bins`는 histogram의 구간 정의다. 분석 tool은 이 값을 document에서 읽으므로, 나중에 구간을 바꾸더라도 tool 수정 없이 동작한다. 단, 구간이 다른 document끼리는 비교하지 않는다.
- Detection 모델은 `confidence`, `classCounts`, `perClass`가 bounding box 기준이고, `boxCount`와 `images`(이미지당 box 수 histogram, box 없는 이미지 수) 항목이 추가된다. threshold는 class별로 `perClass` 안에 들어간다.
- Index: `(modelName, modelVersion, task, mode, gbm, process, granularity, startDate, productId)` unique index 1개와, 보관 기간용 `startDate` TTL index

### 4. Drift 분석은 어떻게 하는가

새 tool은 statistics document만 읽고, **더하기와 나누기 수준의 계산**만 한다. 결과는 아래 순서로 읽는다.

**(1) 언제 바뀌었나 — 기간별 시계열과 연속 비교**

요청한 기간을 period 단위로 나열하고(예: 최근 30일을 하루씩), 각 period의 평균 confidence, class 비율, threshold 미만 비율 등을 보여준다. 그리고 **각 period를 바로 앞 period와 비교한 PSI**를 함께 준다. 어느 날 갑자기 PSI가 튀면 그날이 변화 시점이다.

**(2) 얼마나 바뀌었나 — PSI (Population Stability Index)**

PSI는 "두 분포가 얼마나 다른가"를 숫자 하나로 나타내는 지표다. 두 기간의 confidence histogram을 구간별로 비교해서, 구간마다 비율이 얼마나 옮겨갔는지를 합한 값이다. 신용평가·모델 모니터링 분야에서 오래 쓰여 온 기준이 있다.

| PSI | 해석 |
|---|---|
| 0.10 미만 | 의미 있는 변화 없음 |
| 0.10 ~ 0.25 | 중간 정도 변화, 확인 필요 |
| 0.25 이상 | 큰 변화 |

예를 들어 6월 1주에는 confidence의 95%가 0.95 이상 구간에 있었는데, 6월 3주에는 그 비율이 60%로 줄고 0.8~0.9 구간이 늘었다면 PSI는 0.25를 넘고, "모델이 예전보다 확신을 못 하고 있다"는 뜻이 된다. 이미지가 달라졌을 때(조명, 카메라 오염, 부품 변경) 가장 먼저 나타나는 신호다.

같은 방식으로 **class 비율**(OK/NG 비율이 바뀌었는가)과 detection의 **이미지당 box 수 분포**(box를 예전보다 못 찾거나 더 찾는가)에도 PSI를 계산한다. class 비율에는 통계 검정(chi-square)을 보조로 붙여, 데이터가 아주 많을 때 미세한 변화가 과장되지 않도록 한다.

**(3) 어디서 갈라졌나 — change point**

기간 전체에서 "앞쪽과 뒤쪽의 PSI 합이 가장 커지는 지점"을 찾아, 그 지점 기준으로 before/after 요약(각각의 histogram, class 비율, 평균 confidence, threshold 미만 비율)을 함께 준다. 기준 기간을 직접 지정하면(comparison mode) 그 기준 기간과 현재 기간을 그대로 비교한다. 어느 class가 움직였는지도 class별 PSI로 알 수 있다.

**(4) 설정이 바뀐 것은 아닌가 — hard break**

연속된 period 사이에 backend, threshold, class 목록이 달라지면 "설정 변경"으로 따로 보고한다. 설정 변경은 drift가 아니라 운영 이벤트이므로, 통계적 결론보다 먼저 현장 확인이 필요하다.

**(5) 판단 — flags와 preVerdict**

위 숫자들에 고정된 기준을 적용해 flag(`CONFIDENCE_SHIFT`, `CLASS_SHIFT`, `BOX_COUNT_SHIFT`, `THRESHOLD_PRESSURE`, `HARD_BREAK`, 데이터 부족 flag 등)를 세우고, 이를 종합한 `preVerdict`(stable / suspicious / drift_likely / undetermined)를 준다. 최종 판단은 LLM 또는 담당자가 숫자를 보고 확인하거나 뒤집는다. 데이터가 기준(classification 500개, detection 300개)에 못 미치는 쪽이 있으면 flag를 세우지 않고 undetermined로 둔다.

### 5. 기존 방식과의 비교

| 항목 | 기존 (inspections 직접 조회) | 변경 후 (inspectionStatistics) |
|---|---|---|
| 읽는 데이터 | 원본 document 수천~수만 개 | period당 작은 row 몇 개 |
| 분석 가능 기간 | 최대 30일, 원본 만료 전까지 | 최대 약 400일, 보관 기간 1년 |
| Sampling | record 15,000개 초과 시 sampling | 없음, 항상 전체 |
| 계산 위치 | MCP tool 내부 | 분포·개수는 Data Service, tool은 비교 계산만 |
| 결과 필드 | bucket, trend, outlier, KS, box geometry 등 | period 시계열, 연속 PSI, change point, hard break, flag |
| Product 단위 조회 | 불가 | productId로 가능 |

제거된 항목(KS test, 추세 검정, outlier 점수, box 크기·위치 통계, 이미지 크기 변경 감지)은 원본 값이 있어야 하거나, period 시계열과 연속 PSI로 충분히 대체되는 것들이다.

## 활용 방안

- **모델 상태 monitoring**: 모델·라인·equipment별로 기간에 따른 confidence 분포, class 비율, threshold 미만 비율, 추론 시간 조회
- **Drift 감지**: 위 4절의 tool 결과를 Data Agent가 읽고 "언제, 얼마나, 어느 class가, 설정 변경 때문인지" 설명
- **자동 알림**: `preVerdict`와 flag만으로도 알림 정책 구성 가능 (drift_likely → 알림, suspicious → 기록, hard break → 담당자 통보)
- **장기 비교**: 원본이 만료된 뒤에도 몇 달 전 기준 기간과 현재를 비교 가능

## Considerations

- **생성 방식**: 기존 summary collection과 같이 on-ingest(수신 시 갱신)와 backfill용 API endpoint로 구현. 저장 항목이 모두 더할 수 있는 값이므로, 늦게 도착한 inspection document를 해당 period document에 더하는 방식으로 갱신 가능. 단, 분위수(quantiles)와 min/max는 더할 수 없으므로 period가 닫힌 뒤 재계산하거나 근사값으로 둔다.
- **Histogram 구간(bins)**: 정상 모델의 confidence는 대부분 0.9 이상에 몰리므로, 균등한 0.1 간격 대신 **0.9 이상을 잘게 나눈 구간**(예: 0.9, 0.95, 0.98, 0.99, 1.0)을 권장한다. 실제 모델의 confidence 샘플로 확인한 뒤 확정하고, 확정 후에는 바꾸지 않는 것이 좋다(구간이 다르면 과거와 비교 불가).
- **productId를 key에 포함하는 것의 비용**: productId가 제품 barcode(개별 단위)라면 period당 barcode 수만큼 document가 생긴다. 시간당 300개를 검사하는 라인이면 하루 약 7,000개, 1년에 모델당 250만 개 수준이며, tool은 조회 시 이를 DB 안에서 합산해야 한다. 이 경우 **productId가 null인 "전체 product" document를 period당 1개 추가로 저장**하고, product를 지정하지 않은 분석은 그 document만 읽게 하는 방안을 검토해야 한다.
- **Human feedback**: 통계에 포함하지 않는다.
- **Image spec, box geometry**: 저장하지 않는다. 카메라·해상도 변경은 confidence 변화로만 간접적으로 드러난다.
- **보관**: `startDate` 기준 1년 TTL index. 필요 시 granularity별로 다른 보관 기간을 위해 별도 `expiresAt` 필드로 전환 가능.
