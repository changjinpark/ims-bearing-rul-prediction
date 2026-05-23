# 산출물 해석 가이드

이 문서는 `rul_project`에서 생성되는 파일을 **실제 폴더 구조 기준**으로 설명합니다.

새 산출물은 이제 아래처럼 코드 단계별 폴더로 나눠서 저장합니다.

```text
artifacts/
  01_make_features/
    outputs/      beginner feature 생성 결과
    outputs_all/  확장 feature 생성 결과

  02_train_baseline/
    outputs/      beginner 모델 학습 결과
    outputs_all/  확장 모델 학습 결과

  03_train_gru_torch/
    outputs_all/  PyTorch GRU 결과

  04_train_lstm_torch/
    outputs_all/  PyTorch LSTM 결과

  05_compare_model_predictions/
    outputs_all/  baseline + GRU + LSTM 전체 비교 그래프
```

즉, 먼저 이렇게 나눠서 봅니다.

```text
1. artifacts/01_make_features
   원본 진동 파일을 feature CSV와 EDA 그래프로 바꾼 결과

2. artifacts/02_train_baseline
   feature CSV를 읽어 모델을 학습하고 평가한 결과

3. artifacts/03_train_gru_torch
   feature CSV의 여러 행을 sequence로 묶어 PyTorch GRU를 학습하고 평가한 결과

4. artifacts/04_train_lstm_torch
   feature CSV의 여러 행을 sequence로 묶어 PyTorch LSTM을 학습하고 평가한 결과

5. artifacts/05_compare_model_predictions
   actual, baseline, GRU, LSTM 예측선을 한 장에서 비교한 결과
```

그 다음 각 단계 안에서 `outputs`와 `outputs_all`을 나눠 봅니다.
현재 프로젝트의 산출물은 `artifacts/...` 아래 파일을 기준으로 보면 됩니다.

CSV 파일은 위험한 파일이라서 제외한 것이 아닙니다. 다만 `artifacts/` 아래 CSV와 모델 파일은 코드 실행으로 다시 만들 수 있는 생성 산출물이고, 일부는 Kaggle 원본 데이터에서 파생된 큰 중간 파일입니다. 그래서 GitHub에는 코드, 설명 문서, 블로그에 필요한 요약 이미지만 올리는 방식으로 정리했습니다.

RMS, kurtosis, FFT 같은 용어가 낯설면 먼저 [CONCEPTS_FOR_BEGINNERS.md](CONCEPTS_FOR_BEGINNERS.md)를 읽는 것이 좋습니다.

## 0. 전체 흐름

전체 파이프라인은 아래 순서입니다.

```text
원본 진동 파일들
-> src/01_make_features.py
-> feature CSV + EDA 그래프
-> src/02_train_baseline.py
-> 모델 성능표 + 예측 그래프 + feature importance
-> src/03_train_gru_torch.py
-> PyTorch GRU 성능표 + PyTorch GRU 예측 그래프 + 모델 파일
-> src/04_train_lstm_torch.py
-> PyTorch LSTM 성능표 + PyTorch LSTM 예측 그래프 + 모델 파일
-> src/05_compare_model_predictions.py
-> 전체 모델 예측 비교 그래프
```

쉽게 말하면:

```text
01_make_features.py:
  원본 데이터를 모델이 먹을 수 있는 표로 바꾸는 코드

02_train_baseline.py:
  그 표를 읽어서 RUL 예측 모델을 학습하고 평가하는 코드

03_train_gru_torch.py:
  그 표의 여러 시점 행을 묶어서 PyTorch nn.GRU 모델을 학습하는 코드

04_train_lstm_torch.py:
  같은 표의 여러 시점 행을 묶어서 PyTorch nn.LSTM 모델을 학습하는 코드

05_compare_model_predictions.py:
  actual, baseline 3개, GRU, LSTM 예측선을 한 장의 그래프로 합치는 코드
```

## 1. `src/01_make_features.py`가 만든 파일

이 파일의 역할은 원본 진동 파일을 feature 표와 EDA 그래프로 바꾸는 것입니다.

```text
원본 진동 파일
-> 파일별 feature 계산
-> RUL 라벨 생성
-> CSV 저장
-> feature 변화 그래프 저장
```

### 1-1. 이 코드가 선택하는 데이터 기준

코드 위치:
[src/01_make_features.py](src/01_make_features.py#L114) 114-175행

핵심 코드:

```python
return {
    "2nd_bearing1": ExperimentSpec(... usecols=(0,), failed_bearing="bearing1"),
    "3rd_bearing3": ExperimentSpec(... usecols=(2,), failed_bearing="bearing3"),
    "1st_bearing3": ExperimentSpec(... usecols=(4, 5), failed_bearing="bearing3"),
    "1st_bearing4": ExperimentSpec(... usecols=(6, 7), failed_bearing="bearing4"),
}
```

의도:

```text
모든 베어링을 쓰는 것이 아니라,
IMS Bearing Dataset 설명에서 최종 고장이 보고된 베어링만 고릅니다.
```

선택 기준:

```text
Set 1: bearing 3, bearing 4 고장
Set 2: bearing 1 고장
Set 3: bearing 3 고장
```

중요한 해석:

```text
고장난 베어링만 골랐다
!= 모든 행이 고장 상태라는 뜻
```

각 고장 베어링의 **전체 시간 흐름**을 넣은 것입니다.

```text
초반 파일: 정상에 가까운 상태
중간 파일: 열화 진행 상태
마지막 파일: 고장 시점으로 가정
```

정상 베어링을 넣지 않은 이유:

```text
실험 끝까지 고장나지 않은 베어링은 정확한 고장 시점을 모릅니다.
따라서 단순 RUL 정답 라벨을 만들기 어렵습니다.
```

### 1-2. RUL 정답 라벨은 어디서 만들어지나

코드 위치:
[src/01_make_features.py](src/01_make_features.py#L351) 351-382행

핵심 코드:

```python
rul_step = total - 1 - step

rows.append(
    {
        "step": step,
        "n_files": total,
        "rul_step": rul_step,
        "rul_minutes_approx": rul_step * 10,
        "rul_capped_125": min(rul_step, 125),
        **features,
    }
)
```

의도:

```text
PDF나 원본 데이터에 파일별 RUL 정답지는 없습니다.
우리가 코드에서 직접 만든 가정 기반 정답입니다.
```

기준:

```text
마지막 측정 파일 = failure time = RUL 0
현재 파일의 RUL = 마지막 파일까지 남은 파일 개수
```

예시:

```text
파일이 총 984개라면

첫 번째 파일      step = 0   -> rul_step = 983
두 번째 파일      step = 1   -> rul_step = 982
마지막 전 파일    step = 982 -> rul_step = 1
마지막 파일       step = 983 -> rul_step = 0
```

`rul_capped_125`는 너무 큰 RUL을 125로 제한한 값입니다.

```text
rul_step = 983 -> rul_capped_125 = 125
rul_step = 500 -> rul_capped_125 = 125
rul_step = 30  -> rul_capped_125 = 30
rul_step = 0   -> rul_capped_125 = 0
```

### 1-3. `artifacts/01_make_features/outputs`에 만들어지는 파일

실행 명령:

```bash
python3 src/01_make_features.py
```

이 명령은 beginner 모드입니다.

사용 데이터:

```text
2nd_test의 bearing1만 사용
```

생성 파일:

```text
artifacts/01_make_features/outputs/tables/features_2nd_bearing1.csv
artifacts/01_make_features/outputs/tables/features_combined.csv
artifacts/01_make_features/outputs/figures/2nd_bearing1_feature_trends.png
```

#### `artifacts/01_make_features/outputs/tables/features_2nd_bearing1.csv`

코드 위치:
[src/01_make_features.py](src/01_make_features.py#L461) 461-464행

핵심 코드:

```python
frame = make_features_for_experiment(experiments[name], limit=args.limit)
experiment_output = tables_dir / f"features_{name}.csv"
frame.to_csv(experiment_output, index=False)
```

의도:

```text
2nd_test bearing1의 원본 파일들을 feature 표로 바꿔 저장합니다.
```

용도:

```text
처음 데이터 구조를 이해하는 beginner 실험용 CSV입니다.
```

한 행의 의미:

```text
특정 시점의 bearing1 상태 1개
```

#### `artifacts/01_make_features/outputs/tables/features_combined.csv`

코드 위치:
[src/01_make_features.py](src/01_make_features.py#L468) 468-470행

핵심 코드:

```python
combined = pd.concat(frames, ignore_index=True)
combined_output = tables_dir / "features_combined.csv"
combined.to_csv(combined_output, index=False)
```

의도:

```text
모델 학습 코드는 feature CSV 하나를 입력으로 받습니다.
그래서 실험별 feature를 하나의 대표 CSV로 합칩니다.
```

beginner 모드에서는 실험이 하나뿐이라 `features_2nd_bearing1.csv`와 거의 같은 내용입니다.

용도:

```text
src/02_train_baseline.py의 기본 입력 파일입니다.
```

#### `artifacts/01_make_features/outputs/figures/2nd_bearing1_feature_trends.png`

코드 위치:
[src/01_make_features.py](src/01_make_features.py#L391) 391-417행

핵심 코드:

```python
trend_columns = ["rms", "kurtosis", "crest_factor", "fft_total_power"]
output_path = figures_dir / f"{experiment}_feature_trends.png"
fig.savefig(output_path, dpi=160)
```

의도:

```text
고장에 가까워질수록 feature가 실제로 변하는지 눈으로 확인합니다.
```

용도:

```text
EDA 그래프
블로그에서 "고장 직전 feature 변화"를 보여주는 그림
```

### 1-4. `artifacts/01_make_features/outputs_all`에 만들어지는 파일

실행 명령:

```bash
python3 src/01_make_features.py --experiments all --output-dir artifacts/01_make_features/outputs_all
```

사용 데이터:

```text
1st_bearing3
1st_bearing4
2nd_bearing1
3rd_bearing3
```

생성 파일:

```text
artifacts/01_make_features/outputs_all/tables/features_1st_bearing3.csv
artifacts/01_make_features/outputs_all/tables/features_1st_bearing4.csv
artifacts/01_make_features/outputs_all/tables/features_2nd_bearing1.csv
artifacts/01_make_features/outputs_all/tables/features_3rd_bearing3.csv
artifacts/01_make_features/outputs_all/tables/features_combined.csv

artifacts/01_make_features/outputs_all/figures/1st_bearing3_feature_trends.png
artifacts/01_make_features/outputs_all/figures/1st_bearing4_feature_trends.png
artifacts/01_make_features/outputs_all/figures/2nd_bearing1_feature_trends.png
artifacts/01_make_features/outputs_all/figures/3rd_bearing3_feature_trends.png
```

의도:

```text
고장 베어링 여러 개를 합쳐서 더 과제다운 실험을 하기 위한 데이터입니다.
```

가장 중요한 파일:

```text
artifacts/01_make_features/outputs_all/tables/features_combined.csv
```

이 파일은 이후 `by_experiment` 평가와 GRU/LSTM 확장의 입력 후보입니다.

주의:

```text
artifacts/01_make_features/outputs_all/tables/features_combined.csv는 모든 베어링 데이터가 아닙니다.
최종적으로 고장난 베어링들의 feature를 모은 RUL 예측용 데이터입니다.
```

## 2. `src/02_train_baseline.py`가 만든 파일

이 파일의 역할은 `01_make_features.py`가 만든 `features_combined.csv`를 읽어서 모델을 학습하고 평가하는 것입니다.

```text
feature CSV
-> train/test split
-> X/y 분리
-> Dummy, RandomForest, MLP 학습
-> 성능표 저장
-> 예측 그래프 저장
-> feature importance 저장
```

### 2-1. 이 코드가 feature CSV를 읽고 정답 y를 고르는 위치

코드 위치:
[src/02_train_baseline.py](src/02_train_baseline.py#L291) 291-311행

핵심 코드:

```python
df = pd.read_csv(args.features_csv)
train_df, test_df = make_split(args, df)
cols = feature_columns(df, args.target)

X_train = train_df[cols]
y_train = train_df[args.target]
X_test = test_df[cols]
y_test = test_df[args.target]
```

의도:

```text
features_combined.csv를 읽습니다.
X는 입력 feature로 나눕니다.
y는 맞혀야 하는 정답 RUL로 나눕니다.
```

기본 정답:

코드 위치:
[src/02_train_baseline.py](src/02_train_baseline.py#L398) 398행

```python
parser.add_argument("--target", choices=["rul_step", "rul_capped_125"], default="rul_capped_125")
```

즉 기본값은:

```text
y = rul_capped_125
```

입니다.

### 2-2. 모델을 학습하고 성능을 계산하는 위치

코드 위치:
[src/02_train_baseline.py](src/02_train_baseline.py#L317) 317-342행

핵심 코드:

```python
for model_name, model in build_models(args.random_state).items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    row = {
        "model": model_name,
        "split": args.split,
        "target": args.target,
        **evaluate(y_test.to_numpy(), pred),
    }
    metrics.append(row)
```

의도:

```text
Dummy, RandomForest, MLP 모델을 학습합니다.
테스트 데이터에서 RUL을 예측합니다.
MAE, RMSE, R2를 계산합니다.
```

### 2-3. `artifacts/02_train_baseline/outputs`에 만들어지는 파일

#### random split 결과

실행 명령:

```bash
python3 src/02_train_baseline.py
```

입력 파일:

```text
artifacts/01_make_features/outputs/tables/features_combined.csv
```

생성 파일:

```text
artifacts/02_train_baseline/outputs/tables/baseline_metrics_random_rul_capped_125.csv
artifacts/02_train_baseline/outputs/figures/baseline_predictions_random_rul_capped_125.png
```

성능표 저장 코드:

[src/02_train_baseline.py](src/02_train_baseline.py#L344) 344-350행

```python
metrics_df = pd.DataFrame(metrics)
metrics_path = metrics_path.with_name(f"baseline_metrics_{args.split}_{args.target}.csv")
metrics_df.to_csv(metrics_path, index=False)
```

예측 그래프 저장 코드:

[src/02_train_baseline.py](src/02_train_baseline.py#L359) 359-361행

```python
plot_path = args.figures_dir / f"baseline_predictions_{args.split}_{args.target}.png"
plot_predictions(test_df, predictions, args.target, plot_path)
```

의도:

```text
모델이 일단 feature와 RUL 관계를 학습할 수 있는지 빠르게 확인합니다.
```

주의:

```text
random split은 같은 실험의 가까운 시점이 train/test에 섞일 수 있습니다.
그래서 최종 성능으로 보기보다 sanity check로 봅니다.
```

#### time-based split 결과

실행 명령:

```bash
python3 src/02_train_baseline.py --split last
```

생성 파일:

```text
artifacts/02_train_baseline/outputs/tables/baseline_metrics_last_rul_capped_125.csv
artifacts/02_train_baseline/outputs/figures/baseline_predictions_last_rul_capped_125.png
```

의도:

```text
앞부분으로 학습하고 뒷부분으로 테스트해 시간 순서 평가를 확인합니다.
```

해석:

```text
단일 실험에서 last split 성능이 나쁠 수 있습니다.
앞 80%에는 rul_capped_125가 대부분 125라서,
모델이 고장 직전 하락 패턴을 충분히 배우지 못할 수 있습니다.
```

### 2-4. `artifacts/02_train_baseline/outputs_all`에 만들어지는 파일

실행 명령:

```bash
python3 src/02_train_baseline.py \
  --features-csv artifacts/01_make_features/outputs_all/tables/features_combined.csv \
  --metrics-csv artifacts/02_train_baseline/outputs_all/tables/baseline_metrics.csv \
  --figures-dir artifacts/02_train_baseline/outputs_all/figures \
  --split by_experiment \
  --test-experiment 2nd_bearing1
```

입력 파일:

```text
artifacts/01_make_features/outputs_all/tables/features_combined.csv
```

생성 파일:

```text
artifacts/02_train_baseline/outputs_all/tables/baseline_metrics_by_experiment_rul_capped_125.csv
artifacts/02_train_baseline/outputs_all/tables/feature_importance_by_experiment_rul_capped_125.csv
artifacts/02_train_baseline/outputs_all/figures/baseline_predictions_by_experiment_rul_capped_125.png
```

#### `baseline_metrics_by_experiment_rul_capped_125.csv`

코드 위치:
[src/02_train_baseline.py](src/02_train_baseline.py#L344) 344-350행

의도:

```text
실험 단위 split의 모델 성능을 저장합니다.
```

split 의미:

```text
train: 1st_bearing3, 1st_bearing4, 3rd_bearing3
test : 2nd_bearing1
```

현재 결과:

```text
DummyRegressor MAE = 9.66,  RMSE = 25.28, R2 = -0.06
RandomForest   MAE = 6.24,  RMSE = 14.36, R2 = 0.659
MLP            MAE = 17.03, RMSE = 21.08, R2 = 0.266
```

용도:

```text
최종 baseline 성능표
블로그의 핵심 결과
나중에 GRU/LSTM과 비교할 기준
```

#### `baseline_predictions_by_experiment_rul_capped_125.png`

코드 위치:
[src/02_train_baseline.py](src/02_train_baseline.py#L245) 245-277행,
[src/02_train_baseline.py](src/02_train_baseline.py#L359) 359-361행

핵심 코드:

```python
ax.plot(ordered[target].to_numpy(), label="actual", linewidth=2.0, color="black")
ax.plot(pred_series.to_numpy(), label=model_name, linewidth=1.2, alpha=0.85)
```

의도:

```text
실제 RUL과 예측 RUL을 선 그래프로 비교합니다.
```

읽는 법:

```text
검은 선: 실제 RUL
색깔 선: 모델 예측 RUL
색깔 선이 검은 선에 가까울수록 좋음
```

#### `feature_importance_by_experiment_rul_capped_125.csv`

코드 위치:
[src/02_train_baseline.py](src/02_train_baseline.py#L324) 324-357행

핵심 코드:

```python
feature_importance_df = (
    pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
    .sort_values("importance", ascending=False)
)
feature_importance_df.to_csv(importance_path, index=False)
```

의도:

```text
RandomForest가 어떤 feature를 중요하게 사용했는지 저장합니다.
```

현재 상위 feature:

```text
band_energy_6k_10k
abs_mean
kurtosis
rms
skewness
```

용도:

```text
블로그 해석 파트
"고주파 에너지와 진동 크기 feature가 중요했다"는 근거
```

## 3. `src/03_train_gru_torch.py`가 만든 파일

이 파일의 역할은 PyTorch의 `nn.GRU`를 사용해서 RUL 예측 모델을 학습하는 것입니다.

```text
feature CSV
-> train/test split
-> feature 표준화
-> 여러 행을 window_size 길이 sequence로 변환
-> train sequence 중 일부를 validation으로 분리
-> PyTorch nn.GRU 학습
-> test sequence 예측
-> 성능표, 예측 그래프, 모델 파일 저장
```

### 3-1. 가상환경과 라이브러리

가상환경 위치:

```text
.venv/
```

설치 파일:

```text
requirements.txt
```

주요 라이브러리:

```text
torch
pandas
scikit-learn
matplotlib
```

이번 최종 GRU/LSTM 실행에는 `torch`를 사용했습니다.

### 3-2. 기본 실행 명령

```bash
.venv/bin/python src/03_train_gru_torch.py
```

기본 split:

```text
train: 1st_bearing3, 1st_bearing4, 3rd_bearing3
test : 2nd_bearing1
```

기본 sequence 설정:

```text
window_size = 30
```

즉 최근 30개 시점의 feature 흐름으로 현재 시점의 RUL을 예측합니다.

### 3-3. `artifacts/03_train_gru_torch/outputs_all`에 만들어지는 파일

생성 파일:

```text
artifacts/03_train_gru_torch/outputs_all/tables/gru_torch_metrics_by_experiment_rul_capped_125.csv
artifacts/03_train_gru_torch/outputs_all/tables/gru_torch_predictions_by_experiment_rul_capped_125.csv
artifacts/03_train_gru_torch/outputs_all/tables/gru_torch_training_history_by_experiment_rul_capped_125.csv
artifacts/03_train_gru_torch/outputs_all/figures/gru_torch_predictions_by_experiment_rul_capped_125.png
artifacts/03_train_gru_torch/outputs_all/figures/gru_torch_training_loss_by_experiment_rul_capped_125.png
artifacts/03_train_gru_torch/outputs_all/models/gru_torch_model_by_experiment_rul_capped_125.pt
artifacts/03_train_gru_torch/outputs_all/models/gru_torch_config_by_experiment_rul_capped_125.json
```

#### `gru_torch_metrics_by_experiment_rul_capped_125.csv`

의도:

```text
PyTorch GRU 모델의 최종 test 성능을 저장합니다.
```

현재 결과:

```text
PyTorch GRU MAE  = 4.53
PyTorch GRU RMSE = 13.36
PyTorch GRU R2   = 0.713
```

해석:

```text
PyTorch GRU가 RandomForest와 MLP보다 좋은 결과를 보였습니다.
최근 30개 시점의 feature 흐름을 보는 것이 RUL 예측에 도움이 된 것으로 해석할 수 있습니다.
```

#### `gru_torch_predictions_by_experiment_rul_capped_125.csv`

의도:

```text
test sequence마다 실제 RUL과 PyTorch GRU 예측 RUL을 저장합니다.
```

#### `gru_torch_predictions_by_experiment_rul_capped_125.png`

의도:

```text
검은 선(actual)과 PyTorch GRU 예측 선을 시간 순서대로 비교합니다.
```

#### `gru_torch_training_loss_by_experiment_rul_capped_125.png`

의도:

```text
학습 중 train loss와 validation loss 변화를 보여줍니다.
```

주의:

```text
test 데이터는 최종 평가에만 사용합니다.
학습 중에는 train sequence 일부를 validation으로 떼어 과적합 여부를 확인했습니다.
```

#### `gru_torch_model_by_experiment_rul_capped_125.pt`

의도:

```text
PyTorch 모델 가중치와 feature 표준화 정보를 저장합니다.
```

## 4. `src/04_train_lstm_torch.py`가 만든 파일

이 파일의 역할은 PyTorch의 `nn.LSTM`을 사용해서 RUL 예측 모델을 학습하는 것입니다.

흐름은 GRU와 거의 같습니다.

```text
feature CSV
-> train/test split
-> feature 표준화
-> 여러 행을 window_size 길이 sequence로 변환
-> train sequence 중 일부를 validation으로 분리
-> PyTorch nn.LSTM 학습
-> test sequence 예측
-> 성능표, 예측 그래프, 모델 파일 저장
```

GRU와 LSTM의 차이는 모델 구조입니다.

```text
GRU:
  시계열 흐름을 보는 비교적 단순한 recurrent 모델

LSTM:
  cell state를 사용해 오래 전 정보를 기억하도록 설계된 recurrent 모델
```

이번 프로젝트에서는 두 모델을 공정하게 비교하기 위해 같은 feature CSV, 같은 train/test split, 같은 `window_size = 30`을 사용했습니다.

### 4-1. 기본 실행 명령

```bash
.venv/bin/python src/04_train_lstm_torch.py
```

기본 split:

```text
train: 1st_bearing3, 1st_bearing4, 3rd_bearing3
test : 2nd_bearing1
```

### 4-2. `artifacts/04_train_lstm_torch/outputs_all`에 만들어지는 파일

생성 파일:

```text
artifacts/04_train_lstm_torch/outputs_all/tables/lstm_torch_metrics_by_experiment_rul_capped_125.csv
artifacts/04_train_lstm_torch/outputs_all/tables/lstm_torch_predictions_by_experiment_rul_capped_125.csv
artifacts/04_train_lstm_torch/outputs_all/tables/lstm_torch_training_history_by_experiment_rul_capped_125.csv
artifacts/04_train_lstm_torch/outputs_all/figures/lstm_torch_predictions_by_experiment_rul_capped_125.png
artifacts/04_train_lstm_torch/outputs_all/figures/lstm_torch_training_loss_by_experiment_rul_capped_125.png
artifacts/04_train_lstm_torch/outputs_all/models/lstm_torch_model_by_experiment_rul_capped_125.pt
artifacts/04_train_lstm_torch/outputs_all/models/lstm_torch_config_by_experiment_rul_capped_125.json
```

#### `lstm_torch_metrics_by_experiment_rul_capped_125.csv`

의도:

```text
PyTorch LSTM 모델의 최종 test 성능을 저장합니다.
```

현재 결과:

```text
PyTorch LSTM MAE  = 4.68
PyTorch LSTM RMSE = 14.38
PyTorch LSTM R2   = 0.668
```

해석:

```text
이번 실행에서는 PyTorch GRU가 LSTM보다 조금 더 좋은 결과를 보였습니다.
그래도 LSTM은 RandomForest보다 MAE가 낮아 sequence 모델의 장점을 확인할 수 있었습니다.
```

#### `lstm_torch_predictions_by_experiment_rul_capped_125.csv`

의도:

```text
test sequence마다 실제 RUL과 PyTorch LSTM 예측 RUL을 저장합니다.
```

#### `lstm_torch_predictions_by_experiment_rul_capped_125.png`

의도:

```text
검은 선(actual)과 PyTorch LSTM 예측 선을 시간 순서대로 비교합니다.
```

#### `lstm_torch_training_loss_by_experiment_rul_capped_125.png`

의도:

```text
학습 중 train loss와 validation loss 변화를 보여줍니다.
```

#### `lstm_torch_model_by_experiment_rul_capped_125.pt`

의도:

```text
PyTorch LSTM 모델 가중치와 feature 표준화 정보를 저장합니다.
```

## 5. `src/05_compare_model_predictions.py`가 만든 파일

이 파일의 역할은 baseline, GRU, LSTM 예측 결과를 한 장의 그래프로 합치는 것입니다.

```text
feature CSV
-> baseline 모델 예측 재생성
-> GRU prediction CSV 읽기
-> LSTM prediction CSV 읽기
-> 같은 step 기준으로 합치기
-> 전체 모델 비교 CSV와 PNG 저장
```

### 5-1. 기본 실행 명령

```bash
.venv/bin/python src/05_compare_model_predictions.py
```

### 5-2. `artifacts/05_compare_model_predictions/outputs_all`에 만들어지는 파일

생성 파일:

```text
artifacts/05_compare_model_predictions/outputs_all/tables/model_predictions_combined_by_experiment_rul_capped_125.csv
artifacts/05_compare_model_predictions/outputs_all/figures/model_predictions_combined_by_experiment_rul_capped_125.png
```

#### `model_predictions_combined_by_experiment_rul_capped_125.csv`

의도:

```text
같은 test step에 대해 actual, Dummy, RandomForest, MLP, GRU, LSTM 예측값을 한 표에 모읍니다.
```

주의:

```text
baseline은 test 전체 step을 예측할 수 있습니다.
GRU/LSTM은 window_size=30을 사용하므로 step 29부터 예측합니다.
그래서 전체 비교 CSV와 그래프는 GRU/LSTM 예측이 존재하는 step만 사용합니다.
```

#### `model_predictions_combined_by_experiment_rul_capped_125.png`

의도:

```text
actual, baseline 3개, GRU, LSTM을 한 장에서 비교합니다.
블로그의 최종 결과 섹션에서 가장 직관적으로 보기 좋은 그래프입니다.
```

읽는 법:

```text
검은 선: actual RUL
초록 선: RandomForest
파란 선: PyTorch GRU
빨간 선: PyTorch LSTM
회색/주황 점선: Dummy, MLP 비교 기준
```

검은 선에 가까울수록 좋은 예측입니다.

## 6. 새 폴더 구조를 어떻게 읽으면 되는가

이제 폴더 이름은 이렇게 이해하면 됩니다.

```text
artifacts/01_make_features:
  01_make_features.py가 만든 feature CSV와 EDA 그래프

artifacts/02_train_baseline:
  02_train_baseline.py가 만든 모델 성능표와 예측 그래프

artifacts/03_train_gru_torch:
  03_train_gru_torch.py가 만든 PyTorch GRU 성능표, 예측 그래프, 모델 파일

artifacts/04_train_lstm_torch:
  04_train_lstm_torch.py가 만든 PyTorch LSTM 성능표, 예측 그래프, 모델 파일

artifacts/05_compare_model_predictions:
  05_compare_model_predictions.py가 만든 전체 모델 비교 그래프

각 단계 안의 outputs:
  작은 beginner 실험 결과
  2nd_bearing1만 사용

각 단계 안의 outputs_all:
  확장 실험 결과
  최종 고장 베어링 4개를 사용
```

즉 폴더의 첫 번째 기준은 "어떤 코드 단계가 만들었는가"이고,
두 번째 기준은 "beginner 실험인가, 확장 실험인가"입니다.

## 7. 처음 읽을 때 추천 순서

모든 산출물을 한 번에 보지 말고 아래 순서로 보세요.

```text
1. artifacts/01_make_features/outputs/figures/2nd_bearing1_feature_trends.png
   고장에 가까워질수록 feature가 변하는지 확인

2. artifacts/01_make_features/outputs/tables/features_combined.csv
   모델에 들어가는 표가 어떻게 생겼는지 확인

3. artifacts/02_train_baseline/outputs_all/tables/baseline_metrics_by_experiment_rul_capped_125.csv
   최종 baseline 성능 숫자 확인

4. artifacts/02_train_baseline/outputs_all/figures/baseline_predictions_by_experiment_rul_capped_125.png
   실제 RUL과 예측 RUL의 모양 비교

5. artifacts/03_train_gru_torch/outputs_all/tables/gru_torch_metrics_by_experiment_rul_capped_125.csv
   PyTorch GRU sequence 모델의 성능 확인

6. artifacts/03_train_gru_torch/outputs_all/figures/gru_torch_predictions_by_experiment_rul_capped_125.png
   PyTorch GRU 예측 RUL과 실제 RUL 비교

7. artifacts/04_train_lstm_torch/outputs_all/tables/lstm_torch_metrics_by_experiment_rul_capped_125.csv
   PyTorch LSTM sequence 모델의 성능 확인

8. artifacts/04_train_lstm_torch/outputs_all/figures/lstm_torch_predictions_by_experiment_rul_capped_125.png
   PyTorch LSTM 예측 RUL과 실제 RUL 비교

9. artifacts/05_compare_model_predictions/outputs_all/figures/model_predictions_combined_by_experiment_rul_capped_125.png
   actual, baseline, GRU, LSTM을 한 장에서 비교
```

## 8. 블로그에 쓸 수 있는 문장

```text
본 프로젝트의 산출물은 크게 다섯 단계에서 생성된다.
먼저 src/01_make_features.py에서 원본 진동 파일을 읽고,
각 파일을 RMS, kurtosis, crest factor, FFT band energy 등의 feature로 변환했다.
이 과정에서 마지막 측정 파일을 failure time으로 가정하고,
rul_step과 rul_capped_125 라벨을 생성했다.

그 다음 src/02_train_baseline.py에서 feature CSV를 입력으로 사용해
DummyRegressor, RandomForest, MLP 모델을 학습했다.
모델 성능은 MAE, RMSE, R2가 담긴 metrics CSV와
실제 RUL 대비 예측 RUL 그래프로 평가했다.

그 다음 src/03_train_gru_torch.py에서 최근 30개 시점의 feature를 하나의 sequence로 묶어
PyTorch GRU 모델을 학습했다.

마지막으로 src/04_train_lstm_torch.py에서 같은 입력 조건으로 PyTorch LSTM 모델을 학습했다.

최종적으로 src/05_compare_model_predictions.py에서 actual, baseline 모델, GRU, LSTM의 예측선을
한 장의 그래프로 합쳐 전체 모델의 예측 경향을 비교했다.
이를 통해 한 시점 feature만 보는 모델과 여러 시점의 흐름을 보는 sequence 모델을 비교했고,
sequence 모델 안에서도 GRU와 LSTM의 성능을 비교했다.
```
