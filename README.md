# IMS Bearing RUL Prediction

IMS Bearing Data를 사용해 베어링의 RUL(Remaining Useful Life)을 예측한 딥러닝 수업 기말 프로젝트입니다.

원본 진동 파일에서 통계/주파수 feature를 추출하고, 마지막 측정 파일을 failure time으로 가정해 RUL 라벨을 만든 뒤, feature 기반 비교 모델과 PyTorch GRU/LSTM sequence 모델을 비교했습니다.

## 결과 요약

최종 평가는 `by_experiment` split으로 진행했습니다.

```text
train: 1st_bearing3, 1st_bearing4, 3rd_bearing3
test : 2nd_bearing1
target: rul_capped_125
```

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| DummyRegressor | 9.66 | 25.28 | -0.055 |
| RandomForest | 6.24 | 14.36 | 0.659 |
| MLP | 17.03 | 21.08 | 0.266 |
| PyTorch GRU | 4.53 | 13.36 | 0.713 |
| PyTorch LSTM | 4.68 | 14.38 | 0.668 |

PyTorch GRU가 가장 좋은 결과를 보였고, LSTM도 RandomForest보다 낮은 MAE를 보였습니다. 한 시점의 feature만 보는 모델보다, 최근 30개 시점의 feature 흐름을 보는 sequence 모델이 RUL 예측에 더 유리했습니다.

## 문서 안내

처음 보는 사람은 아래 순서로 읽으면 됩니다.

```text
docs/index.md
  블로그 제출용 글입니다. 문제 정의, 접근 과정, 결과, 한계를 설명합니다.

CONCEPTS_FOR_BEGINNERS.md
  RMS, kurtosis, FFT, RUL 같은 용어를 초보자 관점에서 설명합니다.

OUTPUTS_GUIDE.md
  각 산출물이 어떤 코드에서 만들어졌고, 무슨 의미인지 설명합니다.
```

## 프로젝트 구조

```text
rul_project/
  README.md
  requirements.txt
  .gitignore

  src/
    01_make_features.py
    02_train_baseline.py
    03_train_gru_torch.py
    04_train_lstm_torch.py
    05_compare_model_predictions.py

  docs/
    index.md
    assets/

  artifacts/      # 실행 후 생성됨. GitHub에는 올리지 않음.
  .venv/          # 로컬 가상환경. GitHub에는 올리지 않음.
```

주요 코드:

```text
src/01_make_features.py
  원본 진동 파일을 feature CSV와 EDA 그래프로 변환합니다.

src/02_train_baseline.py
  DummyRegressor, RandomForestRegressor, MLPRegressor를 학습합니다.

src/03_train_gru_torch.py
  PyTorch nn.GRU 기반 sequence 모델입니다.

src/04_train_lstm_torch.py
  PyTorch nn.LSTM 기반 sequence 모델입니다.

src/05_compare_model_predictions.py
  actual, baseline, GRU, LSTM 예측선을 한 장의 비교 그래프로 합칩니다.
```

## 데이터 준비

Kaggle IMS Bearing Dataset을 다운로드한 뒤, 프로젝트의 부모 폴더에 `archive/`가 있도록 둡니다.

예상 구조:

```text
IMS Bearing Data/
  archive/
    1st_test/
    2nd_test/
    3rd_test/

  rul_project/
    src/
    docs/
```

원본 데이터는 용량과 라이선스 문제 때문에 GitHub에 올리지 않습니다.

## 환경 설정

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt`에는 PyTorch, pandas, scikit-learn, matplotlib이 포함되어 있습니다. GRU와 LSTM 모델은 PyTorch를 사용합니다.

실험을 실행한 환경:

```text
OS: macOS 14.6.1, arm64
Python: 3.11.0
Device: CPU

numpy: 2.1.3
pandas: 2.3.3
scikit-learn: 1.8.0
matplotlib: 3.10.9
torch: 2.12.0
```

## 실행 방법

### 1. Beginner feature 생성

```bash
python3 src/01_make_features.py
```

생성물:

```text
artifacts/01_make_features/outputs/tables/features_combined.csv
artifacts/01_make_features/outputs/figures/2nd_bearing1_feature_trends.png
```

### 2. 전체 실패 베어링 feature 생성

```bash
python3 src/01_make_features.py --experiments all --output-dir artifacts/01_make_features/outputs_all
```

여기서 `all`은 모든 베어링이 아니라, 최종 고장이 보고된 베어링들을 의미합니다.

```text
1st_bearing3
1st_bearing4
2nd_bearing1
3rd_bearing3
```

### 3. 비교 기준 모델 학습

기본 random split:

```bash
python3 src/02_train_baseline.py
```

시간 순서 split:

```bash
python3 src/02_train_baseline.py --split last
```

실험 단위 split:

```bash
python3 src/02_train_baseline.py \
  --features-csv artifacts/01_make_features/outputs_all/tables/features_combined.csv \
  --metrics-csv artifacts/02_train_baseline/outputs_all/tables/baseline_metrics.csv \
  --figures-dir artifacts/02_train_baseline/outputs_all/figures \
  --split by_experiment \
  --test-experiment 2nd_bearing1
```

### 4. PyTorch GRU 학습

```bash
.venv/bin/python src/03_train_gru_torch.py
```

생성물:

```text
artifacts/03_train_gru_torch/outputs_all/tables/gru_torch_metrics_by_experiment_rul_capped_125.csv
artifacts/03_train_gru_torch/outputs_all/figures/gru_torch_predictions_by_experiment_rul_capped_125.png
artifacts/03_train_gru_torch/outputs_all/figures/gru_torch_training_loss_by_experiment_rul_capped_125.png
artifacts/03_train_gru_torch/outputs_all/models/gru_torch_model_by_experiment_rul_capped_125.pt
```

### 5. PyTorch LSTM 학습

```bash
.venv/bin/python src/04_train_lstm_torch.py
```

생성물:

```text
artifacts/04_train_lstm_torch/outputs_all/tables/lstm_torch_metrics_by_experiment_rul_capped_125.csv
artifacts/04_train_lstm_torch/outputs_all/figures/lstm_torch_predictions_by_experiment_rul_capped_125.png
artifacts/04_train_lstm_torch/outputs_all/figures/lstm_torch_training_loss_by_experiment_rul_capped_125.png
artifacts/04_train_lstm_torch/outputs_all/models/lstm_torch_model_by_experiment_rul_capped_125.pt
```

### 6. 전체 모델 예측 그래프 생성

```bash
.venv/bin/python src/05_compare_model_predictions.py
```

생성물:

```text
artifacts/05_compare_model_predictions/outputs_all/tables/model_predictions_combined_by_experiment_rul_capped_125.csv
artifacts/05_compare_model_predictions/outputs_all/figures/model_predictions_combined_by_experiment_rul_capped_125.png
```

## 블로그

GitHub Pages용 블로그 본문은 [docs/index.md](docs/index.md)에 있습니다.

GitHub Pages 설정:

```text
Settings
-> Pages
-> Deploy from a branch
-> Branch: main
-> Folder: /docs
```

블로그에 필요한 이미지는 `docs/assets/`에 복사해두었습니다.

CSV, 모델 파일, 전체 산출물은 `artifacts/` 아래에서 재생성할 수 있으므로 GitHub에는 올리지 않습니다. CSV가 위험한 파일이라는 뜻은 아니고, 원본 데이터에서 만들어지는 중간/결과 산출물이기 때문에 코드와 요약 이미지 중심으로 올립니다.

최종 블로그 글은 `docs/index.md`이고, 최종 sequence 모델 코드는 `src/03_train_gru_torch.py`와 `src/04_train_lstm_torch.py`입니다. 전체 모델 비교 그래프는 `src/05_compare_model_predictions.py`로 생성합니다.
