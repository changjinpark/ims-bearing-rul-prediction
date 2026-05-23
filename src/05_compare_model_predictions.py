"""
05_compare_model_predictions.py

이 파일의 역할:
    02_train_baseline.py, 03_train_gru_torch.py, 04_train_lstm_torch.py의 결과를 모아
    한 장의 비교 그래프를 만듭니다.

왜 필요한가?
    baseline 그래프는 Dummy / RandomForest / MLP를 한 장에서 비교합니다.
    그런데 GRU와 LSTM 그래프가 따로 있으면,
    "전체 모델 중 어떤 선이 actual에 가장 가까운지" 한눈에 보기 어렵습니다.

    그래서 이 파일은 아래 선들을 한 그래프에 같이 그립니다.

    - actual
    - DummyRegressor
    - RandomForest
    - MLP
    - PyTorch GRU
    - PyTorch LSTM

전체 코드 흐름:
    1. 01_make_features.py가 만든 feature CSV를 읽습니다.
    2. 02_train_baseline.py의 helper 함수를 불러와 baseline 모델 예측을 다시 만듭니다.
    3. GRU / LSTM이 저장해둔 prediction CSV를 읽습니다.
    4. 같은 step 기준으로 예측값을 합칩니다.
    5. 전체 모델 비교 CSV와 PNG 그래프를 저장합니다.

실행 예:
    .venv/bin/python src/05_compare_model_predictions.py
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib

# 0-1. 그래프 창을 띄우지 않고 PNG 파일로 저장하기 위한 설정입니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root() -> Path:
    """1. 현재 프로젝트 폴더인 rul_project 경로를 구합니다."""
    return Path(__file__).resolve().parents[1]


def load_baseline_module(root: Path) -> ModuleType:
    """
    2. 02_train_baseline.py 안의 함수를 불러옵니다.

    쉬운 설명:
        baseline 모델 정의는 이미 02_train_baseline.py에 있습니다.
        같은 모델을 여기서 다시 손으로 복사하면 나중에 두 파일이 달라질 수 있습니다.

        그래서 이 파일은 02_train_baseline.py의 build_models(), feature_columns(),
        make_split() 함수를 그대로 가져와 사용합니다.

    왜 import 문을 평범하게 쓰지 않나?
        파일 이름이 02_train_baseline.py처럼 숫자로 시작합니다.
        Python에서는 `import 02_train_baseline` 같은 문법을 쓸 수 없습니다.
        그래서 importlib로 파일 경로를 직접 지정해서 불러옵니다.
    """
    module_path = root / "src" / "02_train_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline_helpers", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import baseline helpers from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_baseline_predictions(args: argparse.Namespace, baseline: ModuleType) -> pd.DataFrame:
    """
    3. Dummy / RandomForest / MLP의 예측값을 다시 만듭니다.

    중요한 점:
        여기서 새 모델을 발명하는 것이 아닙니다.
        02_train_baseline.py와 같은 feature, 같은 split, 같은 random_state를 사용합니다.

    결과:
        각 test step마다 actual과 baseline 예측값이 들어 있는 표를 반환합니다.
    """
    df = pd.read_csv(args.features_csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    baseline_args = SimpleNamespace(
        split=args.split,
        test_size=args.test_size,
        test_experiment=args.test_experiment,
        random_state=args.random_state,
        target=args.target,
    )

    train_df, test_df = baseline.make_split(baseline_args, df)
    cols = baseline.feature_columns(df, args.target)

    X_train = train_df[cols]
    y_train = train_df[args.target]
    X_test = test_df[cols]

    ordered_index = test_df.sort_values(["experiment", "step"]).index
    ordered = test_df.loc[ordered_index].copy()

    output = ordered[
        [
            "experiment",
            "file_name",
            "timestamp",
            "step",
            "n_files",
            "rul_step",
            "rul_capped_125",
            "health_stage_simple",
        ]
    ].copy()
    output["actual"] = ordered[args.target].to_numpy()

    for model_name, model in baseline.build_models(args.random_state).items():
        print(f"recreating baseline prediction: {model_name}", flush=True)
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict(X_test), index=test_df.index).loc[ordered_index]
        output[f"prediction_{model_name}"] = pred.to_numpy()

    return output


def read_sequence_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    4. GRU와 LSTM이 이미 저장해둔 prediction CSV를 읽습니다.

    GRU/LSTM은 window_size=30을 쓰기 때문에 step 0부터 예측하지 않습니다.
    예를 들어 step 0~29를 보고 step 29를 예측하므로, 첫 예측 step은 29입니다.
    """
    gru_df = pd.read_csv(args.gru_predictions_csv)
    lstm_df = pd.read_csv(args.lstm_predictions_csv)
    return gru_df, lstm_df


def combine_predictions(args: argparse.Namespace) -> pd.DataFrame:
    """
    5. baseline 예측값과 GRU/LSTM 예측값을 같은 step 기준으로 합칩니다.

    주의:
        baseline은 test 전체 984개 step에 대해 예측할 수 있습니다.
        GRU/LSTM은 최근 30개 step을 묶어야 하므로 955개 sequence만 예측합니다.

    그래서 공정한 한 장 비교 그래프에서는 GRU/LSTM 예측이 존재하는 step만 사용합니다.
    """
    root = project_root()
    baseline = load_baseline_module(root)
    combined = make_baseline_predictions(args, baseline)
    gru_df, lstm_df = read_sequence_predictions(args)

    combined = combined.merge(
        gru_df[["experiment", "step", "prediction_gru_torch"]],
        on=["experiment", "step"],
        how="inner",
    )
    combined = combined.merge(
        lstm_df[["experiment", "step", "prediction_lstm_torch"]],
        on=["experiment", "step"],
        how="inner",
    )

    return combined.sort_values(["experiment", "step"]).reset_index(drop=True)


def plot_combined_predictions(df: pd.DataFrame, target: str, output_path: Path) -> None:
    """
    6. actual과 모든 모델 예측선을 한 장에 그립니다.

    그래프 읽는 법:
        - 검은색 actual 선에 가까울수록 좋은 예측입니다.
        - Dummy/MLP는 비교 기준이라 얇고 연하게 표시합니다.
        - RandomForest/GRU/LSTM은 최종 비교가 중요해서 조금 더 진하게 표시합니다.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    x = df["step"].to_numpy()

    ax.plot(x, df["actual"].to_numpy(), label="actual", color="black", linewidth=2.4)

    line_specs = [
        ("prediction_dummy_mean", "Dummy", "#8c8c8c", "--", 1.0, 0.65),
        ("prediction_random_forest", "RandomForest", "#2ca02c", "-", 1.5, 0.9),
        ("prediction_mlp_small", "MLP", "#ff7f0e", "--", 1.0, 0.7),
        ("prediction_gru_torch", "PyTorch GRU", "#1f77b4", "-", 1.8, 0.95),
        ("prediction_lstm_torch", "PyTorch LSTM", "#d62728", "-", 1.8, 0.95),
    ]

    for column, label, color, linestyle, linewidth, alpha in line_specs:
        ax.plot(
            x,
            df[column].to_numpy(),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.set_title(f"RUL prediction comparison on {df['experiment'].iloc[0]} ({target})")
    ax.set_xlabel("Test step")
    ax.set_ylabel(target)
    prediction_columns = [column for column, *_ in line_specs]
    y_min = min(float(df["actual"].min()), float(df[prediction_columns].min().min()))
    y_max = max(float(df["actual"].max()), float(df[prediction_columns].max().max()))
    ax.set_ylim(np.floor((y_min - 5) / 10) * 10, np.ceil((y_max + 5) / 10) * 10)
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """
    7. 전체 실행 흐름입니다.

    이 함수가 비교용 CSV와 PNG를 저장합니다.
    """
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_predictions(args)

    suffix = f"{args.split}_{args.target}"
    csv_path = args.tables_dir / f"model_predictions_combined_{suffix}.csv"
    figure_path = args.figures_dir / f"model_predictions_combined_{suffix}.png"

    combined.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}", flush=True)

    plot_combined_predictions(combined, args.target, figure_path)
    print(f"saved: {figure_path}", flush=True)


def parse_args() -> argparse.Namespace:
    """8. 터미널에서 바꿀 수 있는 옵션을 정의합니다."""
    root = project_root()
    output_root = root / "artifacts" / "05_compare_model_predictions" / "outputs_all"

    parser = argparse.ArgumentParser(description="Plot baseline, GRU, and LSTM predictions together.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=root / "artifacts" / "01_make_features" / "outputs_all" / "tables" / "features_combined.csv",
    )
    parser.add_argument(
        "--gru-predictions-csv",
        type=Path,
        default=root
        / "artifacts"
        / "03_train_gru_torch"
        / "outputs_all"
        / "tables"
        / "gru_torch_predictions_by_experiment_rul_capped_125.csv",
    )
    parser.add_argument(
        "--lstm-predictions-csv",
        type=Path,
        default=root
        / "artifacts"
        / "04_train_lstm_torch"
        / "outputs_all"
        / "tables"
        / "lstm_torch_predictions_by_experiment_rul_capped_125.csv",
    )
    parser.add_argument("--tables-dir", type=Path, default=output_root / "tables")
    parser.add_argument("--figures-dir", type=Path, default=output_root / "figures")
    parser.add_argument("--target", choices=["rul_step", "rul_capped_125"], default="rul_capped_125")
    parser.add_argument("--split", choices=["by_experiment"], default="by_experiment")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--test-experiment", default="2nd_bearing1")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """9. 실제 실행 시작점입니다."""
    run(parse_args())


if __name__ == "__main__":
    main()
