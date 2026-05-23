"""
02_train_baseline.py

이 파일의 역할:
    01_make_features.py가 만든 feature CSV를 사용해서 RUL 예측 모델을 학습합니다.

처음 보면 중요한 관점:
    - 입력 X: rms, kurtosis, band_energy 같은 feature들
    - 정답 y: rul_capped_125 또는 rul_step
    - 목표: feature를 보고 RUL을 숫자로 예측하는 회귀(regression) 문제

전체 코드 흐름:
    1. feature CSV를 읽습니다.
    2. 어떤 컬럼을 입력 feature로 쓸지 고릅니다.
    3. train/test를 나눕니다.
    4. 여러 baseline 모델을 학습합니다.
    5. MAE, RMSE, R2로 성능을 평가합니다.
    6. 예측 그래프와 결과 CSV를 저장합니다.

실행 예:
    python3 src/02_train_baseline.py
    python3 src/02_train_baseline.py --split last
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# 0-1. 그래프 창을 띄우지 않고 PNG 파일로 저장하기 위한 설정입니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 0-2. 아래 컬럼들은 feature가 아니라 설명/정답/관리용 컬럼입니다.
#      모델 입력 X에서 제외해야 합니다.
#
# 예:
#   - file_name은 파일 이름이라 모델이 배울 물리적 진동 정보가 아닙니다.
#   - rul_capped_125는 정답 y라서 입력 X에 넣으면 정답 유출이 됩니다.
META_COLUMNS = {
    "experiment",
    "failed_bearing",
    "failure_description",
    "file_name",
    "timestamp",
    "step",
    "n_files",
    "rul_step",
    "rul_minutes_approx",
    "rul_capped_125",
    "health_stage_simple",
}


def project_root() -> Path:
    """1. 현재 프로젝트 폴더인 rul_project 경로를 구합니다."""
    return Path(__file__).resolve().parents[1]


def feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """
    2. 모델 입력 X로 사용할 feature 컬럼만 고릅니다.

    쉬운 설명:
        CSV에는 feature, 정답, 파일 이름, 실험 이름이 모두 섞여 있습니다.
        모델에는 진동에서 계산한 숫자 feature만 넣어야 합니다.

    이 함수는 다음을 제외합니다.
        - META_COLUMNS에 들어 있는 설명용 컬럼
        - 현재 예측하려는 정답 target 컬럼
        - 숫자가 아닌 컬럼
    """
    excluded = set(META_COLUMNS)
    excluded.add(target)
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def split_random(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    3-A. random split입니다.

    의미:
        전체 행을 무작위로 train/test에 나눕니다.

    장점:
        모델이 feature와 RUL 관계를 배울 수 있는지 빠르게 확인하기 좋습니다.

    단점:
        같은 실험의 바로 옆 시간 데이터가 train/test에 섞일 수 있습니다.
        그래서 성능이 실제보다 좋게 나올 수 있습니다.

    결론:
        sanity check용입니다. 최종 보고서의 핵심 성능으로는 조심해서 써야 합니다.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train_df.sort_values(["experiment", "step"]), test_df.sort_values(["experiment", "step"])


def split_last(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    3-B. 시간 순서 split입니다.

    의미:
        각 실험의 앞부분은 train, 뒷부분은 test로 둡니다.

    장점:
        미래 데이터를 과거 학습에 섞지 않으므로 random split보다 정직합니다.

    단점:
        capped RUL에서는 앞부분 정답이 대부분 125입니다.
        그래서 모델이 고장 직전의 감소 구간을 충분히 학습하지 못할 수 있습니다.
    """
    train_parts = []
    test_parts = []
    for _, part in df.sort_values(["experiment", "step"]).groupby("experiment", sort=False):
        cutoff = int(len(part) * (1.0 - test_size))
        cutoff = min(max(cutoff, 1), len(part) - 1)
        train_parts.append(part.iloc[:cutoff])
        test_parts.append(part.iloc[cutoff:])
    return pd.concat(train_parts), pd.concat(test_parts)


def split_by_experiment(df: pd.DataFrame, test_experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    3-C. 실험 단위 split입니다.

    의미:
        특정 실험 하나를 통째로 test로 빼고,
        나머지 실험들로 train합니다.

    예:
        train: 1st_bearing3, 1st_bearing4, 3rd_bearing3
        test : 2nd_bearing1

    장점:
        모델이 "처음 보는 다른 실험"에도 통하는지 볼 수 있습니다.
        과제 보고서에서는 random split보다 설득력이 좋습니다.
    """
    if test_experiment not in set(df["experiment"]):
        choices = ", ".join(sorted(df["experiment"].unique()))
        raise ValueError(f"Unknown test experiment: {test_experiment}. Choices: {choices}")
    train_df = df[df["experiment"] != test_experiment]
    test_df = df[df["experiment"] == test_experiment]
    if train_df.empty:
        raise ValueError("by_experiment split needs at least two experiments. Run 01_make_features.py --experiments all first.")
    return train_df, test_df


def make_split(args: argparse.Namespace, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    3-D. 사용자가 고른 split 방식을 실제 함수로 연결합니다.

    터미널에서 --split random, --split last, --split by_experiment 중 하나를 고를 수 있습니다.
    """
    if args.split == "random":
        return split_random(df, args.test_size, args.random_state)
    if args.split == "last":
        return split_last(df, args.test_size)
    if args.split == "by_experiment":
        return split_by_experiment(df, args.test_experiment)
    raise ValueError(f"Unsupported split: {args.split}")


def build_models(random_state: int) -> dict[str, object]:
    """
    4. 비교할 baseline 모델들을 만듭니다.

    baseline이란?
        최종 딥러닝 모델을 만들기 전에 기준점으로 삼는 간단한 모델입니다.
        나중에 LSTM/GRU를 만들더라도 이 baseline보다 좋아야 의미가 있습니다.

    모델 설명:
        - dummy_mean:
            train 정답의 평균값만 계속 예측합니다.
            아무것도 배우지 않는 모델이라 최저 기준선입니다.

        - random_forest:
            여러 decision tree를 묶은 모델입니다.
            표 형태 feature에서 강한 편이라 좋은 baseline입니다.

        - mlp_small:
            작은 신경망입니다.
            딥러닝 수업 관점에서 "feature 기반 신경망" baseline으로 볼 수 있습니다.
    """
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
        "mlp_small": make_pipeline(
            # MLP는 입력 feature 스케일에 민감합니다.
            # 그래서 평균 0, 표준편차 1 정도로 맞춰주는 StandardScaler를 앞에 둡니다.
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                early_stopping=True,
                max_iter=700,
                random_state=random_state,
            ),
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    5. 예측 성능을 숫자로 계산합니다.

    MAE:
        평균 절대 오차입니다.
        예: MAE=6이면 평균적으로 RUL을 약 6 step 정도 틀린다는 뜻입니다.

    RMSE:
        큰 오차에 더 민감한 지표입니다.
        예측이 크게 빗나간 경우를 더 강하게 벌점 줍니다.

    R2:
        1에 가까울수록 좋습니다.
        0보다 작으면 단순 평균 예측보다도 못할 수 있다는 뜻입니다.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_predictions(
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    target: str,
    output_path: Path,
) -> None:
    """
    6. 실제 RUL과 모델 예측 RUL을 한 그래프에 그립니다.

    그래프 해석:
        - 검은색 actual 선: 우리가 만든 정답 RUL
        - 다른 색 선: 각 모델의 예측 RUL
        - 예측 선이 검은색 선에 가까울수록 좋은 모델입니다.
    """
    ordered_index = test_df.sort_values(["experiment", "step"]).index
    ordered = test_df.loc[ordered_index]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(ordered[target].to_numpy(), label="actual", linewidth=2.0, color="black")

    for model_name, pred in predictions.items():
        pred_series = pd.Series(pred, index=test_df.index).loc[ordered_index]
        ax.plot(pred_series.to_numpy(), label=model_name, linewidth=1.2, alpha=0.85)

    ax.set_title(f"RUL prediction on test set ({target})")
    ax.set_xlabel("Test rows sorted by experiment and time")
    ax.set_ylabel(target)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_and_evaluate(args: argparse.Namespace) -> pd.DataFrame:
    """
    7. 학습과 평가의 전체 흐름입니다.

    세부 순서:
        7-1. feature CSV를 읽습니다.
        7-2. train/test로 나눕니다.
        7-3. X와 y를 분리합니다.
        7-4. 모델들을 하나씩 학습합니다.
        7-5. 성능 지표와 예측 그래프를 저장합니다.
    """
    # 7-1. 01_make_features.py가 만든 CSV를 읽습니다.
    df = pd.read_csv(args.features_csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    # 7-2. 평가 방식에 맞춰 train/test를 나눕니다.
    train_df, test_df = make_split(args, df)

    # 7-3. feature 컬럼만 고릅니다.
    cols = feature_columns(df, args.target)

    print(f"features: {len(cols)} columns")
    print(f"train rows: {len(train_df)}, test rows: {len(test_df)}")
    print(f"target: {args.target}")
    print(f"split: {args.split}")

    # 7-4. X는 입력 feature, y는 맞혀야 하는 정답 RUL입니다.
    X_train = train_df[cols]
    y_train = train_df[args.target]
    X_test = test_df[cols]
    y_test = test_df[args.target]

    metrics = []
    predictions = {}
    feature_importance_df = None

    # 7-5. 모델을 하나씩 학습하고 평가합니다.
    for model_name, model in build_models(args.random_state).items():
        print(f"training: {model_name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[model_name] = pred

        # RandomForest는 어떤 feature를 중요하게 봤는지 알려줄 수 있습니다.
        # 이 값은 블로그에서 "모델이 어떤 진동 특징에 주목했나"를 설명할 때 유용합니다.
        if hasattr(model, "feature_importances_"):
            feature_importance_df = (
                pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        row = {
            "model": model_name,
            "split": args.split,
            "target": args.target,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            **evaluate(y_test.to_numpy(), pred),
        }
        metrics.append(row)
        print(f"{model_name}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, R2={row['r2']:.3f}")

    # 7-6. 모델별 성능표를 CSV로 저장합니다.
    metrics_df = pd.DataFrame(metrics)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_csv
    if metrics_path.name == "baseline_metrics.csv":
        metrics_path = metrics_path.with_name(f"baseline_metrics_{args.split}_{args.target}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"saved: {metrics_path}")

    # 7-7. RandomForest feature importance를 저장합니다.
    if feature_importance_df is not None:
        importance_path = metrics_path.with_name(metrics_path.name.replace("baseline_metrics", "feature_importance"))
        feature_importance_df.to_csv(importance_path, index=False)
        print(f"saved: {importance_path}")

    # 7-8. 실제값과 예측값 비교 그래프를 저장합니다.
    plot_path = args.figures_dir / f"baseline_predictions_{args.split}_{args.target}.png"
    plot_predictions(test_df, predictions, args.target, plot_path)
    print(f"saved: {plot_path}")

    return metrics_df


def parse_args() -> argparse.Namespace:
    """
    8. 터미널 옵션을 정의합니다.

    자주 쓰는 옵션:
        --target rul_capped_125
            RUL을 125로 제한한 정답을 예측합니다. 기본값입니다.

        --split random
            빠른 sanity check입니다. 기본값입니다.

        --split by_experiment
            과제 보고서에 더 적합한 평가 방식입니다.
    """
    root = project_root()
    parser = argparse.ArgumentParser(description="Train beginner baseline RUL models.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=root / "artifacts" / "01_make_features" / "outputs" / "tables" / "features_combined.csv",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=root / "artifacts" / "02_train_baseline" / "outputs" / "tables" / "baseline_metrics.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=root / "artifacts" / "02_train_baseline" / "outputs" / "figures",
    )
    parser.add_argument("--target", choices=["rul_step", "rul_capped_125"], default="rul_capped_125")
    parser.add_argument("--split", choices=["random", "last", "by_experiment"], default="random")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--test-experiment", default="2nd_bearing1")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """
    9. 실제 실행 시작점입니다.

    Python 파일을 실행하면 맨 아래 if __name__ == "__main__"에서 main()을 호출하고,
    main()은 train_and_evaluate()를 실행합니다.
    """
    train_and_evaluate(parse_args())

    print("\nInterpretation tips:")
    print("- random split is a sanity check, not a final experimental protocol.")
    print("- last split is harder because the model must predict late degradation from earlier data.")
    print("- by_experiment split is better for a final report, but needs features from multiple experiments.")


if __name__ == "__main__":
    main()
