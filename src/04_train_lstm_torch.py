"""
04_train_lstm_torch.py

이 파일의 역할:
    PyTorch의 nn.LSTM을 사용해서 RUL 예측 모델을 학습합니다.

왜 이 파일을 만들었나:
    - RandomForest/MLP는 한 시점의 feature 행만 보고 RUL을 예측합니다.
    - LSTM은 최근 여러 시점의 feature 흐름을 함께 보고 RUL을 예측합니다.
    - 과제/블로그에서는 이 파일을 "딥러닝 라이브러리를 사용한 LSTM 모델"로 설명하면 됩니다.

처음 보면 중요한 관점:
    - 원본 진동 파일을 다시 읽지 않습니다.
    - 01_make_features.py가 만든 features_combined.csv를 읽습니다.
    - 여러 시점의 feature 행을 window_size 길이로 묶어 sequence 입력을 만듭니다.
    - LSTM은 그 sequence를 보고 마지막 시점의 RUL을 예측합니다.

전체 코드 흐름:
    1. feature CSV를 읽습니다.
    2. train/test 실험을 나눕니다.
    3. feature를 표준화합니다.
    4. 여러 행을 LSTM 입력 sequence로 묶습니다.
    5. train sequence 중 일부를 validation으로 나눕니다.
    6. PyTorch DataLoader를 만듭니다.
    7. nn.LSTM 모델을 학습합니다.
    8. test sequence에서 RUL을 예측합니다.
    9. 성능표, 예측 CSV, 그래프, 모델 파일을 저장합니다.

실행 예:
    .venv/bin/python src/04_train_lstm_torch.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# 0-1. 터미널 실행 환경에서는 화면 창을 띄우지 않고 PNG 파일로 저장합니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 0-2. 아래 컬럼들은 입력 feature가 아니라 설명/정답/관리용 컬럼입니다.
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


def choose_device() -> torch.device:
    """
    2. PyTorch가 사용할 장치를 고릅니다.

    우선순위:
        1. CUDA GPU
        2. Apple Silicon MPS
        3. CPU

    현재 환경에서는 CPU일 가능성이 높습니다. 데이터가 크지 않아서 CPU로도 충분히 실행됩니다.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """3. 같은 설정에서 최대한 비슷한 결과가 나오도록 랜덤 시드를 고정합니다."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """
    4. LSTM 입력으로 사용할 feature 컬럼만 고릅니다.

    features_combined.csv에는 다음이 섞여 있습니다.
        - 입력 feature: rms, kurtosis, fft_total_power 등
        - 정답 y: rul_capped_125 또는 rul_step
        - 설명 컬럼: experiment, file_name 등

    모델에는 입력 feature만 넣어야 합니다.
    """
    excluded = set(META_COLUMNS)
    excluded.add(target)
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def split_by_experiment(df: pd.DataFrame, test_experiment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    5-A. 실험 단위로 train/test를 나눕니다.

    기본값:
        train: 1st_bearing3, 1st_bearing4, 3rd_bearing3
        test : 2nd_bearing1

    중요한 점:
        features_combined.csv 안에 2nd_bearing1이 들어 있어도,
        여기서 test_df로 분리되므로 모델 학습에는 들어가지 않습니다.
    """
    if test_experiment not in set(df["experiment"]):
        choices = ", ".join(sorted(df["experiment"].unique()))
        raise ValueError(f"Unknown test experiment: {test_experiment}. Choices: {choices}")

    train_df = df[df["experiment"] != test_experiment].copy()
    test_df = df[df["experiment"] == test_experiment].copy()
    if train_df.empty:
        raise ValueError("by_experiment split needs at least two experiments.")
    return train_df, test_df


def split_last(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    5-B. 각 실험의 앞부분은 train, 뒷부분은 test로 둡니다.

    이 방식은 시간 순서 split입니다.
    """
    train_parts = []
    test_parts = []
    for _, part in df.sort_values(["experiment", "step"]).groupby("experiment", sort=False):
        cutoff = int(len(part) * (1.0 - test_size))
        cutoff = min(max(cutoff, 1), len(part) - 1)
        train_parts.append(part.iloc[:cutoff].copy())
        test_parts.append(part.iloc[cutoff:].copy())
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def make_split(args: argparse.Namespace, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5-C. 사용자가 고른 split 방식을 실제 함수로 연결합니다."""
    if args.split == "by_experiment":
        return split_by_experiment(df, args.test_experiment)
    if args.split == "last":
        return split_last(df, args.test_size)
    raise ValueError(f"Unsupported split: {args.split}")


def standardize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    6. feature 스케일을 맞춥니다.

    신경망은 입력 숫자의 크기에 민감합니다.
    그래서 train 데이터 기준으로 평균 0, 표준편차 1에 가깝게 변환합니다.

    test 데이터로 scaler를 fit하지 않는 이유:
        test 정보를 미리 보는 데이터 누수를 막기 위해서입니다.
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[cols] = scaler.fit_transform(train_df[cols])
    test_scaled[cols] = scaler.transform(test_df[cols])
    return train_scaled, test_scaled, scaler


def make_sequences(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    7. 여러 행을 LSTM 입력 sequence로 바꿉니다.

    예: window_size = 30

        step 0 ~ 29 feature들  -> step 29의 RUL
        step 1 ~ 30 feature들  -> step 30의 RUL
        step 2 ~ 31 feature들  -> step 31의 RUL

    X shape:
        (sequence 개수, window_size, feature 개수)

    y shape:
        (sequence 개수,)
    """
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    meta_rows: list[dict[str, object]] = []

    meta_cols = [
        "experiment",
        "file_name",
        "timestamp",
        "step",
        "n_files",
        "rul_step",
        "rul_capped_125",
        "health_stage_simple",
    ]
    meta_cols = [col for col in meta_cols if col in df.columns]

    for experiment, part in df.sort_values(["experiment", "step"]).groupby("experiment", sort=False):
        if len(part) < window_size:
            print(f"skip {experiment}: only {len(part)} rows, window_size={window_size}", flush=True)
            continue

        values = part[cols].to_numpy(dtype=np.float32)
        target_values = part[target].to_numpy(dtype=np.float32)

        for end_idx in range(window_size - 1, len(part)):
            start_idx = end_idx - window_size + 1
            sequences.append(values[start_idx : end_idx + 1])
            targets.append(float(target_values[end_idx]))
            meta_rows.append(part.iloc[end_idx][meta_cols].to_dict())

    if not sequences:
        raise ValueError("No sequences were created. Try a smaller --window-size.")

    return np.stack(sequences).astype(np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(meta_rows)


def train_val_indices(n_rows: int, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    8. train sequence 일부를 validation으로 나눕니다.

    test 데이터는 최종 평가용으로만 남겨둡니다.
    validation은 학습 중 과적합 여부를 보는 용도입니다.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    val_count = int(n_rows * val_size)
    val_count = min(max(val_count, 1), n_rows - 1)
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    return train_idx, val_idx


class LstmRegressor(nn.Module):
    """
    9. PyTorch LSTM RUL 예측 모델입니다.

    nn.LSTM:
        sequence를 읽고 hidden state를 만듭니다.

    self.head:
        마지막 hidden state를 RUL 숫자 하나로 바꿉니다.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """10. sequence 입력을 받아 RUL 예측값 하나를 반환합니다."""
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


@dataclass
class TrainArtifacts:
    """11. 학습 결과로 필요한 값들을 묶어서 반환하기 위한 dataclass입니다."""

    model: LstmRegressor
    history: pd.DataFrame
    target_scale: float
    device: torch.device


def make_loader(
    X: np.ndarray,
    y_scaled: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """12. numpy 배열을 PyTorch DataLoader로 바꿉니다."""
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y_scaled).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(
    model: LstmRegressor,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """13. validation loss를 계산합니다."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += float(loss.item()) * len(X_batch)
            total_count += len(X_batch)
    return total_loss / max(total_count, 1)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainArtifacts:
    """
    14. LSTM 모델을 학습합니다.

    target_scale:
        RUL은 0~125 범위입니다.
        신경망 내부에서는 y를 0~1 정도로 줄여 학습하면 더 안정적인 경우가 많습니다.
    """
    train_idx, val_idx = train_val_indices(len(X_train), args.val_size, args.random_state)
    target_scale = max(float(np.max(y_train)), 1.0)
    y_scaled = y_train / target_scale

    train_loader = make_loader(X_train[train_idx], y_scaled[train_idx], args.batch_size, shuffle=True)
    val_loader = make_loader(X_train[val_idx], y_scaled[val_idx], args.batch_size, shuffle=False)

    model = LstmRegressor(
        input_size=X_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history_rows = []
    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(X_batch)
            train_count += len(X_batch)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        improved = val_loss < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs or improved:
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_mse_scaled={train_loss:.5f} "
                f"val_mse_scaled={val_loss:.5f} "
                f"best_val={best_val_loss:.5f} "
                f"patience_left={patience_left}",
                flush=True,
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_mse_scaled": train_loss,
                "val_mse_scaled": val_loss,
                "best_val_mse_scaled": best_val_loss,
            }
        )

        if patience_left <= 0:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainArtifacts(
        model=model,
        history=pd.DataFrame(history_rows),
        target_scale=target_scale,
        device=device,
    )


def predict(
    model: LstmRegressor,
    X: np.ndarray,
    target_scale: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """15. 학습된 LSTM으로 RUL을 예측합니다."""
    loader = make_loader(X, np.zeros(len(X), dtype=np.float32), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            pred_scaled = model(X_batch.to(device)).detach().cpu().numpy()
            preds.append(pred_scaled)

    pred = np.concatenate(preds) * target_scale
    return np.clip(pred, 0.0, target_scale)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """16. MAE, RMSE, R2를 계산합니다."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_predictions(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    output_path: Path,
) -> None:
    """17. test 데이터의 실제 RUL과 LSTM 예측 RUL을 그래프로 저장합니다."""
    order = meta_df.sort_values(["experiment", "step"]).index.to_numpy()
    ordered_true = y_true[order]
    ordered_pred = y_pred[order]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(ordered_true, label="actual", linewidth=2.0, color="black")
    ax.plot(ordered_pred, label="lstm_torch", linewidth=1.3, color="#d62728", alpha=0.9)
    ax.set_title(f"PyTorch LSTM RUL prediction on test sequences ({target})")
    ax.set_xlabel("Test sequences sorted by experiment and time")
    ax.set_ylabel(target)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_history(history_df: pd.DataFrame, output_path: Path) -> None:
    """18. epoch별 train/validation loss를 그래프로 저장합니다."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(history_df["epoch"], history_df["train_mse_scaled"], label="train", linewidth=1.2)
    ax.plot(history_df["epoch"], history_df["val_mse_scaled"], label="validation", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (scaled target)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_model(
    artifacts: TrainArtifacts,
    scaler: StandardScaler,
    cols: list[str],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """19. PyTorch 모델과 scaler 정보를 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "feature_columns": cols,
        "target": args.target,
        "target_scale": artifacts.target_scale,
        "window_size": args.window_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "split": args.split,
        "test_experiment": args.test_experiment,
    }
    torch.save(
        {
            "model_state_dict": artifacts.model.state_dict(),
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "config": config,
        },
        output_path,
    )


def run(args: argparse.Namespace) -> None:
    """
    20. 전체 실행 흐름입니다.

    여기서 실제로 CSV를 읽고, sequence를 만들고, LSTM을 학습하고, 산출물을 저장합니다.
    """
    set_seed(args.random_state)
    device = choose_device()

    df = pd.read_csv(args.features_csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    train_df, test_df = make_split(args, df)
    cols = feature_columns(df, args.target)
    train_scaled, test_scaled, scaler = standardize_features(train_df, test_df, cols)

    X_train, y_train, train_meta = make_sequences(train_scaled, cols, args.target, args.window_size)
    X_test, y_test, test_meta = make_sequences(test_scaled, cols, args.target, args.window_size)

    print(f"device: {device}", flush=True)
    print(f"feature columns: {len(cols)}", flush=True)
    print(f"window_size: {args.window_size}", flush=True)
    print(f"train source rows: {len(train_df)}, test source rows: {len(test_df)}", flush=True)
    print(f"train sequences: {len(X_train)}, test sequences: {len(X_test)}", flush=True)
    print(f"train experiments: {sorted(train_df['experiment'].unique())}", flush=True)
    print(f"test experiments: {sorted(test_df['experiment'].unique())}", flush=True)

    artifacts = train_model(X_train, y_train, args, device)
    pred_test = predict(artifacts.model, X_test, artifacts.target_scale, args.batch_size, device)
    scores = evaluate_predictions(y_test, pred_test)

    name_suffix = f"{args.split}_{args.target}"
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.tables_dir / f"lstm_torch_metrics_{name_suffix}.csv"
    metrics_df = pd.DataFrame(
        [
            {
                "model": "lstm_torch",
                "split": args.split,
                "target": args.target,
                "window_size": args.window_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "epochs_ran": int(artifacts.history["epoch"].max()),
                "device": str(device),
                "train_source_rows": len(train_df),
                "test_source_rows": len(test_df),
                "train_sequences": len(X_train),
                "test_sequences": len(X_test),
                **scores,
            }
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"saved: {metrics_path}", flush=True)

    predictions_path = args.tables_dir / f"lstm_torch_predictions_{name_suffix}.csv"
    predictions_df = test_meta.copy()
    predictions_df["actual"] = y_test
    predictions_df["prediction_lstm_torch"] = pred_test
    predictions_df["absolute_error"] = np.abs(predictions_df["actual"] - predictions_df["prediction_lstm_torch"])
    predictions_df.to_csv(predictions_path, index=False)
    print(f"saved: {predictions_path}", flush=True)

    history_path = args.tables_dir / f"lstm_torch_training_history_{name_suffix}.csv"
    artifacts.history.to_csv(history_path, index=False)
    print(f"saved: {history_path}", flush=True)

    prediction_plot_path = args.figures_dir / f"lstm_torch_predictions_{name_suffix}.png"
    plot_predictions(test_meta, y_test, pred_test, args.target, prediction_plot_path)
    print(f"saved: {prediction_plot_path}", flush=True)

    history_plot_path = args.figures_dir / f"lstm_torch_training_loss_{name_suffix}.png"
    plot_training_history(artifacts.history, history_plot_path)
    print(f"saved: {history_plot_path}", flush=True)

    model_path = args.models_dir / f"lstm_torch_model_{name_suffix}.pt"
    save_model(artifacts, scaler, cols, args, model_path)
    print(f"saved: {model_path}", flush=True)

    config_path = args.models_dir / f"lstm_torch_config_{name_suffix}.json"
    config_path.write_text(
        json.dumps(
            {
                "features_csv": str(args.features_csv),
                "target": args.target,
                "split": args.split,
                "test_experiment": args.test_experiment,
                "window_size": args.window_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "device": str(device),
                "metrics": scores,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {config_path}", flush=True)

    print("\nFinal PyTorch LSTM metrics:", flush=True)
    print(metrics_df[["model", "mae", "rmse", "r2"]].to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    """21. 터미널 옵션을 정의합니다."""
    root = project_root()
    output_root = root / "artifacts" / "04_train_lstm_torch" / "outputs_all"

    parser = argparse.ArgumentParser(description="Train a PyTorch LSTM for IMS Bearing RUL.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=root / "artifacts" / "01_make_features" / "outputs_all" / "tables" / "features_combined.csv",
    )
    parser.add_argument("--tables-dir", type=Path, default=output_root / "tables")
    parser.add_argument("--figures-dir", type=Path, default=output_root / "figures")
    parser.add_argument("--models-dir", type=Path, default=output_root / "models")
    parser.add_argument("--target", choices=["rul_step", "rul_capped_125"], default="rul_capped_125")
    parser.add_argument("--split", choices=["by_experiment", "last"], default="by_experiment")
    parser.add_argument("--test-experiment", default="2nd_bearing1")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """22. Python 파일을 직접 실행했을 때 시작되는 함수입니다."""
    run(parse_args())


if __name__ == "__main__":
    main()
