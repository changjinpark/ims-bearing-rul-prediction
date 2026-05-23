"""
01_make_features.py

이 파일의 역할:
    원본 IMS Bearing 진동 파일을 "모델이 학습할 수 있는 표 형태 CSV"로 바꿉니다.

처음 보면 중요한 관점:
    - 원본 파일 1개 = 특정 시점의 1초짜리 진동 신호입니다.
    - 원본 파일 안에는 20,480개의 숫자가 들어 있습니다.
    - 머신러닝 모델은 이 긴 숫자 배열을 바로 이해하기 어렵습니다.
    - 그래서 파일 1개를 RMS, kurtosis 같은 요약값(feature) 여러 개로 바꿉니다.

전체 코드 흐름:
    1. 데이터 경로와 실험 정보를 정의합니다.
    2. 원본 진동 파일들을 시간 순서대로 읽습니다.
    3. 파일 하나에서 여러 feature를 계산합니다.
    4. 시간 순서로 RUL 라벨을 직접 만듭니다.
    5. feature CSV와 EDA 그래프를 저장합니다.

실행 예:
    python3 src/01_make_features.py
    python3 src/01_make_features.py --experiments all --output-dir artifacts/01_make_features/outputs_all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

# 0-1. 서버/터미널 환경에서는 화면 창을 띄우지 않고 이미지 파일로만 그래프를 저장합니다.
#      이 설정이 없으면 matplotlib가 GUI 창을 열려고 하다가 실패할 수 있습니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 0-2. IMS Bearing 데이터 설명에 따르면 샘플링 주파수는 20 kHz입니다.
#      즉 1초 동안 20,000번 진동을 측정했다는 뜻입니다.
SAMPLE_RATE_HZ = 20_000

# 0-3. 0으로 나누는 상황을 피하기 위한 아주 작은 숫자입니다.
#      예: rms가 0이면 peak_abs / rms 계산이 불가능해집니다.
EPS = 1e-12


@dataclass(frozen=True)
class ExperimentSpec:
    """
    1. 실험 하나를 설명하는 작은 설정 묶음입니다.

    예를 들어 2nd_test에서는 bearing 1이 고장났고, 그 신호는 1번째 채널에 있습니다.
    이런 정보를 코드 곳곳에 흩뿌리지 않고 한 군데에 모아두기 위해 사용합니다.
    """

    name: str
    data_dir: Path
    usecols: tuple[int, ...]
    signal_mode: str
    failed_bearing: str
    failure_description: str


def project_root() -> Path:
    """2. 현재 프로젝트 폴더인 rul_project 경로를 구합니다."""
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    """
    3. 원본 데이터 archive 폴더 위치를 구합니다.

    현재 파일은 rul_project/src 안에 있고,
    원본 데이터는 rul_project의 부모 폴더 아래 archive에 있습니다.
    """
    return project_root().parent / "archive"


def parse_timestamp(path: Path) -> datetime:
    """
    4. 파일 이름을 날짜/시간으로 바꿉니다.

    예:
        2004.02.12.10.32.39 -> 2004-02-12 10:32:39

    IMS Bearing 데이터에서는 파일 이름이 측정 시각입니다.
    """
    return datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S")


def list_signal_files(data_dir: Path, limit: int | None = None) -> list[Path]:
    """
    5. 실험 폴더 안의 진동 파일들을 시간 순서대로 가져옵니다.

    중요한 점:
        - RUL은 시간 순서가 핵심입니다.
        - 그래서 반드시 파일 이름 기준으로 정렬해야 합니다.
        - .DS_Store 같은 숨김 파일은 제외합니다.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(path for path in data_dir.iterdir() if path.is_file() and not path.name.startswith("."))
    if limit is not None:
        files = files[:limit]
    return files


def build_experiments(data_root: Path) -> dict[str, ExperimentSpec]:
    """
    6. 이 프로젝트에서 사용할 실험 목록을 만듭니다.

    중요한 점:
        여기서는 모든 베어링을 다 쓰지 않습니다.
        IMS Bearing Dataset 설명에서 "실험 마지막에 실제 고장이 발생했다"고 설명된 베어링만 고릅니다.

    이유:
        RUL 라벨은 "고장까지 얼마나 남았는가"입니다.
        실험 끝까지 고장나지 않은 베어링은 정확한 고장 시점을 모르기 때문에,
        이 단순 RUL 회귀 baseline에는 넣지 않았습니다.

    PDF 설명 기준:
        - 1st_test: bearing 3, bearing 4 고장
        - 2nd_test: bearing 1 고장
        - 3rd_test: bearing 3 고장

    usecols는 0부터 시작합니다.
        - usecols=(0,)은 첫 번째 채널을 뜻합니다.
        - usecols=(4, 5)는 5번째, 6번째 채널을 뜻합니다.

    1st_test는 bearing마다 x축/y축 센서가 2개라서 magnitude로 합쳐서 씁니다.
    """
    third_dir = data_root / "3rd_test" / "4th_test" / "txt"
    if not third_dir.exists():
        third_dir = data_root / "3rd_test" / "3rd_test"

    return {
        "2nd_bearing1": ExperimentSpec(
            name="2nd_bearing1",
            data_dir=data_root / "2nd_test" / "2nd_test",
            usecols=(0,),
            signal_mode="single",
            failed_bearing="bearing1",
            failure_description="outer race failure",
        ),
        "3rd_bearing3": ExperimentSpec(
            name="3rd_bearing3",
            data_dir=third_dir,
            usecols=(2,),
            signal_mode="single",
            failed_bearing="bearing3",
            failure_description="outer race failure",
        ),
        "1st_bearing3": ExperimentSpec(
            name="1st_bearing3",
            data_dir=data_root / "1st_test" / "1st_test",
            usecols=(4, 5),
            signal_mode="magnitude",
            failed_bearing="bearing3",
            failure_description="inner race defect",
        ),
        "1st_bearing4": ExperimentSpec(
            name="1st_bearing4",
            data_dir=data_root / "1st_test" / "1st_test",
            usecols=(6, 7),
            signal_mode="magnitude",
            failed_bearing="bearing4",
            failure_description="roller element defect",
        ),
    }


def load_signal(path: Path, usecols: tuple[int, ...], signal_mode: str) -> np.ndarray:
    """
    7. 원본 진동 파일 하나를 numpy 배열로 읽습니다.

    입력 파일 모양:
        - 행: 시간 순서의 진동 측정값, 보통 20,480행
        - 열: 센서 채널

    signal_mode:
        - single: 채널 하나만 그대로 사용합니다.
        - magnitude: x축/y축 두 채널을 하나의 크기값으로 합칩니다.

    magnitude를 쓰는 이유:
        x축과 y축 센서가 따로 있을 때, 두 방향의 진동을 하나의 대표 진동 크기로 보기 위해서입니다.
    """
    data = np.loadtxt(path, delimiter="\t", usecols=usecols)
    if data.ndim == 1:
        return data.astype(np.float64)

    if signal_mode == "magnitude":
        return np.sqrt(np.sum(data.astype(np.float64) ** 2, axis=1))

    return data[:, 0].astype(np.float64)


def band_energy(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    """
    8. 특정 주파수 구간의 에너지를 계산합니다.

    쉽게 말하면:
        "0~1kHz 구간에 진동 에너지가 얼마나 있나?"
        "6~10kHz 구간에 진동 에너지가 얼마나 있나?"

    베어링 고장은 특정 주파수 대역의 진동 에너지를 키울 수 있어서,
    이런 feature가 RUL 예측에 도움이 될 수 있습니다.
    """
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def extract_features(signal: np.ndarray) -> dict[str, float]:
    """
    9. 진동 신호 1개를 여러 feature로 요약합니다.

    이 함수가 가장 중요합니다.

    원래 입력:
        [0.056, 0.051, 0.105, ...]처럼 긴 진동 숫자 20,480개

    바뀐 출력:
        rms=0.07, kurtosis=3.6, crest_factor=6.1 ... 같은 작은 표 1행

    왜 이렇게 하나요?
        머신러닝 모델은 "파일 하나가 어떤 상태인지"를 숫자 feature로 받아야 합니다.
        RMS, kurtosis 같은 값은 고장 징후를 요약하는 건강 지표처럼 볼 수 있습니다.
    """
    n = len(signal)

    # 9-1. 시간영역 기본 통계입니다.
    #      시간영역은 FFT를 하기 전, 원래 진동 파형에서 바로 계산하는 값입니다.
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    centered = signal - mean
    abs_signal = np.abs(signal)

    # 9-2. 진동 크기를 나타내는 대표 feature입니다.
    #      RMS가 커진다는 것은 전체 진동 에너지가 커진다는 뜻으로 해석할 수 있습니다.
    rms = float(np.sqrt(np.mean(signal**2)))
    abs_mean = float(np.mean(abs_signal))
    peak_abs = float(np.max(abs_signal))
    peak_to_peak = float(np.ptp(signal))

    # 9-3. 신호 모양을 나타내는 feature입니다.
    #      kurtosis는 튀는 충격성 신호가 많아질수록 커지는 경향이 있습니다.
    if std > EPS:
        z = centered / std
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    # 9-4. 주파수영역 feature를 만들기 위해 FFT를 합니다.
    #      FFT는 "시간에 따른 진동"을 "주파수별 진동"으로 바꿔주는 도구라고 보면 됩니다.
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)
    magnitude = np.abs(spectrum) / n
    power = magnitude**2
    power_sum = float(np.sum(power))

    # 9-5. 가장 강하게 나타나는 주파수를 찾습니다.
    #      0Hz는 평균값에 가까운 성분이라 제외하고 찾습니다.
    if len(magnitude) > 1:
        dominant_idx = int(np.argmax(magnitude[1:]) + 1)
        dominant_freq = float(freqs[dominant_idx])
    else:
        dominant_freq = 0.0

    # 9-6. spectral centroid는 주파수 에너지의 중심입니다.
    #      값이 커지면 상대적으로 높은 주파수 성분이 강해졌다고 볼 수 있습니다.
    spectral_centroid = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + EPS))

    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "abs_mean": abs_mean,
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "peak_abs": peak_abs,
        "peak_to_peak": peak_to_peak,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": peak_abs / (rms + EPS),
        "impulse_factor": peak_abs / (abs_mean + EPS),
        "shape_factor": rms / (abs_mean + EPS),
        "clearance_factor": peak_abs / ((np.mean(np.sqrt(abs_signal)) ** 2) + EPS),
        "dominant_freq_hz": dominant_freq,
        "spectral_centroid_hz": spectral_centroid,
        "fft_total_power": power_sum,
        "band_energy_0_1k": band_energy(freqs, power, 0, 1_000),
        "band_energy_1k_3k": band_energy(freqs, power, 1_000, 3_000),
        "band_energy_3k_6k": band_energy(freqs, power, 3_000, 6_000),
        "band_energy_6k_10k": band_energy(freqs, power, 6_000, 10_000),
    }


def health_stage(step: int, total: int) -> str:
    """
    10. 아주 단순한 상태 라벨을 만듭니다.

    이 값은 모델 학습의 최종 정답으로 쓰는 핵심 라벨은 아닙니다.
    EDA나 블로그 설명에서 "초반/중반/후반"을 쉽게 구분하려고 넣은 보조 라벨입니다.
    """
    ratio = step / max(total - 1, 1)
    if ratio < 0.6:
        return "normal"
    if ratio < 0.8:
        return "transition"
    return "near_failure"


def make_features_for_experiment(spec: ExperimentSpec, limit: int | None = None) -> pd.DataFrame:
    """
    11. 실험 하나를 통째로 feature 표로 바꿉니다.

    핵심 변환:
        원본 파일 984개
            -> feature 표 984행

    한 행의 의미:
        특정 시점의 베어링 상태 1개
    """
    files = list_signal_files(spec.data_dir, limit=limit)
    if not files:
        raise ValueError(f"No signal files found in {spec.data_dir}")

    rows = []
    total = len(files)
    print(f"[{spec.name}] extracting {total} files from {spec.data_dir}")

    for step, path in enumerate(files):
        # 11-1. 파일 하나를 읽습니다.
        signal = load_signal(path, spec.usecols, spec.signal_mode)

        # 11-2. 파일 하나에서 feature 여러 개를 계산합니다.
        features = extract_features(signal)

        if step % 100 == 0 or step == total - 1:
            print(f"[{spec.name}] {step + 1}/{total}: {path.name}")

        # 11-3. RUL 라벨을 직접 만듭니다.
        #
        # 예를 들어 파일이 총 984개라면:
        #     첫 파일 step=0    -> rul_step=983
        #     마지막 파일 step=983 -> rul_step=0
        #
        # 즉 고장 시점에 가까워질수록 RUL이 작아집니다.
        #
        # 중요:
        #     이 값은 원본 데이터나 PDF에 들어 있던 정답지가 아닙니다.
        #     "마지막 측정 파일을 failure time으로 본다"는 가정으로
        #     우리가 코드에서 직접 만든 RUL 라벨입니다.
        rul_step = total - 1 - step

        rows.append(
            {
                "experiment": spec.name,
                "failed_bearing": spec.failed_bearing,
                "failure_description": spec.failure_description,
                "file_name": path.name,
                "timestamp": parse_timestamp(path),
                "step": step,
                "n_files": total,
                "rul_step": rul_step,
                "rul_minutes_approx": rul_step * 10,
                # 11-4. capped RUL은 너무 먼 미래의 RUL을 모두 125로 제한합니다.
                #
                # 이유:
                #     고장까지 900 step 남았는지 800 step 남았는지는 초반 정상 구간에서는
                #     진동만 보고 구분하기 어렵습니다.
                #     그래서 초반은 "아직 충분히 정상"이라는 의미로 125에 묶어둡니다.
                "rul_capped_125": min(rul_step, 125),
                "health_stage_simple": health_stage(step, total),
                **features,
            }
        )

    return pd.DataFrame(rows)


def plot_feature_trends(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    12. feature가 시간에 따라 어떻게 변하는지 그래프로 저장합니다.

    이 그래프를 보는 이유:
        모델을 만들기 전에 "고장에 가까워질수록 진동 feature가 실제로 변하나?"
        를 눈으로 확인하기 위해서입니다.

    블로그에서 가장 먼저 보여주기 좋은 그림입니다.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    for experiment, part in df.groupby("experiment", sort=False):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        trend_columns = ["rms", "kurtosis", "crest_factor", "fft_total_power"]
        for ax, col in zip(axes, trend_columns):
            ax.plot(part["step"], part[col], linewidth=1.1)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("File order / time step")
        fig.suptitle(f"{experiment} feature trends")
        fig.tight_layout()
        output_path = figures_dir / f"{experiment}_feature_trends.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        print(f"saved: {output_path}")


def parse_args() -> argparse.Namespace:
    """
    13. 터미널에서 받을 옵션을 정의합니다.

    옵션을 쓰지 않으면 beginner 모드입니다.
    beginner 모드는 처음 학습하기 쉬운 2nd_test bearing1만 사용합니다.
    """
    parser = argparse.ArgumentParser(description="Extract IMS bearing features and simple RUL labels.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--output-dir", type=Path, default=project_root() / "artifacts" / "01_make_features" / "outputs")
    parser.add_argument(
        "--experiments",
        choices=["beginner", "all"],
        default="beginner",
        help="beginner uses only 2nd_test bearing1. all extracts failed bearings from all tests.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional quick debug limit per experiment.")
    return parser.parse_args()


def main() -> None:
    """
    14. 실제 실행 흐름입니다.

    이 파일을 실행하면 Python은 아래 순서대로 움직입니다.
        14-1. 옵션을 읽습니다.
        14-2. 저장 폴더를 만듭니다.
        14-3. 사용할 실험 목록을 고릅니다.
        14-4. 실험별 feature CSV를 저장합니다.
        14-5. 전체 feature CSV를 저장합니다.
        14-6. EDA 그래프를 저장합니다.
    """
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    experiments = build_experiments(args.data_root)
    selected_names = ["2nd_bearing1"] if args.experiments == "beginner" else list(experiments.keys())

    frames = []
    for name in selected_names:
        frame = make_features_for_experiment(experiments[name], limit=args.limit)
        experiment_output = tables_dir / f"features_{name}.csv"
        frame.to_csv(experiment_output, index=False)
        print(f"saved: {experiment_output}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined_output = tables_dir / "features_combined.csv"
    combined.to_csv(combined_output, index=False)
    print(f"saved: {combined_output}")

    plot_feature_trends(combined, figures_dir)

    print("\nDone. Next step:")
    print("python3 src/02_train_baseline.py")


if __name__ == "__main__":
    main()
