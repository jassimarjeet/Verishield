"""Download a c23 training subsample through the authors' script.

The script is not in the public GitHub repository. The authors email it after
the access form. This module refuses to run until that form has been accepted,
and it refuses YouTube source zips, raw video, c40, and DeepFakeDetection.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

from src.data.ffpp_splits import load_frozen_regime, pair_ids

TERMS_FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform"
)
OFFICIAL_REPO = "https://github.com/ondyari/FaceForensics"

ALLOWED_DATASETS = (
    "original",
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)
BLOCKED_DATASETS = (
    "original_youtube_videos",
    "original_youtube_videos_info",
    "DeepFakeDetection",
    "DeepFakeDetection_original",
)


def load_author_script(script_path: Path):
    script_path = Path(script_path)
    if not script_path.is_file():
        raise SystemExit(
            "The authors' download script is not at "
            f"{script_path}. Upload the file they emailed after the access form. "
            "This run does not scrape YouTube and does not substitute another downloader."
        )
    spec = importlib.util.spec_from_file_location("ffpp_authors_download", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load the authors' script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def author_download_args(module, script_path: Path, output_dir: Path, server: str):
    if not hasattr(module, "parse_args"):
        raise SystemExit(
            "The uploaded file has no parse_args. It does not look like the authors' "
            "download-FaceForensics.py. No other downloader will be used."
        )
    argv = [
        str(script_path),
        str(output_dir),
        "-d",
        "original",
        "-c",
        "c23",
        "-t",
        "videos",
        "--server",
        server,
    ]
    previous = sys.argv
    sys.argv = argv
    try:
        args = module.parse_args()
    finally:
        sys.argv = previous
    if not getattr(args, "base_url", None) or not getattr(args, "tos_url", None):
        raise SystemExit(
            "The authors' script did not provide base_url and tos_url. "
            f"Open the access form again: {TERMS_FORM_URL}"
        )
    return args


def manipulation_stems(pair: list[str]) -> list[str]:
    left, right = pair
    return [f"{left}_{right}", f"{right}_{left}"]


def stems_for_dataset(dataset: str, pairs: list[list[str]]) -> list[str]:
    if dataset in BLOCKED_DATASETS:
        raise SystemExit(f"Refusing dataset {dataset}. YouTube source zips and DeepFakeDetection are not part of this run.")
    if dataset not in ALLOWED_DATASETS:
        raise SystemExit(f"Refusing dataset {dataset}.")
    if dataset == "original":
        stems = []
        for pair in pairs:
            stems.extend(pair)
        return stems
    stems = []
    for pair in pairs:
        stems.extend(manipulation_stems(pair))
    return stems


def download_c23_subset(script_path: Path, output_dir: Path, server: str, terms_accepted: bool, tos_confirmed: bool) -> None:
    if not terms_accepted:
        raise SystemExit(
            "Stopped. Submit the FaceForensics++ access form before any download:\n"
            f"{TERMS_FORM_URL}"
        )
    regime = load_frozen_regime()
    pairs = regime["train_pairs"] + regime["monitor_pairs"]
    ids = pair_ids(pairs)
    from src.data.ffpp_splits import assert_train_ids_only

    assert_train_ids_only(ids)
    module = load_author_script(script_path)
    datasets = getattr(module, "DATASETS", None)
    if not isinstance(datasets, dict) or not hasattr(module, "download_files"):
        raise SystemExit(
            "The uploaded file is missing DATASETS or download_files. "
            "No YouTube scraper will be used instead."
        )
    args = author_download_args(module, Path(script_path), Path(output_dir), server)
    print("Authors' terms URL from their script:")
    print(args.tos_url)
    if not tos_confirmed:
        raise SystemExit(
            "Stopped until tos_confirmed is True after you have read the terms URL above."
        )
    output_dir = Path(output_dir)
    for dataset in ALLOWED_DATASETS:
        if dataset not in datasets:
            raise SystemExit(f"The authors' script has no dataset entry for {dataset}.")
        dataset_path = datasets[dataset]
        stems = stems_for_dataset(dataset, pairs)
        filenames = [f"{stem}.mp4" for stem in stems]
        base_url = args.base_url + f"{dataset_path}/c23/videos/"
        destination = output_dir / dataset_path / "c23" / "videos"
        print(f"Downloading {len(filenames)} c23 files for {dataset} into {destination}")
        signature = inspect.signature(module.download_files)
        extra = {}
        if "report_progress" in signature.parameters:
            extra["report_progress"] = True
        module.download_files(filenames, base_url, str(destination), **extra)
