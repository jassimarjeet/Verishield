"""Download the Regime S c23 videos through FaceForensics++ script v4.

Training uses https://kaldir.vc.in.tum.de/faceforensics_download_v4.py.
The original FaceForensics script is v1 and is not called here.

v4's ``main`` prints its terms URL and waits on ``input``. This module calls
``main`` so that check still runs. It does not add a second form, and it does
not answer the prompt. After the prompt, only the frozen Regime S c23 names
are passed to v4's ``download_files``.
"""

import importlib.util
import sys
from pathlib import Path

from src.data.ffpp_splits import assert_train_ids_only, load_frozen_regime, pair_ids

V4_URL = "https://kaldir.vc.in.tum.de/faceforensics_download_v4.py"
V1_URL = "https://kaldir.vc.in.tum.de/faceforensics_download_v1.py"
V4_SERVERS = ("EU", "EU2", "CA")

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
    "FaceShifter",
    "all",
)


def v4_argv(script_path: Path, output_dir: Path, dataset: str, server: str) -> list[str]:
    if dataset in BLOCKED_DATASETS or dataset not in ALLOWED_DATASETS:
        raise SystemExit(
            f"Refusing v4 dataset {dataset}. This run does not scrape YouTube, "
            "does not download raw or c40, and does not download DeepFakeDetection or FaceShifter."
        )
    if server not in V4_SERVERS:
        raise SystemExit(f"v4 --server must be one of {V4_SERVERS}.")
    return [
        str(script_path),
        str(output_dir),
        "-d",
        dataset,
        "-c",
        "c23",
        "-t",
        "videos",
        "--server",
        server,
    ]


def manipulation_stems(pair: list[str]) -> list[str]:
    left, right = pair
    return [f"{left}_{right}", f"{right}_{left}"]


def stems_for_dataset(dataset: str, pairs: list[list[str]]) -> list[str]:
    if dataset in BLOCKED_DATASETS or dataset not in ALLOWED_DATASETS:
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


def regime_filenames(pairs: list[list[str]]) -> set[str]:
    names = set()
    for dataset in ALLOWED_DATASETS:
        names.update(f"{stem}.mp4" for stem in stems_for_dataset(dataset, pairs))
    return names


def select_regime_filenames(filenames, allowed_mp4: set[str]) -> list[str]:
    return [name for name in filenames if Path(str(name)).name in allowed_mp4]


def load_author_script(script_path: Path):
    script_path = Path(script_path)
    if not script_path.is_file():
        raise SystemExit(
            f"v4 is not at {script_path}. Fetch {V4_URL} before calling it. "
            f"{V1_URL} is the original FaceForensics script and is not the training download."
        )
    spec = importlib.util.spec_from_file_location("faceforensics_download_v4", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_v4_args(module, argv: list[str]):
    if not hasattr(module, "parse_args") or not hasattr(module, "main"):
        raise SystemExit(
            "The fetched file has no v4 parse_args/main. "
            f"Training does not fall back to {V1_URL}."
        )
    previous = sys.argv
    sys.argv = argv
    try:
        args = module.parse_args()
    finally:
        sys.argv = previous
    if getattr(args, "compression", None) != "c23" or getattr(args, "type", None) != "videos":
        raise SystemExit("Refusing a v4 invocation that is not -c c23 -t videos.")
    if getattr(args, "dataset", None) not in ALLOWED_DATASETS:
        raise SystemExit(f"Refusing v4 dataset {getattr(args, 'dataset', None)}.")
    return args


def download_c23_subset(script_path: Path, output_dir: Path, server: str = "EU") -> None:
    regime = load_frozen_regime()
    pairs = regime["train_pairs"] + regime["monitor_pairs"]
    assert_train_ids_only(pair_ids(pairs))
    allowed = regime_filenames(pairs)
    module = load_author_script(script_path)
    if not hasattr(module, "download_files"):
        raise SystemExit("v4 is missing download_files. No other downloader will be used.")
    original_download = module.download_files

    def filtered(filenames, base_url, output_path, report_progress=True):
        names = list(filenames)
        kept = select_regime_filenames(names, allowed)
        print(f"v4 listed {len(names)} files; downloading {len(kept)} frozen Regime S c23 files")
        if not kept:
            return None
        return original_download(kept, base_url, output_path, report_progress=report_progress)

    module.download_files = filtered
    try:
        for dataset in ALLOWED_DATASETS:
            argv = v4_argv(script_path, output_dir, dataset, server)
            args = parse_v4_args(module, argv)
            # v4 main prints its terms URL and blocks on input(). Leave that prompt intact.
            module.main(args)
    finally:
        module.download_files = original_download
