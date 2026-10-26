"""The example run config must carry every key named in the schema."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs" / "schema.md"
EXAMPLE = ROOT / "configs" / "examples" / "xception-face-c23.yaml"


def required_keys() -> list[str]:
    text = SCHEMA.read_text()
    match = re.search(r"```text\n(.*?)```", text, re.DOTALL)
    if match is None:
        raise AssertionError("configs/schema.md has no key block")
    keys = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        keys.append(stripped.split()[0])
    if not keys:
        raise AssertionError("configs/schema.md key block is empty")
    return keys


def load_example() -> dict:
    loaded = yaml.safe_load(EXAMPLE.read_text())
    if not isinstance(loaded, dict):
        raise AssertionError("example config must be a mapping")
    return loaded


def assert_required_keys(config: dict, keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise AssertionError(f"missing required keys: {missing}")


def test_example_has_every_schema_key():
    config = load_example()
    keys = required_keys()
    assert_required_keys(config, keys)
    assert list(config) == keys


def test_missing_required_key_fails():
    config = load_example()
    keys = required_keys()
    del config[keys[0]]
    with pytest.raises(AssertionError, match="missing required keys"):
        assert_required_keys(config, keys)
