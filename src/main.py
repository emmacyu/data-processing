"""Skeleton only. Replace, split into modules, or ignore as you like.

Usage (suggested):
    python starter/main.py normalize
    python starter/main.py copy --client northmart
    python starter/main.py evaluate
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"


def load_feed() -> pd.DataFrame:
    return pd.read_csv(DATA / "promo_feed.csv", dtype=str)


def load_brands() -> pd.DataFrame:
    return pd.read_csv(DATA / "brands.csv", dtype=str)


def load_clients() -> dict:
    return yaml.safe_load((ROOT / "config" / "clients.yaml").read_text())


def normalize(feed: pd.DataFrame, brands: pd.DataFrame) -> list[dict]:
    """Part A. Return one normalised record per unique item."""
    raise NotImplementedError


def generate_copy(items: list[dict], client_cfg: dict, brands: pd.DataFrame) -> list[dict]:
    """Part B. Return one entry per item per language."""
    raise NotImplementedError


def validate_copy(entry: dict, client_cfg: dict) -> list[str]:
    """Part C.1. Return a list of rule violations (empty list = OK)."""
    raise NotImplementedError


def evaluate(pred: list[dict], gold: list[dict]) -> dict:
    """Part C.2. Return per-field accuracy."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["normalize", "copy", "evaluate"])
    ap.add_argument("--client", default="northmart")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    # TODO: wire the steps together


if __name__ == "__main__":
    main()
