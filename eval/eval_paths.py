# -*- coding: utf-8 -*-
"""
Portable dataset/model path defaults for benchmark eval scripts.

After ``source examples/starkv_local_paths.sh`` (or a custom
``starkv_local_paths.env``), CLI flags default from environment variables so
launch scripts do not need hardcoded machine paths.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional, Sequence


def _env(*keys: str) -> Optional[str]:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


def add_model_path_argument(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    default = _env("MODEL_PATH", "UITARS_MODEL_PATH")
    parser.add_argument(
        "--model_path",
        type=str,
        default=default,
        required=required and default is None,
        help="HF model directory (default: MODEL_PATH or UITARS_MODEL_PATH from starkv_local_paths.sh).",
    )


def add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    default = _env("STARKV_RESULTS_DIR")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=default,
        help="Output directory (default: STARKV_RESULTS_DIR from starkv_local_paths.sh).",
    )


def add_screenspot_pro_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--screenspot_imgs",
        type=str,
        default=_env("SCREENSPOTPRO_IMGS"),
        help="ScreenSpot-Pro images root (default: SCREENSPOTPRO_IMGS).",
    )
    parser.add_argument(
        "--screenspot_test",
        type=str,
        default=_env("SCREENSPOTPRO_TEST"),
        help="ScreenSpot-Pro annotations (default: SCREENSPOTPRO_TEST).",
    )


def add_screenspot_v2_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--screenspot_imgs",
        type=str,
        default=_env("SCREENSPOTV2_IMGS"),
        help="ScreenSpot-v2 images root (default: SCREENSPOTV2_IMGS).",
    )
    parser.add_argument(
        "--screenspot_test",
        type=str,
        default=_env("SCREENSPOTV2_TEST"),
        help="ScreenSpot-v2 annotations (default: SCREENSPOTV2_TEST).",
    )


def add_androidcontrol_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--androidcontrol_imgs",
        type=str,
        default=_env("ANDROIDCONTROL_IMGS"),
        help="AndroidControl images (default: ANDROIDCONTROL_IMGS).",
    )
    parser.add_argument(
        "--androidcontrol_test",
        type=str,
        default=_env("ANDROIDCONTROL_TEST"),
        help="AndroidControl test JSON dir (default: ANDROIDCONTROL_TEST).",
    )


def add_agentnetbench_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agentnetbench_data",
        type=str,
        default=_env("ANB_DATA"),
        help="AgentNetBench data dir (default: ANB_DATA).",
    )
    parser.add_argument(
        "--agentnetbench_imgs",
        type=str,
        default=_env("ANB_IMGS"),
        help="AgentNetBench images (default: ANB_IMGS).",
    )


def validate_required_paths(args: argparse.Namespace, fields: Iterable[str]) -> None:
    missing = [f for f in fields if not getattr(args, f, None)]
    if missing:
        raise SystemExit(
            "Missing required path argument(s): "
            + ", ".join(missing)
            + "\nPass them on the command line or configure via:\n"
            "  export STARKV_MODEL_DIR / STARKV_DATASETS_DIR, or create examples/starkv_local_paths.env\n"
            "  source examples/starkv_local_paths.sh"
        )


def resolve_opencua_model_path(args: argparse.Namespace) -> None:
    """Use OPENCUA_MODEL_PATH when model_path looks like default UI-TARS."""
    opencua = _env("OPENCUA_MODEL_PATH")
    if not opencua:
        return
    mp = getattr(args, "model_path", None) or ""
    uitars = _env("UITARS_MODEL_PATH") or ""
    if "OpenCUA" in mp or (uitars and os.path.normpath(mp) == os.path.normpath(uitars)):
        args.model_path = opencua
