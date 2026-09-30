"""Shared inference CLI options (no model or GPU dependencies)."""

import argparse


def nonnegative_int(value):
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from None
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def add_vram_argument(parser):
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=nonnegative_int,
        default=None,
        help=(
            "DiT parameter count to keep on GPU. Omit for adaptive VRAM caching; "
            "0 streams DiT weights from CPU; positive values keep a module prefix "
            "within this parameter budget. Streaming is slower and does not "
            "guarantee that inference fits in VRAM."
        ),
    )
