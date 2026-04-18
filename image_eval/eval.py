#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Sequence, Tuple


try:
    from PIL import Image, ImageOps  # type: ignore
except ImportError:
    Image = None
    ImageOps = None


@dataclass(frozen=True)
class LoadedImage:
    width: int
    height: int
    channels: int
    pixels: Tuple[float, ...]


def _read_token(handle: BinaryIO) -> bytes:
    token = bytearray()

    while True:
        ch = handle.read(1)
        if not ch:
            break
        if ch == b"#":
            handle.readline()
            continue
        if ch.isspace():
            continue
        token.extend(ch)
        break

    while True:
        ch = handle.read(1)
        if not ch or ch.isspace():
            break
        token.extend(ch)

    if not token:
        raise ValueError("Unexpected end of file while reading image header.")

    return bytes(token)


def _validate_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Image resolution must be positive.")


def _load_pnm(path: Path) -> LoadedImage:
    with path.open("rb") as handle:
        magic = _read_token(handle)
        if magic not in {b"P5", b"P6"}:
            raise ValueError(f"Unsupported PNM format {magic.decode('ascii', errors='replace')}.")

        width = int(_read_token(handle))
        height = int(_read_token(handle))
        max_value = int(_read_token(handle))

        _validate_size(width, height)
        if not (0 < max_value <= 65535):
            raise ValueError("PNM max value must be in the range [1, 65535].")

        channels = 3 if magic == b"P6" else 1
        bytes_per_sample = 1 if max_value < 256 else 2
        raw = handle.read(width * height * channels * bytes_per_sample)
        if len(raw) != width * height * channels * bytes_per_sample:
            raise ValueError("PNM image data is truncated.")

        pixels: List[float] = []
        if bytes_per_sample == 1:
            pixels.extend(value / max_value for value in raw)
        else:
            for index in range(0, len(raw), 2):
                value = int.from_bytes(raw[index : index + 2], byteorder="big", signed=False)
                pixels.append(value / max_value)

    return LoadedImage(width=width, height=height, channels=channels, pixels=tuple(pixels))


def _load_pfm(path: Path) -> LoadedImage:
    with path.open("rb") as handle:
        magic = _read_token(handle)
        if magic not in {b"PF", b"Pf"}:
            raise ValueError(f"Unsupported PFM format {magic.decode('ascii', errors='replace')}.")

        width = int(_read_token(handle))
        height = int(_read_token(handle))
        scale = float(_read_token(handle))

        _validate_size(width, height)
        if scale == 0:
            raise ValueError("PFM scale must be non-zero.")

        channels = 3 if magic == b"PF" else 1
        data = handle.read()
        count = width * height * channels
        expected = count * 4
        if len(data) != expected:
            raise ValueError("PFM image data size does not match header.")

        endian = "<" if scale < 0 else ">"
        unpacked = struct.unpack(f"{endian}{count}f", data)

        # PFM stores scanlines bottom-to-top; flip so all loaders use top-to-bottom.
        row_size = width * channels
        pixels: List[float] = []
        for row in range(height - 1, -1, -1):
            start = row * row_size
            pixels.extend(unpacked[start : start + row_size])

    return LoadedImage(width=width, height=height, channels=channels, pixels=tuple(pixels))


def _load_with_pillow(path: Path) -> LoadedImage:
    if Image is None or ImageOps is None:
        raise RuntimeError(
            "Pillow is required to read this image format. "
            "Install it with `python3 -m pip install pillow`."
        )

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        bands = image.getbands()
        image = image.convert("L" if len(bands) == 1 else "RGBA" if "A" in bands else "RGB")

        channels = len(image.getbands())
        pixels: List[float] = []
        for value in image.getdata():
            if isinstance(value, int):
                pixels.append(value / 255.0)
            else:
                pixels.extend(component / 255.0 for component in value)

        return LoadedImage(
            width=image.width,
            height=image.height,
            channels=channels,
            pixels=tuple(pixels),
        )


def load_image(path_str: str) -> LoadedImage:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pfm":
        return _load_pfm(path)
    if suffix in {".ppm", ".pgm"}:
        return _load_pnm(path)
    return _load_with_pillow(path)


def compute_relmse(reference: LoadedImage, test: LoadedImage, epsilon: float) -> float:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if reference.width != test.width or reference.height != test.height:
        raise ValueError(
            "Reference image and test image must have the same resolution: "
            f"{reference.width}x{reference.height} vs {test.width}x{test.height}."
        )
    if reference.channels != test.channels:
        raise ValueError(
            "Reference image and test image must have the same channel count: "
            f"{reference.channels} vs {test.channels}."
        )

    total = 0.0
    for ref_value, test_value in zip(reference.pixels, test.pixels):
        diff = test_value - ref_value
        denom = ref_value + epsilon
        total += (diff * diff) / (denom * denom)

    return total / len(reference.pixels) if reference.pixels else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a reference image and a test image and output a RelMSE value. "
            "The metric is mean(((test - reference)^2) / (reference + epsilon)^2)."
        )
    )
    parser.add_argument("reference_image", help="Path to the reference image.")
    parser.add_argument("test_image", help="Path to the test image.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Stability constant used in the denominator. Default: 0.01.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only the numeric RelMSE value.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        reference = load_image(args.reference_image)
        test = load_image(args.test_image)
        relmse = compute_relmse(reference, test, args.epsilon)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.plain:
        print(f"{relmse:.12g}")
    else:
        print(f"RelMSE: {relmse:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
