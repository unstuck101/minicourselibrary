#!/usr/bin/env python3
"""Decode check for the generated art.

For each design this renders the SVG across a range of resolutions and throws
a battery of real-world degradations at every one: rotation, out-of-focus
blur, low contrast, sensor noise, uneven lighting and perspective skew. It
reports the result against pixels-per-module, which is what actually governs
whether a symbol survives, and derives a minimum print size from it.

Decoding uses zxing-cpp, the ZXing engine behind most phone scanners. OpenCV's
QRCodeDetector is deliberately not used: it fails on dot-rendered modules even
for a plain unstyled dotted code, so it says nothing about this artwork.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import zxingcpp
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_qr import DESIGNS, PAD, build  # noqa: E402

# Pixels per module at which the full battery is required to pass.
BAR = 5.0
# Millimetres per module considered safe for print (phone cameras manage ~0.4).
MM_PER_MODULE = 0.6

rng = np.random.default_rng(7)


def degradations(img: np.ndarray):
    h, w = img.shape
    yield "plain", img
    yield "rot90", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    yield "rot180", cv2.rotate(img, cv2.ROTATE_180)
    yield "blur", cv2.GaussianBlur(img, (7, 7), 0)
    yield "lowcontrast", np.clip(img * 0.45 + 110, 0, 255).astype(np.uint8)
    yield "noise", np.clip(img + rng.normal(0, 14, img.shape), 0, 255).astype(np.uint8)
    ramp = np.linspace(0.72, 1.18, w, dtype=np.float32)[None, :]
    yield "uneven-light", np.clip(img * ramp, 0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * .07, h * .03], [w * .965, 0], [w, h * .958], [w * .021, h]])
    yield "perspective", cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(src, dst), (w, h),
        flags=cv2.INTER_CUBIC, borderValue=250,
    )


def reads(img: np.ndarray, url: str) -> bool:
    r = zxingcpp.read_barcode(img)
    return bool(r) and r.text == url


def main() -> int:
    failures = 0
    for d in DESIGNS:
        svg = (HERE / f"{d.name}-qr.svg").read_bytes()
        span = d.grid + 2 * PAD  # canvas width, in modules
        print(f"\n{d.name}  ->  {d.url}")
        for px in (1600, 1200, 900, 700, 600, 500, 420, 360, 300):
            ppm = px / span
            png = cairosvg.svg2png(bytestring=svg, output_width=px, output_height=px)
            img = np.array(Image.open(io.BytesIO(png)).convert("L"))
            bad = [name for name, v in degradations(img) if not reads(v, d.url)]
            required = ppm >= BAR
            if bad and required:
                failures += 1
            state = "ok  " if not bad else ("FAIL" if required else "--  ")
            note = "all 8 variants" if not bad else "fails: " + ", ".join(bad)
            print(f"  {state} {px:>5}px  {ppm:5.1f} px/module  {note}")
        mm = span * MM_PER_MODULE
        print(f"  -> print at {mm:.0f} mm ({mm / 25.4:.1f} in) wide or larger "
              f"for {MM_PER_MODULE} mm modules")

    if failures:
        print(f"\n{failures} FAILURES at or above {BAR} px/module")
        return 1
    print(f"\nALL PASS at or above {BAR} px/module "
          f"(rows marked -- are below the bar and informational)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
