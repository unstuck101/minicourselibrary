#!/usr/bin/env python3
"""
Wedding QR art generator - Hannah & Travis.

Produces two shaped, scannable QR codes in the style of the reference art:
  * heart  -> The Knot registry
  * camera -> Google Drive photo folder

Design approach
---------------
Clipping a QR symbol into a heart/camera silhouette destroys the finder
patterns and far more codewords than error correction can recover, so the
reference images are decorative only. Here the silhouette is a dot field and
the *real* symbol lives inside it at full size, wrapped in an intact cream
quiet zone. The result keeps the look and actually scans.

Output is SVG (vector, print-ready at any size) plus PNG previews.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import segno
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).resolve().parent
FONT_DIR = Path("/mnt/skills/examples/canvas-design/canvas-fonts")

# ---------------------------------------------------------------- palette ---
CREAM = "#FAF6EE"
INK = "#111111"
GOLD = "#B08D4F"  # decorative gold: outlines, monogram
# Finder rings must binarize as *dark* for scanners, so they use a deeper
# bronze. Luma ~110 against a ~131 threshold between ink and cream.
GOLD_FINDER = "#8C6B2F"
# Alignment rings are the most fragile part of the symbol -- without them a
# code cannot be read at an angle at all -- so they keep full ink contrast.
# Setting this to GOLD_FINDER also passes the scan battery, if you prefer the
# softer look.
ALIGN_COLOR = INK

QUIET = 5  # modules of cream around the symbol (spec minimum is 4)
PAD = 2  # modules of cream between the silhouette and the artboard edge
DOT_R = 0.46  # dot radius as a fraction of one module

# Alignment-pattern centre coordinates per QR version (ISO/IEC 18004 annex E).
ALIGNMENT = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
    7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
    11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62], 14: [6, 26, 46, 66],
    15: [6, 26, 48, 70], 16: [6, 26, 50, 74], 17: [6, 30, 54, 78],
    18: [6, 30, 56, 82], 19: [6, 30, 58, 86], 20: [6, 34, 62, 90],
}

TARGETS = {
    "heart": "https://www.theknot.com/us/hannah-garbutt-and-travis-ala/registry",
    "camera": "https://drive.google.com/drive/folders/1AM0JzJBxKYEtNEiqoe8fBeYp5nqc0ROl",
}


# ----------------------------------------------------------------- shapes ---
def heart_mask(size: int) -> np.ndarray:
    """Implicit heart (x^2+y^2-1)^3 - x^2 y^3 <= 0, with the top cusp opened
    into a proper cleft so the two lobes read at a glance."""
    span_x, y_lo, y_hi = 1.16, -1.30, 1.03
    xs = np.linspace(-span_x, span_x, size)
    ys = np.linspace(y_hi, y_lo, size)
    x, y = np.meshgrid(xs, ys)
    m = ((x * x + y * y - 1.0) ** 3 - x * x * y**3 <= 0).astype(np.uint8)

    to_px = lambda X, Y: (
        int(round((X + span_x) / (2 * span_x) * size)),
        int(round((y_hi - Y) / (y_hi - y_lo) * size)),
    )
    nw, ny, r = 0.34, 0.52, 0.10  # cleft half-width, depth, valley rounding
    cv2.fillPoly(m, [np.array([to_px(-nw, 1.10), to_px(0, ny), to_px(nw, 1.10)], np.int32)], 0)
    cv2.circle(m, to_px(0, ny + r * 1.15), int(round(r / (2 * span_x) * size)), 0, -1)
    return m


def _rounded_rect(mask, x0, y0, x1, y1, r, size):
    px = lambda v: int(round(v * size))
    cv2.rectangle(mask, (px(x0), px(y0 + r)), (px(x1), px(y1 - r)), 1, -1)
    cv2.rectangle(mask, (px(x0 + r), px(y0)), (px(x1 - r), px(y1)), 1, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(mask, (px(cx), px(cy)), px(r), 1, -1)


def camera_mask(size: int) -> np.ndarray:
    """Camera silhouette: body + pentaprism hump + viewfinder bump."""
    m = np.zeros((size, size), np.uint8)
    px = lambda v: int(round(v * size))
    # body
    _rounded_rect(m, 0.005, 0.215, 0.995, 0.975, 0.055, size)
    # pentaprism hump (rounded trapezoid, slanted left shoulder)
    hump = np.array(
        [[0.330, 0.240], [0.372, 0.096], [0.700, 0.096], [0.742, 0.240]], np.float32
    )
    cv2.fillPoly(m, [np.round(hump * size).astype(np.int32)], 1)
    cv2.circle(m, (px(0.392), px(0.116)), px(0.020), 1, -1)
    cv2.circle(m, (px(0.680), px(0.116)), px(0.020), 1, -1)
    cv2.rectangle(m, (px(0.392), px(0.096)), (px(0.680), px(0.140)), 1, -1)
    # small viewfinder bump on the left shoulder
    _rounded_rect(m, 0.128, 0.155, 0.262, 0.260, 0.052, size)
    return m


# -------------------------------------------------------------- mask fit ---
def fit_mask(fn, size: int) -> np.ndarray:
    """Render a silhouette, crop it to its bounding box and centre it in a
    square canvas, so every shape fills the artboard the same way."""
    m = fn(size)
    ys, xs = np.where(m)
    m = m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = m.shape
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    m = (cv2.resize(m.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0.5)
    out = np.zeros((size, size), np.uint8)
    out[(size - nh) // 2 : (size - nh) // 2 + nh, (size - nw) // 2 : (size - nw) // 2 + nw] = m
    return out


# ------------------------------------------------------------- placement ---
def place_square(cells: np.ndarray, need: int, drop: float) -> tuple[int, int]:
    """Top-left cell for a `need`-wide square that lies wholly inside `cells`.

    Horizontally it is centred. Vertically, `drop` picks a row from the band of
    feasible positions: 0 is as high as it will go, 1 as low. A heart wants a
    high value so its lobes and cleft stay visible above the symbol.
    """
    g = cells.shape[0]
    ii = np.pad(cells.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    area = need * need
    ok = [
        (x, y)
        for y in range(g - need + 1)
        for x in range(g - need + 1)
        if ii[y + need, x + need] - ii[y, x + need] - ii[y + need, x] + ii[y, x] == area
    ]
    if not ok:
        raise SystemExit(f"no room for a {need}-module square in a {g}-module silhouette")
    ys = sorted({y for _, y in ok})
    row = ys[min(len(ys) - 1, round(drop * (len(ys) - 1)))]
    xs = [x for x, y in ok if y == row]
    mid = (g - need) / 2.0
    return min(xs, key=lambda x: abs(x - mid)), row


# ------------------------------------------------------------------ glyph ---
_FONT_CACHE: dict[str, tuple[TTFont, float]] = {}


def glyph_path(char: str, font_file: str) -> tuple[str, float, float]:
    """Return (svg path data, advance width, upem) in font units, y-up."""
    if font_file not in _FONT_CACHE:
        f = TTFont(FONT_DIR / font_file)
        _FONT_CACHE[font_file] = (f, f["head"].unitsPerEm)
    font, upem = _FONT_CACHE[font_file]
    gs = font.getGlyphSet()
    name = font.getBestCmap()[ord(char)]
    pen = SVGPathPen(gs)
    gs[name].draw(pen)
    return pen.getCommands(), font["hmtx"][name][0], upem


def monogram_svg(cx: float, cy: float, cap_h: float, color: str) -> str:
    """'H&T' centred on (cx, cy) with cap height cap_h, in user units."""
    parts = [
        ("H", "InstrumentSerif-Regular.ttf", 1.00),
        ("&", "InstrumentSerif-Italic.ttf", 1.02),
        ("T", "InstrumentSerif-Regular.ttf", 1.00),
    ]
    tracking = 0.09  # em, extra space between glyphs
    laid, total = [], 0.0
    for ch, ff, rel in parts:
        d, adv, upem = glyph_path(ch, ff)
        scale = cap_h * rel / (upem * 0.70)  # ~0.70 upem cap height
        laid.append((d, adv * scale, scale))
        total += adv * scale
    total += tracking * cap_h * (len(parts) - 1)

    out, x = [], cx - total / 2.0
    for (d, adv, scale) in laid:
        # SVG y grows downward; font y grows upward -> flip.
        out.append(
            f'<g transform="translate({x:.3f},{cy + cap_h / 2:.3f}) '
            f'scale({scale:.6f},{-scale:.6f})"><path d="{d}" fill="{color}"/></g>'
        )
        x += adv + tracking * cap_h
    return "".join(out)


# --------------------------------------------------------------- dot field ---
def decorative_field(w: int, h: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """QR-looking noise: 50% random bits, then one clumping pass.

    Also returns a second uniform field, used to thin the dots out as they
    approach the symbol's quiet zone.
    """
    rnd = random.Random(seed)
    g = np.array([[rnd.random() < 0.5 for _ in range(w)] for _ in range(h)], np.uint8)
    keep = np.array([[rnd.random() for _ in range(w)] for _ in range(h)], np.float64)
    n = cv2.filter2D(g, cv2.CV_16S, np.ones((3, 3), np.int16), borderType=cv2.BORDER_CONSTANT)
    n = n - g  # neighbour sum, excluding self
    out = g.copy()
    out[n >= 7] = 1
    out[n <= 1] = 0
    return out, keep


# ------------------------------------------------------------------ design ---
@dataclass
class Design:
    name: str
    url: str
    mask_fn: callable
    grid: int  # silhouette width in modules
    outline: bool  # stroke the silhouette in gold
    logo_r: float  # cleared circle radius, in modules
    logo_rings: bool  # concentric gold rings around the monogram
    mono_cap: float  # monogram cap height, in modules
    inset: float  # modules the dot field is held back from the silhouette edge
    drop: float  # vertical placement of the symbol: 0 = highest, 1 = lowest
    fade: float  # modules over which decorative density ramps back up
    seed: int


DESIGNS = [
    Design("heart", TARGETS["heart"], heart_mask, 132, False, 6.0, False, 6.4, 0.45,
           0.72, 11.0, 20240614),
    Design("camera", TARGETS["camera"], camera_mask, 82, True, 8.2, True, 5.6, 1.15,
           0.5, 3.0, 771982),
]


def build(d: Design) -> tuple[str, dict]:
    qr = segno.make(d.url, error="h", boost_error=False)
    mat = [[int(v) for v in row] for row in qr.matrix]
    n = len(mat)

    # Silhouette mask at module resolution, sampled at cell centres. The dot
    # field uses an eroded copy so no dot spills over the silhouette edge.
    S = 8  # subsamples per module
    hi = d.grid * S
    mask_hi = fit_mask(d.mask_fn, hi)
    k = max(1, int(round(d.inset * S)) * 2 + 1)
    mask_in = cv2.erode(mask_hi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cells = np.zeros((d.grid, d.grid), np.uint8)
    for gy in range(d.grid):
        for gx in range(d.grid):
            cells[gy, gx] = mask_in[int((gy + 0.5) * S), int((gx + 0.5) * S)]

    need = n + 2 * QUIET
    px, py = place_square(cells, need, d.drop)
    qx, qy = px + QUIET, py + QUIET

    cxm, cym = qx + n / 2.0, qy + n / 2.0  # symbol centre, in modules

    # ---- decorative dots ---------------------------------------------------
    # Inside the silhouette, absent from the quiet zone, and thinning out over
    # `fade` modules on the approach so the clear zone reads as a halo rather
    # than a rectangular hole punched in the pattern.
    deco, keep = decorative_field(d.grid, d.grid, d.seed)
    x0, x1 = qx - QUIET, qx + n + QUIET - 1
    y0, y1 = qy - QUIET, qy + n + QUIET - 1
    body: list[str] = []
    for gy in range(d.grid):
        for gx in range(d.grid):
            if not cells[gy, gx] or not deco[gy, gx]:
                continue
            gap = math.hypot(max(x0 - gx, gx - x1, 0), max(y0 - gy, gy - y1, 0))
            if gap < 1 or keep[gy, gx] > min(1.0, (gap - 1) / d.fade):
                continue
            body.append(f'<circle cx="{gx + .5}" cy="{gy + .5}" r="{DOT_R}"/>')
    deco_dots = len(body)

    # ---- the symbol itself -------------------------------------------------
    # Finder and alignment patterns are drawn as concentric rings rather than
    # dots. Their nested 1:1:3:1:1 and 1:1:1:1:1 run lengths are what a scanner
    # locks onto; rendered as loose dots they stop being findable, and without
    # the alignment patterns a code cannot be read at an angle at all.
    finders = [(0, 0), (0, n - 7), (n - 7, 0)]  # (row, col) of 7x7 top-left
    in_finder = lambda r, c: any(fr <= r < fr + 7 and fc <= c < fc + 7 for fr, fc in finders)
    centres = ALIGNMENT.get(qr.version, [])
    aligns = [(r, c) for r in centres for c in centres if not in_finder(r, c)]
    in_align = lambda r, c: any(abs(r - ar) <= 2 and abs(c - ac) <= 2 for ar, ac in aligns)

    sym, covered = [], 0
    for r in range(n):
        for c in range(n):
            if in_finder(r, c) or in_align(r, c):
                continue
            gx, gy = qx + c, qy + r
            if math.hypot(gx + 0.5 - cxm, gy + 0.5 - cym) < d.logo_r:
                covered += 1
                continue
            if mat[r][c]:
                sym.append(f'<circle cx="{gx + .5}" cy="{gy + .5}" r="{DOT_R}"/>')
    body += sym

    # ---- finder eyes: gold annulus / cream gap / ink core -------------------
    eyes = []
    for fr, fc in finders:
        ex, ey = qx + fc + 3.5, qy + fr + 3.5
        eyes.append(
            f'<circle cx="{ex}" cy="{ey}" r="3.0" fill="none" '
            f'stroke="{GOLD_FINDER}" stroke-width="1.0"/>'
            f'<circle cx="{ex}" cy="{ey}" r="1.5" fill="{INK}"/>'
        )
    for ar, ac in aligns:
        ex, ey = qx + ac + 0.5, qy + ar + 0.5
        if math.hypot(ex - cxm, ey - cym) < d.logo_r + 3:
            covered += 25  # swallowed whole by the monogram disc
            continue
        eyes.append(
            f'<circle cx="{ex}" cy="{ey}" r="2.0" fill="none" '
            f'stroke="{ALIGN_COLOR}" stroke-width="1.0"/>'
            f'<circle cx="{ex}" cy="{ey}" r="0.5" fill="{ALIGN_COLOR}"/>'
        )

    # ---- centre monogram ---------------------------------------------------
    logo = [f'<circle cx="{cxm}" cy="{cym}" r="{d.logo_r}" fill="{CREAM}"/>']
    if d.logo_rings:
        logo.append(
            f'<circle cx="{cxm}" cy="{cym}" r="{d.logo_r - 0.35}" fill="none" '
            f'stroke="{GOLD}" stroke-width="0.30"/>'
            f'<circle cx="{cxm}" cy="{cym}" r="{d.logo_r - 1.25}" fill="none" '
            f'stroke="{GOLD}" stroke-width="0.22"/>'
        )
    logo.append(monogram_svg(cxm, cym, d.mono_cap, GOLD))

    # ---- silhouette outline (traced from the mask) -------------------------
    outline = ""
    if d.outline:
        cont, _ = cv2.findContours(mask_hi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cont, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.0012 * cv2.arcLength(c, True), True)
        pts = " ".join(f"{p[0][0] / S:.3f},{p[0][1] / S:.3f}" for p in c)
        outline = (
            f'<polygon points="{pts}" fill="none" stroke="{GOLD}" '
            f'stroke-width="0.42" stroke-linejoin="round"/>'
        )

    vb = d.grid + 2 * PAD
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vb} {vb}" '
        f'width="{vb * 24}" height="{vb * 24}">'
        f'<rect width="{vb}" height="{vb}" fill="{CREAM}"/>'
        f'<g transform="translate({PAD},{PAD})">'
        f"{outline}"
        f'<g fill="{INK}">{"".join(body)}</g>'
        f'{"".join(eyes)}'
        f'{"".join(logo)}'
        f"</g></svg>"
    )

    stats = {
        "version": qr.version,
        "modules": n,
        "grid": d.grid,
        "symbol_at": (qx, qy),
        "logo_covered_modules": covered,
        "logo_pct": 100.0 * covered / (n * n),
        "deco_dots": deco_dots,
    }
    return svg, stats


def main() -> None:
    import cairosvg

    for d in DESIGNS:
        svg, st = build(d)
        (OUT / f"{d.name}-qr.svg").write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / f"{d.name}-qr.png"),
                         output_width=2400, output_height=2400)
        # 150 mm square artboard: comfortably above the minimum print size, and
        # the PDF is vector so it rescales without loss.
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / f"{d.name}-qr.pdf"),
                         output_width=150 * 96 / 25.4, output_height=150 * 96 / 25.4)
        span = d.grid + 2 * PAD
        print(
            f"{d.name}: v{st['version']} {st['modules']}x{st['modules']} in a "
            f"{st['grid']}-module silhouette | logo covers {st['logo_covered_modules']} "
            f"modules ({st['logo_pct']:.1f}%) | {st['deco_dots']} decorative dots | "
            f"min print {span * 0.6:.0f} mm wide"
        )


if __name__ == "__main__":
    main()
