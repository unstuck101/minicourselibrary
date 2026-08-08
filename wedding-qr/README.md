# Wedding QR art — Hannah & Travis

Two shaped QR codes in the style of the reference art, built so they actually scan.

| file | scans to |
| --- | --- |
| `heart-qr.svg` / `.pdf` / `.png` | https://www.theknot.com/us/hannah-garbutt-and-travis-ala/registry |
| `camera-qr.svg` / `.pdf` / `.png` | https://drive.google.com/drive/folders/1AM0JzJBxKYEtNEiqoe8fBeYp5nqc0ROl |

Use the **SVG or PDF** for anything printed — they are vector and rescale without
loss. The PNGs (2400 px) are previews for screens and proofs.

## Minimum print size

| design | minimum width | comfortable |
| --- | --- | --- |
| heart | 82 mm / 3.2 in | 100 mm / 4 in |
| camera | 52 mm / 2.0 in | 65 mm / 2.6 in |

The heart needs more room because a heart holds a smaller square than a camera
body does, so the symbol is a smaller share of the artwork. Below these sizes the
modules drop under ~0.6 mm and scanning gets unreliable in poor light or at an
angle. Keep the cream background — do not place the art on a dark or busy field,
and do not crop into the pale margin, which is the symbol's quiet zone.

## How the shape and the code coexist

Clipping a QR symbol into a heart or camera outline — which is what the reference
images do — deletes the corner finder patterns and far more codewords than error
correction can rebuild. Those images are decoration; they do not scan.

Here the silhouette is a **decorative dot field**, and a complete, unclipped
version-8 symbol at error-correction level H sits inside it, wrapped in an intact
five-module quiet zone. Decorative density feathers out on the approach to that
quiet zone so it reads as a halo rather than a rectangular hole. Visually it is
one continuous field of dots; functionally the scanner only ever sees a clean,
whole QR code.

Three details are load-bearing:

- **Finder eyes** are concentric circles, not dots. A scanner locks onto their
  1:1:3:1:1 run lengths; rendered as loose dots they stop being findable. The
  gold ring is a deliberately deep bronze so it binarizes as dark — a lighter
  gold reads as white to a scanner and destroys the pattern.
- **Alignment patterns** are concentric rings for the same reason. Rendered as
  dots, the code still scans flat-on but cannot be read at *any* angle, because
  perspective correction has nothing to lock onto. This is the single most
  fragile element, which is why they stay full ink; `ALIGN_COLOR` in the
  generator flips them to bronze if you prefer the softer look (it passes the
  scan battery too, with less margin).
- **The centre monogram** covers 4.5% (heart) and 8.9% (camera) of the symbol,
  well inside level H's ~30% recovery budget.

## Regenerating

```
pip install segno pillow opencv-python-headless numpy cairosvg fonttools zxing-cpp
python3 generate_qr.py    # writes the SVG, PNG and PDF for both designs
python3 verify_scan.py    # decode battery; exits non-zero on any failure
```

`verify_scan.py` renders each design at a range of resolutions and decodes it
under rotation, out-of-focus blur, low contrast, sensor noise, uneven lighting
and perspective skew — 8 variants per size. Everything passes at or above 5
pixels per module.

It decodes with zxing-cpp, the ZXing engine behind most phone scanners. OpenCV's
`QRCodeDetector` is deliberately not used: it fails on dot-rendered modules even
for a plain unstyled dotted code, so it reports nothing meaningful about this
artwork.

Type is [Instrument Serif](https://fonts.google.com/specimen/Instrument+Serif)
(SIL Open Font License), converted to outlines so the files carry no font
dependency.
