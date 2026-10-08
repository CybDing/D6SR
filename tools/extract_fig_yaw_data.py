#!/usr/bin/env python3
"""Recover the paper's Fig. 9 (hardware yaw) data from the rendered PDF.

The Raspberry Pi log behind Fig. 9 was never persisted — `_plot_yaw_cascaded`
in raspberry_pi/main.py only ever wrote a PNG, and the raw arrays died with
the process. The submitted figure is a matplotlib *vector* PDF, so the two
traces survive in it as explicit polyline coordinates; this script reads them
back and maps device units to data units using the axis tick marks.

Calibration is exact, not fitted: matplotlib emits each tick as a line
segment immediately followed by its own text label, so the anchors are read
straight out of the content stream and the mapping is a two-point linear
solve per axis (verified against every remaining tick).

Usage:
    python tools/extract_fig_yaw_data.py [--pdf docs/media/fig_yaw.pdf]
                                         [--out data/fig9_yaw_hardware.csv]
"""

import argparse
import csv
import re
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# stroke colours matplotlib wrote for the two series. Matched whitespace-
# tolerantly: the writer wraps long content lines, and the red operator in
# fig_yaw.pdf really is split across a newline mid-colour.
RED = r'0\.9019607843\s+0\.2235294118\s+0\.2745098039\s+RG'    # Measured
BLUE = r'0\.1294117647\s+0\.5882352941\s+0\.9529411765\s+RG'   # Reference


def page_content(pdf_bytes):
    """Return the decompressed content stream that holds the page drawing."""
    for m in re.finditer(rb'stream\r?\n', pdf_bytes):
        s = m.end()
        e = pdf_bytes.find(b'endstream', s)
        try:
            t = zlib.decompress(pdf_bytes[s:e])
        except zlib.error:
            continue
        if b' m\n' in t and b' l\n' in t and b'Tf' in t:
            return t.decode('latin-1')
    raise SystemExit('no page content stream found in PDF')


def read_ticks(text):
    """Axis tick anchors: [(device_coord, data_value), ...] for x and y.

    A tick is a 2-point line segment followed by the label's text-drawing
    block; the label glyphs are NUL-separated in the PDF string.
    """
    pat = re.compile(
        r'([\d.]+) ([\d.]+) m\n([\d.]+) ([\d.]+) l\n\nB\n1 w\nq\n'
        r'1 0 -0 1 ([\d.-]+) ([\d.-]+) cm\nBT\n/F\d+ \d+ Tf\n0 0 Td\n'
        r'\[ \(([^)]*)\) \] TJ')
    xt, yt = [], []
    for m in pat.finditer(text):
        x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
        value = float(m.group(7).replace('\x00', '').replace(' ', ''))
        if x0 == x1:          # vertical stub -> x-axis tick
            xt.append((x0, value))
        elif y0 == y1:        # horizontal stub -> y-axis tick
            yt.append((y0, value))
    return xt, yt


def linear_from_ticks(ticks, axis):
    """Solve device = a*data + b from the two extreme ticks, then assert the
    rest of the ticks land on it. Any real deviation means the axis is not
    linear (log scale, broken axis) and the extraction would be wrong."""
    ticks = sorted(ticks, key=lambda p: p[1])
    (d0, v0), (d1, v1) = ticks[0], ticks[-1]
    a = (d1 - d0) / (v1 - v0)
    b = d0 - a * v0
    worst = max(abs(a * v + b - d) for d, v in ticks)
    if worst > 1e-3:
        raise SystemExit(f'{axis}: ticks not collinear (max resid {worst})')
    print(f'  {axis}: device = {a:.9f}*data + {b:.6f}   '
          f'({len(ticks)} ticks, max resid {worst:.2e} pt)')
    return a, b


def read_polyline(text, colour):
    """Vertices of the longest polyline stroked in `colour`, device units.

    Longest, not first: each series is stroked twice — once as the data and
    once as the two-point sample line inside the legend box.
    """
    best = []
    for m in re.finditer(colour, text):
        seg = text[m.start():text.find('\nS', m.start())]
        pts = [(float(a), float(b)) for a, b in
               re.findall(r'([\d.-]+) ([\d.-]+) [ml]\n', seg)]
        if len(pts) > len(best):
            best = pts
    if not best:
        raise SystemExit(f'colour {colour} not in PDF')
    # matplotlib emits each vertex twice; collapse consecutive duplicates
    return [p for j, p in enumerate(best) if j == 0 or p != best[j - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', default=str(REPO / 'docs/media/fig_yaw.pdf'))
    ap.add_argument('--out', default=str(REPO / 'data/fig9_yaw_hardware.csv'))
    args = ap.parse_args()

    text = page_content(Path(args.pdf).read_bytes())
    print(f'source: {args.pdf}')
    xt, yt = read_ticks(text)
    ax, bx = linear_from_ticks(xt, 'x (time, s)')
    ay, by = linear_from_ticks(yt, 'y (yaw, deg)')

    to_t = lambda d: (d - bx) / ax
    to_y = lambda d: (d - by) / ay

    series = {}
    for name, colour in (('measured', RED), ('reference', BLUE)):
        pts = read_polyline(text, colour)
        series[name] = [(to_t(x), to_y(y)) for x, y in pts]
        print(f'  {name}: {len(pts)} vertices, '
              f't {series[name][0][0]:.2f}..{series[name][-1][0]:.2f} s, '
              f'yaw {min(p[1] for p in series[name]):.2f}..'
              f'{max(p[1] for p in series[name]):.2f} deg')

    # Two independent check digits on the calibration. The reference is a step
    # command, so its levels must come back as the round numbers the operator
    # typed; the measured trace is IMU yaw logged at 0.1 deg, so every sample
    # must land on that grid. Both hold to ~1e-6, which is what makes this a
    # recovery of the original values rather than an estimate off a plot.
    levels = sorted({round(y, 4) + 0.0 for _, y in series['reference']})
    print(f'  reference levels: {levels}')
    for lv in levels:
        if abs(lv - round(lv)) > 0.02:
            raise SystemExit(f'reference level {lv} is not an integer — '
                             f'calibration suspect')
    worst = max(abs(y * 10 - round(y * 10)) for _, y in series['measured'])
    print(f'  measured vs 0.1-deg grid: max deviation {worst:.2e}')
    if worst > 0.02:
        raise SystemExit('measured trace is off the 0.1-deg grid — '
                         'calibration suspect')

    # Snap both series back onto the grids just verified. The PDF stores
    # device coords to 6 dp, so the inverse map lands ~1e-5 off the true
    # value; snapping to the proven grid removes that render round-trip
    # error rather than carrying it into the CSV as fake precision.
    snap_meas = [(t, round(y * 10) / 10) for t, y in series['measured']]
    snap_ref = [(t, float(round(y))) for t, y in series['reference']]

    # Resample the step reference onto the measured trace's timestamps so the
    # CSV is one tidy time series (the two polylines have different vertices).
    def ref_at(t):
        lv = snap_ref[0][1]
        for rt, ry in snap_ref:
            if rt <= t + 1e-9:
                lv = ry
            else:
                break
        return lv

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'measured_yaw_deg', 'reference_yaw_deg'])
        for t, y in snap_meas:
            w.writerow([f'{t:.4f}', f'{y:.1f}', f'{ref_at(t):.1f}'])
    print(f'\nwrote {out}  ({len(series["measured"])} rows)')


if __name__ == '__main__':
    main()
