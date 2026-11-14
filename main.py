# SPDX-License-Identifier: MIT
"""Empirical-quantile forecaster for Polymarket price series.

For each market, the h-step price changes over the observed history are
collected at several anchor horizons; their empirical quantiles are
interpolated in sqrt(h) and blended 50/50 with the house persistence
baseline (Gaussian random walk, sigma = 0.0025). Reads every *.in case
from MAKEFUN_INPUT_DIR, writes matching *.out files to
MAKEFUN_OUTPUT_DIR. Stdlib only, fully deterministic.
"""
import json
import math
import os

Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
SIG0 = 0.0025
BLEND_BASE = 0.5
ANCHORS = [15, 30, 60, 120, 240]


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def r6(v):
    return round(clamp(v, 0.0, 1.0), 6)


def quantile(sorted_a, q):
    n = len(sorted_a)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_a[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_a[lo] * (1.0 - frac) + sorted_a[hi] * frac


def cummax(row):
    out = []
    m = -1e18
    for v in row:
        if v > m:
            m = v
        out.append(m)
    return out


def baseline_rows(x, horizon):
    rows = []
    for h in range(1, horizon + 1):
        w = SIG0 * math.sqrt(h)
        rows.append(cummax([r6(x + z * w) for z in Z]))
    return rows


def adaptive_rows(p, horizon):
    n = len(p)
    x = p[-1]
    if n < 130:
        return baseline_rows(x, horizon)
    table = {}
    for a in ANCHORS:
        ch = sorted(p[i] - p[i - a] for i in range(a, n, 3))
        if len(ch) >= 20:
            table[a] = [quantile(ch, t) for t in TAUS]
        else:
            table[a] = [0.0] * 5
    base = baseline_rows(x, horizon)
    sq = [math.sqrt(a) for a in ANCHORS]
    rows = []
    for h in range(1, horizon + 1):
        sh = math.sqrt(h)
        if sh <= sq[0]:
            vals = table[ANCHORS[0]]
        elif sh >= sq[-1]:
            vals = table[ANCHORS[-1]]
        else:
            vals = table[ANCHORS[-1]]
            for i in range(len(ANCHORS) - 1):
                if sq[i] <= sh <= sq[i + 1]:
                    t = (sh - sq[i]) / (sq[i + 1] - sq[i])
                    vals = [table[ANCHORS[i]][j] * (1.0 - t) + table[ANCHORS[i + 1]][j] * t for j in range(5)]
                    break
        brow = base[h - 1]
        row = [r6((1.0 - BLEND_BASE) * (x + vals[j]) + BLEND_BASE * brow[j]) for j in range(5)]
        rows.append(cummax(row))
    return rows


def process_case(case):
    try:
        horizon = int(case.get("horizon_minutes") or 240)
    except Exception:
        horizon = 240
    forecasts = []
    for m in case.get("markets") or []:
        mid = m.get("id")
        prices = []
        try:
            prices = [float(v) for v in (m.get("prices") or [])]
        except Exception:
            prices = []
        try:
            rows = adaptive_rows(prices, horizon)
        except Exception:
            try:
                rows = baseline_rows(prices[-1] if prices else 0.5, horizon)
            except Exception:
                rows = baseline_rows(0.5, horizon)
        forecasts.append({"id": mid, "q": rows})
    return {"forecasts": forecasts}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    try:
        names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    except Exception:
        names = []
    for name in names:
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        try:
            with open(os.path.join(in_dir, name)) as fh:
                case = json.load(fh)
            result = process_case(case)
        except Exception:
            result = {"forecasts": []}
        try:
            with open(out_path, "w") as fh:
                fh.write(json.dumps(result, separators=(",", ":")) + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
