"""CM30 Plate Viewer: confluency heatmap and scratch assay closure for CM30 exports."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLATE_LAYOUTS = {6: (2, 3), 12: (3, 4), 24: (4, 6), 48: (6, 8), 96: (8, 12), 384: (16, 24)}

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
INK = "#111827"
MUTED = "#6B7280"
FAINT = "#9CA3AF"
TRAY = "#F5F6F8"
TRAY_EDGE = "#E3E6EB"
GAP_FILL = "#FFFFFF"
NODATA_FILL = "#E5E7EB"
EMPTY_RING = "#C9CED6"
RING = "rgba(17,24,39,0.25)"

WELL_R = 0.42       # well radius in plate units (one well = 1 x 1)
GAP_HALF = 0.5      # half-width of the schematic scratch, as a fraction of the well radius
HEAT_BINS = 48      # color steps used to draw the heatmap
FULL_DRAW = 0.975   # closure above this is drawn as fully closed (no hairline gap)
SPEEDS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}

WELL_LINE = re.compile(r"^Well([A-Za-z]+\d+|\d+)$", re.I)


# ---------------------------------------------------------------- parsing
def _letters_to_index(letters):
    i = 0
    for ch in letters.upper():
        i = i * 26 + (ord(ch) - 64)
    return i - 1


def _resolve_layout(vessel_name, ids):
    """Return (rows, cols) from the vessel name, or infer from the well ids."""
    m = re.search(r"(\d+)\s*well", vessel_name or "", re.I)
    if m and int(m.group(1)) in PLATE_LAYOUTS:
        return PLATE_LAYOUTS[int(m.group(1))]

    lettered = [re.fullmatch(r"([A-Z]+)(\d+)", w) for w in ids]
    if all(lettered):
        need_r = max(_letters_to_index(m.group(1)) for m in lettered) + 1
        need_c = max(int(m.group(2)) for m in lettered)
        for _, (r, c) in sorted(PLATE_LAYOUTS.items()):
            if r >= need_r and c >= need_c:
                return r, c
        return need_r, need_c

    need_n = max(int(w) for w in ids if w.isdigit())
    for n, (r, c) in sorted(PLATE_LAYOUTS.items()):
        if n >= need_n:
            return r, c
    return 1, need_n


@st.cache_resource(show_spinner=False, max_entries=4)
def parse_cm30(raw: bytes):
    """Parse a CM30 analysis export. Handles Well1 and WellA2 naming and both result sections."""
    text = raw.decode("utf-8-sig", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    project, vessel = "", ""
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("<project>") and i + 2 < len(lines):
            project = lines[i + 2].split(",")[0]
        if low.startswith("<vessel type>") and i + 2 < len(lines):
            parts = lines[i + 2].split(",")
            vessel = parts[5] if len(parts) > 5 else ""

    recs, cur, t_idx, c_idx = [], None, 1, 2
    for ln in lines:
        if not ln:
            continue
        if WELL_LINE.match(ln):
            cur = ln[4:].upper()
            continue
        if ln.startswith("Passage#"):
            cols = [c.strip().lower() for c in ln.split(",")]
            t_idx = next((k for k, c in enumerate(cols) if c == "time"), 1)
            c_idx = next((k for k, c in enumerate(cols) if "confluency" in c), 2)
            continue
        if cur and ln[0].isdigit():
            p = ln.split(",")
            if len(p) > max(t_idx, c_idx):
                try:
                    recs.append((cur, p[t_idx], float(p[c_idx])))
                except ValueError:
                    pass

    df = pd.DataFrame(recs, columns=["WellId", "Time", "Confluency"])
    if df.empty:
        return None, None
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time", "Confluency"])

    ids = sorted(df["WellId"].unique())
    rows, cols = _resolve_layout(vessel, ids)

    def locate(wid):
        m = re.fullmatch(r"([A-Z]+)(\d+)", wid)
        if m:
            return wid, m.group(1), _letters_to_index(m.group(1)), int(m.group(2)) - 1
        n = int(wid)
        i, j = divmod(n - 1, cols)
        return str(n), chr(65 + i), i, j

    loc = {w: locate(w) for w in ids}
    df["Well"] = df["WellId"].map(lambda w: loc[w][0])
    df["Row"] = df["WellId"].map(lambda w: loc[w][1])
    df["RowIdx"] = df["WellId"].map(lambda w: loc[w][2])
    df["ColIdx"] = df["WellId"].map(lambda w: loc[w][3])
    df["Column"] = df["ColIdx"] + 1

    df = df.sort_values(["RowIdx", "ColIdx", "Time"]).reset_index(drop=True)
    df["T_index"] = df.groupby("Well", sort=False).cumcount() + 1
    df["ReferenceTime"] = df.groupby("T_index")["Time"].transform("min")
    df["Elapsed_h"] = (
        (df["ReferenceTime"] - df["ReferenceTime"].min()).dt.total_seconds() / 3600
    ).round(2)

    order = (
        df[["Well", "RowIdx", "ColIdx"]].drop_duplicates().sort_values(["RowIdx", "ColIdx"])["Well"].tolist()
    )
    meta = {
        "project": project,
        "vessel": vessel,
        "rows": rows,
        "cols": cols,
        "numbered": not any(re.fullmatch(r"[A-Z]+\d+", w) for w in ids),
        "well_order": order,
    }
    return df, meta


# ---------------------------------------------------------------- scratch maths and exports
def add_closure(df, full_closed_pct, cap):
    """Closure % = (C_t - C_0) / (C_full - C_0) * 100, with C_0 the first timepoint of each well."""
    out = df.copy()
    c0 = out.groupby("Well")["Confluency"].transform("first")
    denom = (full_closed_pct - c0).where(lambda s: s > 0)
    closure = (out["Confluency"] - c0) / denom * 100
    if cap:
        closure = closure.clip(0, 100)
    out["Confluency_T0"] = c0
    out["Full_closure_ref"] = full_closed_pct
    out["Closure_pct"] = closure.round(2)
    return out


def export_long(cdf):
    out = cdf.sort_values(["RowIdx", "ColIdx", "T_index"]).copy()
    out["Time"] = out["ReferenceTime"].dt.strftime("%Y-%m-%d %H:%M")
    out = out.rename(columns={"T_index": "Timepoint"})
    cols = ["Well", "Row", "Column", "Timepoint", "Time", "Elapsed_h", "Confluency",
            "Confluency_T0", "Full_closure_ref", "Closure_pct"]
    return out[cols]


def export_wide(cdf, well_order):
    tmp = cdf.copy()
    tmp["Time"] = tmp["ReferenceTime"].dt.strftime("%Y-%m-%d %H:%M")
    wide = tmp.pivot_table(
        index=["T_index", "Time", "Elapsed_h"], columns="Well", values="Closure_pct", aggfunc="first"
    )
    wide = wide.reindex(columns=well_order).reset_index().rename(columns={"T_index": "Timepoint"})
    wide.columns.name = None
    return wide


@st.cache_resource(show_spinner=False, max_entries=8)
def closure_bundle(raw: bytes, full_closed_pct: float, cap: bool):
    """Closure table plus both CSV exports, cached so slider moves never recompute them."""
    df, meta = parse_cm30(raw)
    cdf = add_closure(df, full_closed_pct, cap)
    long_df = export_long(cdf)
    wide_df = export_wide(cdf, meta["well_order"])
    n_bad = int((cdf.drop_duplicates("Well")["Confluency_T0"] >= full_closed_pct).sum())
    return {
        "cdf": cdf,
        "long_csv": long_df.to_csv(index=False).encode(),
        "wide_csv": wide_df.to_csv(index=False).encode(),
        "preview": long_df.head(200),
        "n_bad": n_bad,
    }


@st.cache_resource(show_spinner=False, max_entries=4)
def confluency_csv(raw: bytes):
    df, _ = parse_cm30(raw)
    tidy = df.sort_values(["RowIdx", "ColIdx", "T_index"]).copy()
    tidy["Time"] = tidy["ReferenceTime"].dt.strftime("%Y-%m-%d %H:%M")
    tidy = tidy.rename(columns={"T_index": "Timepoint"})[
        ["Well", "Row", "Column", "Timepoint", "Time", "Elapsed_h", "Confluency"]]
    return tidy.to_csv(index=False).encode()


# ---------------------------------------------------------------- colour helpers
def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lerp_color(c0, c1, t):
    a, b = _rgb(c0), _rgb(c1)
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


def _css(rgb):
    return f"rgb({int(round(rgb[0]))},{int(round(rgb[1]))},{int(round(rgb[2]))})"


def _text_on(rgb):
    lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return INK if lum > 0.55 else "#FFFFFF"


# ---------------------------------------------------------------- geometry helpers
# Every well is drawn as a polygon inside a handful of shared traces (NaN separates polygons),
# which keeps the browser's work small even for a 96-well plate.
def _flat(X, Y):
    pad = np.full((X.shape[0], 1), np.nan)
    # 3 decimals is far below a pixel and cuts the data sent to the browser by about two thirds
    return np.round(np.hstack([X, pad]).ravel(), 3), np.round(np.hstack([Y, pad]).ravel(), 3)


def _circle_xy(cx, cy, r, n=40):
    th = np.linspace(0, 2 * np.pi, n + 1)
    return cx[:, None] + r * np.cos(th), cy[:, None] + r * np.sin(th)


def _strip_xy(cx, cy, r, a, b, k=12):
    """Polygons for the part of each circle between y offsets a and b from the well centre."""
    pa = np.arcsin(np.clip(a / r, -1, 1))
    pb = np.arcsin(np.clip(b / r, -1, 1))
    s = np.linspace(0, 1, k)
    ph_r = pa[:, None] + (pb - pa)[:, None] * s
    ph_l = pb[:, None] + (pa - pb)[:, None] * s
    xr, yr = cx[:, None] + r * np.cos(ph_r), cy[:, None] + r * np.sin(ph_r)
    xl, yl = cx[:, None] - r * np.cos(ph_l), cy[:, None] + r * np.sin(ph_l)
    return np.hstack([xr, xl, xr[:, :1]]), np.hstack([yr, yl, yr[:, :1]])


def _rounded_rect_xy(x0, y0, x1, y1, rad, n=10):
    pts_x, pts_y = [], []
    for ccx, ccy, a0, a1 in [(x1 - rad, y0 + rad, -90, 0), (x1 - rad, y1 - rad, 0, 90),
                             (x0 + rad, y1 - rad, 90, 180), (x0 + rad, y0 + rad, 180, 270)]:
        th = np.radians(np.linspace(a0, a1, n))
        pts_x.append(ccx + rad * np.cos(th))
        pts_y.append(ccy + rad * np.sin(th))
    x, y = np.concatenate(pts_x), np.concatenate(pts_y)
    return np.append(x, x[0]), np.append(y, y[0])


def _fill_trace(X, Y, color):
    x, y = _flat(X, Y)
    return go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=color, line=dict(width=0),
                      hoverinfo="skip", showlegend=False)


def _line_trace(X, Y, color, width=1, dash=None):
    x, y = _flat(X, Y)
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash),
                      hoverinfo="skip", showlegend=False)


def _empty_positions(rows, cols, cx, cy):
    taken = {(int(round(y - 0.5)), int(round(x - 0.5))) for x, y in zip(cx, cy)}
    pos = [(i, j) for i in range(rows) for j in range(cols) if (i, j) not in taken]
    if not pos:
        return np.array([]), np.array([])
    arr = np.array(pos, dtype=float)
    return arr[:, 1] + 0.5, arr[:, 0] + 0.5


def _canvas(rows, cols, numbered, x_pad):
    cell = min(130.0, 900.0 / (cols + x_pad))
    left = 0.1 if numbered else 0.45
    fig = go.Figure()
    fig.update_layout(
        height=int(rows * cell + 120),
        margin=dict(l=8 if numbered else 30, r=8, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=13),
        showlegend=False,
        dragmode=False,
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=13)),
        xaxis=dict(
            range=[-left, cols + x_pad], side="top", showgrid=False, zeroline=False, showline=False,
            ticks="", fixedrange=True, constrain="domain",
            tickvals=[] if numbered else [j + 0.5 for j in range(cols)],
            ticktext=[] if numbered else [str(j + 1) for j in range(cols)],
        ),
        yaxis=dict(
            range=[rows + 0.1, -0.1], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False,
            showline=False, ticks="", fixedrange=True, constrain="domain",
            tickvals=[] if numbered else [i + 0.5 for i in range(rows)],
            ticktext=[] if numbered else [chr(65 + i) for i in range(rows)],
        ),
    )
    tx, ty = _rounded_rect_xy(0, 0, cols, rows, 0.28)
    fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", fill="toself", fillcolor=TRAY,
                             line=dict(color=TRAY_EDGE, width=1), hoverinfo="skip", showlegend=False))
    return fig, cell


def _well_numbers(fig, meta):
    if not meta["numbered"]:
        return
    rows, cols = meta["rows"], meta["cols"]
    n = np.arange(rows * cols)
    fig.add_trace(go.Scatter(
        x=(n % cols) + 0.5 - WELL_R * 0.78, y=(n // cols) + 0.5 - WELL_R * 0.78, mode="text",
        text=[str(k + 1) for k in n], textfont=dict(size=11, color=MUTED, family=FONT),
        hoverinfo="skip", showlegend=False))


def _hover_layer(fig, cx, cy, size, hovertext, text=None, text_colors=None, fs=13):
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers+text" if text is not None else "markers",
        marker=dict(size=size, color="rgba(0,0,0,0)"),
        text=text, textfont=dict(size=fs, family=FONT, color=text_colors) if text is not None else None,
        textposition="middle center", hovertext=hovertext, hoverinfo="text", showlegend=False))


# ---------------------------------------------------------------- figures
def heatmap_figure(sl, meta, vmin, vmax, c0, c1, show_values=True):
    rows, cols = meta["rows"], meta["cols"]
    fig, cell = _canvas(rows, cols, meta["numbered"], 1.3)
    r = WELL_R
    s = sl.dropna(subset=["Confluency"])
    cx = s["ColIdx"].to_numpy(float) + 0.5
    cy = s["RowIdx"].to_numpy(float) + 0.5
    v = s["Confluency"].to_numpy(float)

    ex, ey = _empty_positions(rows, cols, cx, cy)
    if len(ex):
        fig.add_trace(_line_trace(*_circle_xy(ex, ey, r), EMPTY_RING, 1, "dot"))

    t = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    bins = np.rint(t * (HEAT_BINS - 1)).astype(int)
    colors = {k: lerp_color(c0, c1, k / (HEAT_BINS - 1)) for k in np.unique(bins)}
    for k in colors:
        sel = bins == k
        fig.add_trace(_fill_trace(*_circle_xy(cx[sel], cy[sel], r), _css(colors[k])))
    if len(cx):
        fig.add_trace(_line_trace(*_circle_xy(cx, cy, r), RING, 1))

    _well_numbers(fig, meta)
    fmt = ".1f" if cols <= 6 else ".0f"
    fs = int(max(9, min(16, cell * 0.17)))
    _hover_layer(
        fig, cx, cy, min(cell * 0.8, 100),
        [f"<b>Well {w}</b><br>Confluency {val:.1f}%" for w, val in zip(s["Well"], v)],
        [f"{val:{fmt}}%" for val in v] if show_values else None,
        [_text_on(colors[k]) for k in bins], fs)

    # scale bar, drawn in plate coordinates so it hugs the plate at any window width
    xb, y_top, y_bot = cols + 0.3, 0.4, rows - 0.4
    zvals = np.linspace(vmax, vmin, 60)
    fig.add_trace(go.Heatmap(
        x=[xb - 0.04, xb + 0.04], y=np.linspace(y_top, y_bot, 60), z=np.column_stack([zvals, zvals]),
        colorscale=[[0, c0], [1, c1]], zmin=vmin, zmax=vmax, showscale=False, hoverinfo="skip",
        xgap=0, ygap=0))
    ticks = np.linspace(vmin, vmax, 5)
    fig.add_trace(go.Scatter(
        x=[xb + 0.14] * 5, y=y_top + (vmax - ticks) / (vmax - vmin) * (y_bot - y_top), mode="text",
        text=[f"{x:g}" for x in ticks], textposition="middle right",
        textfont=dict(size=12, color=MUTED, family=FONT), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[xb - 0.08], y=[y_top - 0.22], mode="text", text=["% confluency"], textposition="middle right",
        textfont=dict(size=12, color=MUTED, family=FONT), hoverinfo="skip", showlegend=False))
    return fig


def scratch_figure(sl, meta, c_start, c_mig, show_values=True):
    rows, cols = meta["rows"], meta["cols"]
    fig, cell = _canvas(rows, cols, meta["numbered"], 1.9)
    r = WELL_R
    cx_all = sl["ColIdx"].to_numpy(float) + 0.5
    cy_all = sl["RowIdx"].to_numpy(float) + 0.5
    cl_all = sl["Closure_pct"].to_numpy(float) / 100
    ok = ~np.isnan(cl_all)

    ex, ey = _empty_positions(rows, cols, cx_all, cy_all)
    if len(ex):
        fig.add_trace(_line_trace(*_circle_xy(ex, ey, r), EMPTY_RING, 1, "dot"))

    # layer 1: open gap (white disc under everything), and wells with no closure value
    fig.add_trace(_fill_trace(*_circle_xy(cx_all, cy_all, r), GAP_FILL))
    if (~ok).any():
        fig.add_trace(_fill_trace(*_circle_xy(cx_all[~ok], cy_all[~ok], r), NODATA_FILL))

    cx, cy, c = cx_all[ok], cy_all[ok], np.clip(cl_all[ok], 0, 1)
    half0 = GAP_HALF * r
    gh = half0 * (1 - c)                       # half-width of the gap that is still open
    eps = 0.02 * r                             # overlap hides the seam between the two colors

    # layer 2: migrated cells grow from the original cell fronts toward the middle of the gap
    m = c > 0.002
    full = c > FULL_DRAW
    if m.any():
        top_b = np.where(full, half0 + eps, -gh)[m]
        Xt, Yt = _strip_xy(cx[m], cy[m], r, np.full(m.sum(), -half0 - eps), top_b)
        bm = m & ~full
        Xb, Yb = _strip_xy(cx[bm], cy[bm], r, gh[bm], np.full(bm.sum(), half0 + eps))
        mig = [np.vstack([Xt, Xb]), np.vstack([Yt, Yb])]
        fig.add_trace(_fill_trace(mig[0], mig[1], c_mig))

    # layer 3: starting cells (the untouched monolayer) above and below the original scratch
    if len(cx):
        n = len(cx)
        Xs1, Ys1 = _strip_xy(cx, cy, r, np.full(n, -r), np.full(n, -half0))
        Xs2, Ys2 = _strip_xy(cx, cy, r, np.full(n, half0), np.full(n, r))
        fig.add_trace(_fill_trace(np.vstack([Xs1, Xs2]), np.vstack([Ys1, Ys2]), c_start))
        fig.add_trace(_line_trace(*_circle_xy(cx_all, cy_all, r), RING, 1))

    _well_numbers(fig, meta)

    # label pills and hover layer
    fmt = ".1f" if cols <= 6 else ".0f"
    fs = int(max(9, min(15, cell * 0.16)))
    chars = 6 if cols <= 6 else 4
    pw, ph = (chars * fs * 0.6 + 10) / cell, fs * 1.5 / cell
    if show_values and len(cx):
        tx, ty = _rounded_rect_xy(-pw / 2, -ph / 2, pw / 2, ph / 2, ph / 2, 6)
        fig.add_trace(_fill_trace(cx[:, None] + tx, cy[:, None] + ty, "rgba(255,255,255,0.9)"))

    labels, hover = [], []
    for w, cl, conf, c0v, good in zip(sl["Well"], cl_all, sl["Confluency"], sl["Confluency_T0"], ok):
        if good:
            labels.append(f"{cl * 100:{fmt}}%")
            hover.append(f"<b>Well {w}</b><br>Closure {cl * 100:.1f}%<br>Confluency {conf:.1f}% (start {c0v:.1f}%)")
        else:
            labels.append("")
            hover.append(f"<b>Well {w}</b><br>Closure unavailable<br>Starts at {c0v:.1f}% confluency")
    _hover_layer(fig, cx_all, cy_all, min(cell * 0.8, 100), hover, labels if show_values else None, INK, fs)

    # key, drawn next to the plate
    kx, ky = cols + 0.3, rows / 2 + np.array([-0.4, 0.0, 0.4])
    fig.add_trace(go.Scatter(
        x=[kx] * 3, y=ky, mode="markers+text", text=["Starting cells", "Migrated", "Open gap"],
        textposition="middle right", textfont=dict(size=12, color=MUTED, family=FONT),
        marker=dict(symbol="square", size=14, color=[c_start, c_mig, GAP_FILL],
                    line=dict(width=1, color="rgba(17,24,39,0.3)")),
        hoverinfo="skip", showlegend=False))
    return fig


PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                               "autoScale2d", "resetScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "plate"},
}


# ---------------------------------------------------------------- time chart (also the scrubber)
def _rgba(hex_color, alpha):
    r, g, b = _rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def time_chart(hours, values, t, color):
    """Plate-average curve with a marker at the current timepoint. Clicking anywhere selects a timepoint."""
    hours = np.asarray(hours, float)
    values = np.asarray(values, float)
    n = len(hours)
    now = hours[t - 1]
    dx = float(np.median(np.diff(hours))) if n > 1 else 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=values, mode="lines", line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=_rgba(color, 0.16), showlegend=False,
        hovertemplate="%{x:.1f} h, %{y:.1f}%<extra></extra>"))
    # invisible full-height bars make the whole chart area clickable
    fig.add_trace(go.Bar(
        x=hours, y=np.full(n, 100.0), width=dx, customdata=np.arange(1, n + 1),
        marker=dict(color="rgba(0,0,0,0)"), selected=dict(marker=dict(opacity=0)),
        unselected=dict(marker=dict(opacity=0)), hoverinfo="none", showlegend=False))
    fig.add_shape(type="line", x0=now, x1=now, y0=0, y1=100, line=dict(color=INK, width=1.2, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[now], y=[0], mode="markers", cliponaxis=False, hoverinfo="skip", showlegend=False,
        marker=dict(size=13, color=INK, line=dict(color="white", width=2))))
    fig.update_layout(
        height=150, margin=dict(l=44, r=8, t=8, b=40), bargap=0, hovermode="x", showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=12),
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=12)),
        xaxis=dict(range=[hours[0], hours[-1]], title=dict(text="Elapsed hours", standoff=6), fixedrange=True,
                   showgrid=False, zeroline=False, showline=True, linecolor=TRAY_EDGE, ticks="outside",
                   tickcolor=TRAY_EDGE),
        yaxis=dict(range=[0, 100], tickvals=[0, 50, 100], fixedrange=True, gridcolor="#EEF0F3", zeroline=False,
                   showline=False, ticks=""),
    )
    return fig


def _t_from_points(points, hours, n_t):
    """Turn Streamlit's plotly selection points into a 1-based timepoint (or None)."""
    hours = np.asarray(hours, float)
    for p in points or []:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is not None:
            try:
                return int(min(max(int(cd), 1), n_t))
            except (TypeError, ValueError):
                pass
    for p in points or []:
        if "x" in p:
            try:
                return int(np.abs(hours - float(p["x"])).argmin()) + 1
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------- timepoint player
def _header_html(t, n_t, elapsed, ref):
    clock = ref.strftime("%I:%M %p").lstrip("0")
    date = f"{ref:%b} {ref.day}, {ref.year}"
    return (
        "<div style='display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;margin:0.1rem 0 0.3rem'>"
        f"<span style='font-size:2rem;font-weight:600;line-height:1.1'>{elapsed:.1f} h</span>"
        f"<span style='color:{MUTED}'>elapsed</span>"
        f"<span style='color:{MUTED}'>Timepoint {t} of {n_t}</span>"
        f"<span style='color:{MUTED}'>{clock}, {date}</span></div>")


def _speed_control(container, key):
    if hasattr(st, "segmented_control"):
        container.segmented_control("Speed", list(SPEEDS), default="1x", key=key, label_visibility="collapsed")
    else:
        container.radio("Speed", list(SPEEDS), key=key, horizontal=True, label_visibility="collapsed")


def _player(prefix, n_t, df, meta, figure_fn, series, chart_color, used_speed):
    """Player row, header, plate, footnote and time chart. Runs as a fragment so ticks only redraw this block."""
    tkey, pkey, skey, gkey = f"{prefix}_t", f"{prefix}_play", f"{prefix}_speed", f"{prefix}_gen"
    st.session_state.setdefault(tkey, 1)
    st.session_state.setdefault(pkey, False)
    st.session_state.setdefault(gkey, 0)
    hours, values = series

    if st.session_state[pkey]:
        if st.session_state[tkey] >= n_t:
            st.session_state[pkey] = False
            st.rerun()
        else:
            st.session_state[tkey] += 1

    if n_t > 1:
        c_play, c_prev, c_next, c_speed, _ = st.columns([1, 0.5, 0.5, 2.4, 4], vertical_alignment="center")
        playing = st.session_state[pkey]
        if c_play.button("Pause" if playing else "Play", key=f"{prefix}_btn"):
            if playing:
                st.session_state[pkey] = False
            else:
                if st.session_state[tkey] >= n_t:
                    st.session_state[tkey] = 1
                st.session_state[pkey] = True
            st.rerun()  # full rerun so the fragment timer is redefined
        if c_prev.button("‹", key=f"{prefix}_prev", help="Previous timepoint"):
            st.session_state[tkey] = max(1, st.session_state[tkey] - 1)
        if c_next.button("›", key=f"{prefix}_next", help="Next timepoint"):
            st.session_state[tkey] = min(n_t, st.session_state[tkey] + 1)
        _speed_control(c_speed, skey)
        if (st.session_state.get(skey) or "1x") != used_speed:
            st.rerun()

    t = int(min(max(st.session_state[tkey], 1), n_t))
    sl = df[df["T_index"] == t]
    ref, elapsed = sl["ReferenceTime"].iloc[0], sl["Elapsed_h"].iloc[0]
    st.markdown(_header_html(t, n_t, elapsed, ref), unsafe_allow_html=True)
    st.plotly_chart(figure_fn(sl), config=PLOT_CONFIG, key=f"{prefix}_plot")
    st.markdown(
        f"<div style='text-align:right;font-size:0.75rem;color:{FAINT}'>"
        f"{meta['project']}: {meta['vessel']}, {ref:%Y-%m-%d %H:%M}, {elapsed:.1f} h</div>",
        unsafe_allow_html=True)

    if n_t > 1:
        st.caption("Plate average over time. Click the chart to jump to a timepoint.")
        event = st.plotly_chart(
            time_chart(hours, values, t, chart_color), config={"displayModeBar": False},
            key=f"{prefix}_tc_{st.session_state[gkey]}", on_select="rerun", selection_mode="points")
        points = (event or {}).get("selection", {}).get("points", []) if event else []
        new_t = _t_from_points(points, hours, n_t)
        if new_t is not None:
            st.session_state[tkey] = new_t
            st.session_state[gkey] += 1      # new chart key clears the selection so the next click registers
            try:
                st.rerun(scope="fragment")
            except Exception:
                st.rerun()


def run_player(prefix, n_t, df, meta, figure_fn, series, chart_color):
    playing = st.session_state.get(f"{prefix}_play", False)
    speed_label = st.session_state.get(f"{prefix}_speed") or "1x"
    interval = 1.0 / SPEEDS[speed_label]
    frag = st.fragment(run_every=interval if playing else None)(_player)
    frag(prefix, n_t, df, meta, figure_fn, series, chart_color, speed_label)


def _chart_color(hex_color):
    r, g, b = _rgb(hex_color)
    return hex_color if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.85 else MUTED


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="CM30 Plate Viewer", page_icon="🧫", layout="wide")
    st.title("CM30 Plate Viewer")
    st.markdown(
        "**Step 1:** Export and download your analysis .csv from the CM30  \n"
        "**Step 2:** Extract the files  \n"
        "**Step 3:** Upload the file that contains the averages per well. The filename will contain EV"
    )
    up = st.file_uploader("Upload CM30 .csv", type=["csv"])
    if up is None:
        st.info("Upload a CM30 export to begin.")
        return

    raw = up.getvalue()
    df, meta = parse_cm30(raw)
    if df is None:
        st.error("No well data found. Upload the analysis file whose name contains EV.")
        return

    base = Path(up.name).stem
    n_t = int(df["T_index"].max())
    plate_size = meta["rows"] * meta["cols"]
    values_default = plate_size <= 24
    hours = df.groupby("T_index")["Elapsed_h"].first().to_numpy()
    tab_heat, tab_scratch = st.tabs(["Heatmap", "Scratch assay"])

    with tab_heat:
        left, right = st.columns([1, 3.4], gap="large")
        with left:
            st.markdown("**Display**")
            a, b = st.columns(2)
            vmin = a.number_input("Min %", value=0.0, step=5.0, key="h_min")
            vmax = b.number_input("Max %", value=100.0, step=5.0, key="h_max")
            a, b = st.columns(2)
            col0 = a.color_picker("Color at min", "#FFFFFF", key="h_c0")
            col1 = b.color_picker("Color at max", "#8B0000", key="h_c1")
            show_vals = st.toggle("Show values on wells", value=values_default, key=f"h_vals_{plate_size}")
            st.markdown("**Export**")
            st.download_button("Confluency data (CSV)", confluency_csv(raw), f"{base}_confluency.csv",
                               "text/csv", key="dl_conf")
            st.caption("One row per well and timepoint, ready for pivot tables and Prism.")
        with right:
            if vmax <= vmin:
                st.warning("Max must be greater than min.")
            else:
                mean_conf = df.groupby("T_index")["Confluency"].mean().to_numpy()
                run_player("heat", n_t, df, meta,
                           lambda sl: heatmap_figure(sl, meta, vmin, vmax, col0, col1, show_vals),
                           (hours, mean_conf), _chart_color(col1))

    with tab_scratch:
        left, right = st.columns([1, 3.4], gap="large")
        with left:
            st.markdown("**Display**")
            full = st.number_input("Fully closed at (% confluency)", min_value=1.0, max_value=100.0,
                                   value=95.0, step=1.0, key="s_full",
                                   help="Closure is 100% when a well reaches this confluency. "
                                        "Closure = (current - start) / (this value - start).")
            cap = st.checkbox("Cap closure at 0 to 100%", value=True, key="s_cap")
            a, b = st.columns(2)
            c_start = a.color_picker("Starting cells", "#94A3B8", key="s_c0")
            c_mig = b.color_picker("Migration", "#0F9D8A", key="s_c1")
            show_vals = st.toggle("Show values on wells", value=values_default, key=f"s_vals_{plate_size}")
            bundle = closure_bundle(raw, float(full), bool(cap))
            if bundle["n_bad"]:
                st.warning(f"{bundle['n_bad']} well(s) start at or above {full:g}% confluency, so closure can't "
                           "be calculated for them. Raise the fully closed value if that is unexpected.")
            st.caption("Migration bands show how far cells have moved into the scratch. The gap width is "
                       "schematic. Time zero is the first timepoint.")
            st.markdown("**Export**")
            st.download_button("Closure, long (CSV)", bundle["long_csv"], f"{base}_closure_long.csv",
                               "text/csv", key="dl_long")
            st.download_button("Closure, wide (CSV)", bundle["wide_csv"], f"{base}_closure_wide.csv",
                               "text/csv", key="dl_wide")
            st.caption("Long has one row per well and timepoint, best for pivot tables and Prism. "
                       "Wide has one column per well.")
            with st.expander("Preview export"):
                st.dataframe(bundle["preview"], hide_index=True)
        with right:
            mean_closure = bundle["cdf"].groupby("T_index")["Closure_pct"].mean().to_numpy()
            run_player("scratch", n_t, bundle["cdf"], meta,
                       lambda sl: scratch_figure(sl, meta, c_start, c_mig, show_vals),
                       (hours, mean_closure), _chart_color(c_mig))


if __name__ == "__main__":
    main()
