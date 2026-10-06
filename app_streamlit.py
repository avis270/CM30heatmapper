"""CM30 Plate Viewer: confluency heatmap and scratch assay closure for CM30 exports."""
import io
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

CELL = 100.0        # one well cell in SVG units
WELL_R = 42.0       # well radius in SVG units
GAP_HALF = 0.5      # half-width of the schematic scratch, as a fraction of the well radius
FULL_DRAW = 0.975   # closure above this is drawn as fully closed (no hairline gap)
INTERVALS = {"0.5x": 1.0, "1x": 0.5, "2x": 0.25}   # seconds per timepoint while playing

PLOT_L, PLOT_R = 44, 8   # time chart plot margins in px (the playhead marker is aligned to these)

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
        "wide_preview": wide_df.head(200),
        "n_bad": n_bad,
    }


@st.cache_resource(show_spinner=False, max_entries=4)
def confluency_export(raw: bytes):
    df, _ = parse_cm30(raw)
    tidy = df.sort_values(["RowIdx", "ColIdx", "T_index"]).copy()
    tidy["Time"] = tidy["ReferenceTime"].dt.strftime("%Y-%m-%d %H:%M")
    tidy = tidy.rename(columns={"T_index": "Timepoint"})[
        ["Well", "Row", "Column", "Timepoint", "Time", "Elapsed_h", "Confluency"]]
    return {"csv": tidy.to_csv(index=False).encode(), "preview": tidy.head(200)}


# ---------------------------------------------------------------- colour helpers
def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def lerp_color(c0, c1, t):
    a, b = _rgb(c0), _rgb(c1)
    return tuple(a[k] + (b[k] - a[k]) * t for k in range(3))


def _css(rgb):
    return f"rgb({int(round(rgb[0]))},{int(round(rgb[1]))},{int(round(rgb[2]))})"


def _rgba(hex_color, alpha):
    r, g, b = _rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"


def _text_on(rgb):
    lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return INK if lum > 0.55 else "#FFFFFF"


# ---------------------------------------------------------------- plate drawing (inline SVG)
# The plate is drawn as one SVG string. Streamlit updates it in place, so playback does not flash,
# it scales to any window width, and the same drawing can be downloaded as a vector file.
SVG_FONT = "Source Sans Pro, Inter, Arial, sans-serif"


def _well_label(i, j, meta):
    return str(i * meta["cols"] + j + 1) if meta["numbered"] else f"{chr(65 + i)}{j + 1}"


def _plate_dims(meta, legend_w):
    left = 0 if meta["numbered"] else 46
    top = 40
    pw, ph = meta["cols"] * CELL, meta["rows"] * CELL
    return left, top, pw, ph, left + pw + legend_w + 10, top + ph + 14


def _tray_and_labels(meta, left, top, pw, ph):
    rows, cols = meta["rows"], meta["cols"]
    out = [f'<rect x="{left}" y="{top}" width="{pw:.0f}" height="{ph:.0f}" rx="28" fill="{TRAY}" '
           f'stroke="{TRAY_EDGE}" stroke-width="1.5"/>']
    if not meta["numbered"]:
        for j in range(cols):
            out.append(f'<text x="{left + j * CELL + CELL / 2:.1f}" y="{top - 16}" text-anchor="middle" '
                       f'font-size="16" fill="{MUTED}">{j + 1}</text>')
        for i in range(rows):
            out.append(f'<text x="{left - 16}" y="{top + i * CELL + CELL / 2:.1f}" text-anchor="end" dy=".35em" '
                       f'font-size="16" fill="{MUTED}">{chr(65 + i)}</text>')
    return "".join(out)


def _empty_well(cx, cy):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{WELL_R}" fill="none" stroke="{EMPTY_RING}" '
            f'stroke-width="1.5" stroke-dasharray="3 5"/>')


def _number_tag(cx, cy, label):
    return (f'<text x="{cx - WELL_R * 0.8:.1f}" y="{cy - WELL_R * 0.8:.1f}" text-anchor="middle" font-size="14" '
            f'fill="{MUTED}">{label}</text>')


def _tip_svg(uid, k, cx, cy, lines, on_right, bound_w):
    tw = max(len(s) for s in lines) * 8.4 + 24
    th = len(lines) * 22 + 14
    x = cx + WELL_R + 10 if on_right else cx - WELL_R - 10 - tw
    x = min(max(x, 2), bound_w - tw - 2)
    y = cy - th / 2
    rows_svg = "".join(
        f'<text x="{x + 12:.1f}" y="{y + 25 + n * 22:.1f}" font-size="15" fill="{INK}" '
        f'{"font-weight=" + chr(34) + "700" + chr(34) if n == 0 else ""}>{s}</text>'
        for n, s in enumerate(lines))
    return (f'<g class="tip {uid}t{k}"><rect x="{x:.1f}" y="{y:.1f}" width="{tw:.1f}" height="{th}" rx="8" '
            f'fill="white" stroke="{TRAY_EDGE}" stroke-width="1.5"/>{rows_svg}</g>')


def _svg_frame(uid, W, H, body, tips, n_tips, responsive):
    css = ""
    if responsive:
        css = (f".{uid} .tip{{display:none;pointer-events:none}}"
               f".{uid} .w:hover .ring{{stroke:{INK};stroke-width:3}}"
               + "".join(f".{uid}:has(.{uid}w{k}:hover) .{uid}t{k}{{display:block}}" for k in range(n_tips)))
        size = (f'viewBox="0 0 {W:.0f} {H:.0f}" '
                f'style="width:100%;max-width:{int(W * 1.25)}px;height:auto;display:block;margin:0 auto"')
        bg, tips_svg = "", tips
    else:
        size = f'width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}"'
        bg, tips_svg = f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>', ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="pv {uid}" font-family="{SVG_FONT}" {size}>'
            f'<style>{css}</style>{bg}{body}{tips_svg}</svg>')


def heatmap_svg(sl, meta, vmin, vmax, c0, c1, show_values, responsive=True):
    rows, cols = meta["rows"], meta["cols"]
    uid = "h"
    left, top, pw, ph, W, H = _plate_dims(meta, 150)
    s = sl.dropna(subset=["Confluency"])
    vals = {w: v for w, v in zip(s["Well"], s["Confluency"])}
    fmt = ".1f" if cols <= 6 else ".0f"
    fs = 17 if cols <= 6 else 15 if cols <= 12 else 13

    body = [_tray_and_labels(meta, left, top, pw, ph)]
    tips, k = [], 0
    for i in range(rows):
        for j in range(cols):
            cx, cy = left + j * CELL + CELL / 2, top + i * CELL + CELL / 2
            well = _well_label(i, j, meta)
            if meta["numbered"]:
                body.append(_number_tag(cx, cy, well))
            v = vals.get(well)
            if v is None:
                body.append(_empty_well(cx, cy))
                continue
            t = float(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1))
            rgb = lerp_color(c0, c1, t)
            label = (f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dy=".35em" font-size="{fs}" '
                     f'fill="{_text_on(rgb)}">{v:{fmt}}%</text>') if show_values else ""
            body.append(
                f'<g class="w {uid}w{k}"><circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="{WELL_R}" '
                f'fill="{_css(rgb)}" stroke="rgba(17,24,39,.25)" stroke-width="1.5"/>{label}'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{WELL_R}" fill="transparent"/></g>')
            tips.append(_tip_svg(uid, k, cx, cy, [f"Well {well}", f"Confluency {v:.1f}%"], j < cols / 2, W))
            k += 1

    # scale bar next to the plate
    x0, y0, bh = left + pw + 30, top + 24, ph - 48
    body.append(f'<defs><linearGradient id="{uid}g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" '
                f'stop-color="{c1}"/><stop offset="1" stop-color="{c0}"/></linearGradient></defs>')
    body.append(f'<rect x="{x0}" y="{y0}" width="18" height="{bh}" rx="3" fill="url(#{uid}g)" '
                f'stroke="{TRAY_EDGE}"/>')
    body.append(f'<text x="{x0 - 2}" y="{y0 - 12}" font-size="14" fill="{MUTED}">% confluency</text>')
    for tick in np.linspace(vmin, vmax, 5):
        ty = y0 + (vmax - tick) / (vmax - vmin) * bh
        body.append(f'<text x="{x0 + 28}" y="{ty:.1f}" dy=".35em" font-size="14" fill="{MUTED}">{tick:g}</text>')
    return _svg_frame(uid, W, H, "".join(body), "".join(tips), k, responsive)


def scratch_svg(sl, meta, c_start, c_mig, show_values, responsive=True):
    rows, cols = meta["rows"], meta["cols"]
    uid = "s"
    left, top, pw, ph, W, H = _plate_dims(meta, 180)
    by_well = sl.set_index("Well")
    fmt = ".1f" if cols <= 6 else ".0f"
    fs = 15 if cols <= 6 else 13 if cols <= 12 else 12
    half0 = GAP_HALF * WELL_R
    eps = 0.04 * WELL_R

    body = [_tray_and_labels(meta, left, top, pw, ph)]
    defs, tips, k = [], [], 0
    for i in range(rows):
        for j in range(cols):
            cx, cy = left + j * CELL + CELL / 2, top + i * CELL + CELL / 2
            well = _well_label(i, j, meta)
            if meta["numbered"]:
                body.append(_number_tag(cx, cy, well))
            if well not in by_well.index:
                body.append(_empty_well(cx, cy))
                continue
            row = by_well.loc[well]
            cl, conf, c0v = row["Closure_pct"], row["Confluency"], row["Confluency_T0"]
            circle = f'cx="{cx:.1f}" cy="{cy:.1f}" r="{WELL_R}"'
            hit = f'<circle {circle} fill="transparent"/>'
            if pd.isna(cl):
                body.append(f'<g class="w {uid}w{k}"><circle class="ring" {circle} fill="{NODATA_FILL}" '
                            f'stroke="{EMPTY_RING}" stroke-width="1.5"/>{hit}</g>')
                tips.append(_tip_svg(uid, k, cx, cy, [f"Well {well}", "Closure unavailable",
                                                      f"Starts at {c0v:.1f}% confluency"], j < cols / 2, W))
                k += 1
                continue
            c = float(np.clip(cl / 100, 0, 1))
            gh = half0 * (1 - c)                     # half-width of the gap that is still open
            full = c > FULL_DRAW
            defs.append(f'<clipPath id="{uid}c{k}"><circle {circle}/></clipPath>')
            x, w = cx - WELL_R, 2 * WELL_R
            mig = ""
            if c > 0.002:
                if full:
                    mig = f'<rect x="{x:.1f}" y="{cy - half0 - eps:.1f}" width="{w}" height="{2 * (half0 + eps):.1f}" fill="{c_mig}"/>'
                else:
                    hh = half0 + eps - gh
                    mig = (f'<rect x="{x:.1f}" y="{cy - half0 - eps:.1f}" width="{w}" height="{hh:.1f}" fill="{c_mig}"/>'
                           f'<rect x="{x:.1f}" y="{cy + gh:.1f}" width="{w}" height="{hh:.1f}" fill="{c_mig}"/>')
            start = (f'<rect x="{x:.1f}" y="{cy - WELL_R:.1f}" width="{w}" height="{WELL_R - half0:.1f}" fill="{c_start}"/>'
                     f'<rect x="{x:.1f}" y="{cy + half0:.1f}" width="{w}" height="{WELL_R - half0:.1f}" fill="{c_start}"/>')
            pill = ""
            if show_values:
                txt = f"{cl:{fmt}}%"
                pwid, phei = len(txt) * fs * 0.58 + 16, fs * 1.6
                pill = (f'<rect x="{cx - pwid / 2:.1f}" y="{cy - phei / 2:.1f}" width="{pwid:.1f}" height="{phei:.1f}" '
                        f'rx="{phei / 2:.1f}" fill="rgba(255,255,255,.92)"/>'
                        f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dy=".35em" font-size="{fs}" '
                        f'fill="{INK}">{txt}</text>')
            body.append(
                f'<g class="w {uid}w{k}"><circle {circle} fill="{GAP_FILL}"/>'
                f'<g clip-path="url(#{uid}c{k})">{mig}{start}</g>'
                f'<circle class="ring" {circle} fill="none" stroke="rgba(17,24,39,.28)" stroke-width="1.5"/>'
                f'{pill}{hit}</g>')
            tips.append(_tip_svg(uid, k, cx, cy, [f"Well {well}", f"Closure {cl:.1f}%",
                                                  f"Confluency {conf:.1f}% (start {c0v:.1f}%)"], j < cols / 2, W))
            k += 1

    # key next to the plate
    kx, ky = left + pw + 30, top + ph / 2
    for n, (name, color, stroke) in enumerate([("Starting cells", c_start, "none"), ("Migrated", c_mig, "none"),
                                                ("Open gap", GAP_FILL, EMPTY_RING)]):
        yy = ky + (n - 1) * 36
        body.append(f'<rect x="{kx}" y="{yy - 9:.1f}" width="18" height="18" rx="3" fill="{color}" '
                    f'stroke="{stroke if stroke != "none" else "rgba(17,24,39,.2)"}"/>')
        body.append(f'<text x="{kx + 28}" y="{yy:.1f}" dy=".35em" font-size="15" fill="{MUTED}">{name}</text>')
    return _svg_frame(uid, W, H, f'<defs>{"".join(defs)}</defs>' + "".join(body), "".join(tips), k, responsive)


# ---------------------------------------------------------------- time chart (also the scrubber)
def time_chart(hours, values, color):
    """Plate-average curve. Static (the marker is drawn separately) so it never redraws during playback."""
    hours = np.asarray(hours, float)
    values = np.asarray(values, float)
    n = len(hours)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=values, mode="lines", line=dict(color=color, width=2), fill="tozeroy",
        fillcolor=_rgba(color, 0.16), hoverinfo="skip", showlegend=False))
    # invisible markers on the curve are the click targets; the nearest one wins wherever you click
    fig.add_trace(go.Scatter(
        x=hours, y=values, mode="markers", customdata=np.arange(1, n + 1),
        marker=dict(size=14, color="rgba(0,0,0,0)"), selected=dict(marker=dict(opacity=0)),
        unselected=dict(marker=dict(opacity=0)), hovertemplate="%{x:.1f} h, %{y:.1f}%<extra></extra>",
        showlegend=False))
    fig.update_layout(
        height=108, margin=dict(l=PLOT_L, r=PLOT_R, t=10, b=26), hovermode="closest", hoverdistance=-1,
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=12),
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=12)),
        xaxis=dict(range=[hours[0], hours[-1]], ticksuffix=" h", fixedrange=True, showgrid=False,
                   zeroline=False, showline=True, linecolor=TRAY_EDGE, ticks="outside", tickcolor=TRAY_EDGE),
        yaxis=dict(range=[0, 100], tickvals=[0, 50, 100], fixedrange=True, gridcolor="#EEF0F3",
                   zeroline=False, showline=False, ticks=""),
    )
    return fig


def _playhead_html(hours, t):
    """Marker that sits above the chart at the current time. Plain HTML, so it moves without a redraw."""
    hours = np.asarray(hours, float)
    span = max(hours[-1] - hours[0], 1e-9)
    pct = (hours[t - 1] - hours[0]) / span * 100
    return (f"<div style='margin:0 {PLOT_R}px -0.9rem {PLOT_L}px;position:relative;height:14px'>"
            f"<div style='position:absolute;left:{pct:.2f}%;top:0;transform:translateX(-50%);width:0;height:0;"
            f"border-left:8px solid transparent;border-right:8px solid transparent;border-top:12px solid {INK}'>"
            f"</div></div>")


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
        container.segmented_control("Speed", list(INTERVALS), default="1x", key=key, label_visibility="collapsed")
    else:
        container.radio("Speed", list(INTERVALS), key=key, horizontal=True, label_visibility="collapsed")


def _tc_signature(points):
    out = []
    for p in points or []:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        out.append(cd if cd is not None else p.get("x"))
    return tuple(out)


def _player(prefix, n_t, df, meta, svg_fn, series, chart_color, used_speed, base):
    """Controls, time chart, header, plate and footnote. Runs as a fragment so ticks only redraw this block."""
    tkey, pkey, skey, gkey = f"{prefix}_t", f"{prefix}_play", f"{prefix}_speed", f"{prefix}_gen"
    lkey = f"{prefix}_tc_last"
    st.session_state.setdefault(tkey, 1)
    st.session_state.setdefault(pkey, False)
    st.session_state.setdefault(gkey, 0)
    hours, values = series
    tc_key = f"{prefix}_tc_{st.session_state[gkey]}"

    # a click on the time chart is applied first, before anything is drawn, so it costs one pass
    if n_t > 1:
        state = st.session_state.get(tc_key)
        try:
            pts = state["selection"]["points"] if state else []
        except Exception:  # noqa: BLE001
            pts = []
        sig = _tc_signature(pts)
        if sig and sig != st.session_state.get(lkey):
            new_t = _t_from_points(pts, hours, n_t)
            if new_t is not None:
                st.session_state[tkey] = new_t
            st.session_state[lkey] = sig

    if st.session_state[pkey]:
        if st.session_state[tkey] >= n_t:
            st.session_state[pkey] = False
            st.rerun()
        else:
            st.session_state[tkey] += 1

    if n_t > 1:
        c_play, c_prev, c_next, c_speed, c_hint = st.columns([0.5, 0.5, 0.5, 2.6, 5], vertical_alignment="center",
                                                            gap="small")
        playing = st.session_state[pkey]
        if c_play.button("", key=f"{prefix}_btn", icon=":material/pause:" if playing else ":material/play_arrow:",
                         help="Pause" if playing else "Play"):
            if playing:
                st.session_state[pkey] = False
            else:
                if st.session_state[tkey] >= n_t:
                    st.session_state[tkey] = 1
                st.session_state[pkey] = True
            st.rerun()  # full rerun so the fragment timer is redefined
        if c_prev.button("", key=f"{prefix}_prev", icon=":material/skip_previous:", help="Previous timepoint"):
            st.session_state[tkey] = max(1, st.session_state[tkey] - 1)
            st.session_state[gkey] += 1
            st.session_state[lkey] = None
        if c_next.button("", key=f"{prefix}_next", icon=":material/skip_next:", help="Next timepoint"):
            st.session_state[tkey] = min(n_t, st.session_state[tkey] + 1)
            st.session_state[gkey] += 1
            st.session_state[lkey] = None
        _speed_control(c_speed, skey)
        if (st.session_state.get(skey) or "1x") != used_speed:
            st.rerun()
        c_hint.markdown(f"<div style='text-align:right;font-size:0.85rem;color:{FAINT}'>"
                        "Click the chart to jump to a time</div>", unsafe_allow_html=True)

    t = int(min(max(st.session_state[tkey], 1), n_t))
    tc_key = f"{prefix}_tc_{st.session_state[gkey]}"

    if n_t > 1:
        st.markdown(_playhead_html(hours, t), unsafe_allow_html=True)
        st.plotly_chart(time_chart(hours, values, chart_color), config={"displayModeBar": False},
                        key=tc_key, on_select="rerun", selection_mode="points")

    sl = df[df["T_index"] == t]
    ref, elapsed = sl["ReferenceTime"].iloc[0], sl["Elapsed_h"].iloc[0]
    st.markdown(_header_html(t, n_t, elapsed, ref), unsafe_allow_html=True)
    st.markdown(svg_fn(sl, True), unsafe_allow_html=True)
    c_dl, c_note = st.columns([1.3, 3], vertical_alignment="center")
    playing_now = st.session_state[pkey]
    c_dl.download_button("Download this view (SVG)", b"" if playing_now else svg_fn(sl, False).encode(),
                         f"{base}_{prefix}_T{t:02d}.svg", "image/svg+xml", key=f"{prefix}_svgdl",
                         disabled=playing_now)
    c_note.markdown(
        f"<div style='text-align:right;font-size:0.75rem;color:{FAINT}'>"
        f"{meta['project']}: {meta['vessel']}, {ref:%Y-%m-%d %H:%M}, {elapsed:.1f} h</div>",
        unsafe_allow_html=True)


def run_player(prefix, n_t, df, meta, svg_fn, series, chart_color, base):
    playing = st.session_state.get(f"{prefix}_play", False)
    speed_label = st.session_state.get(f"{prefix}_speed") or "1x"
    interval = INTERVALS[speed_label]
    frag = st.fragment(run_every=interval if playing else None)(_player)
    frag(prefix, n_t, df, meta, svg_fn, series, chart_color, speed_label, base)


def _chart_color(hex_color):
    r, g, b = _rgb(hex_color)
    return hex_color if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.85 else MUTED


TAB_CSS = """<style>
[data-testid="stTab"], button[data-baseweb="tab"] {padding: 0.65rem 1.6rem !important;}
[data-testid="stTab"] p, button[data-baseweb="tab"] p {font-size: 1.2rem !important; font-weight: 700 !important;}
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"]
    {background: rgba(255,75,75,0.10); border-radius: 10px 10px 0 0;}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"] {height: 4px !important;}
</style>"""


# ---------------------------------------------------------------- analysis tab: conditions and group charts
MAX_F = 5
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#8E44AD", "#7A7A7A",
           "#2C3E50", "#B8860B"]
CLOSE_AT = 95.0   # mean closure (%) that counts as "closed" when choosing the default hour
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
AN_CONFIG = {"displaylogo": False, "displayModeBar": "hover",
             "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                        "autoScale2d", "resetScale2d"],
             "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "chart"}}


def _well_pos(label, meta):
    m = re.fullmatch(r"([A-Z]+)(\d+)", label)
    if m:
        return _letters_to_index(m.group(1)), int(m.group(2)) - 1
    return divmod(int(label) - 1, meta["cols"])


def _seg(label, options, key, default=None, **kw):
    default = default if default is not None else options[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key, **kw)
        return v or default
    return st.radio(label, options, key=key, horizontal=True, **kw)


def _level_input(options, key):
    try:
        return st.selectbox("Level", options, index=None, accept_new_options=True,
                            placeholder="Pick or type a level", key=key)
    except TypeError:
        return st.text_input("Level", key=key, placeholder="Type a level")


ERASE = "Erase"


def _init_state(meta, wells):
    ss = st.session_state
    sig = (tuple(wells), meta["rows"], meta["cols"])
    if ss.get("an_sig") != sig:
        ss["an_sig"] = sig
        ss["an_layout"] = pd.DataFrame({"Excluded": False, **{f"f{i}": "" for i in range(MAX_F)}}, index=wells)
        ss["an_nf"] = 2
        ss["an_names"] = ["Cell type", "Treatment"] + [f"Condition {i + 1}" for i in range(2, MAX_F)]
        ss["an_levels"] = {i: [] for i in range(MAX_F)}
        ss["an_colors"], ss["an_ctrl"] = {}, {}
        ss["an_brush"], ss["an_undo"] = None, []
        ss["an_ver"], ss["an_gen"] = 0, 0
        ss["an_snap"] = ss["an_layout"].copy()


def _bump():
    ss = st.session_state
    ss["an_ver"] += 1
    ss["an_gen"] = 0
    ss["an_snap"] = ss["an_layout"].copy()


def _sync_levels():
    """Make sure every level used in the layout (typed, pasted or loaded) is in its condition's level list."""
    ss = st.session_state
    for fi in range(MAX_F):
        lv = ss["an_levels"].setdefault(fi, [])
        for v in pd.unique(ss["an_layout"][f"f{fi}"]):
            if v != "" and v not in lv:
                lv.append(v)
    b = ss.get("an_brush")
    if b and b[1] != ERASE and b[1] not in ss["an_levels"].get(b[0], []):
        ss["an_brush"] = None


def _ordered_levels(fi):
    return list(st.session_state["an_levels"].get(fi, []))


FAMILIES = [
    ["#0072B2", "#56B4E9", "#2C3E50", "#5E60CE", "#7A7A7A", "#8E44AD"],      # condition 1: blues and purples
    ["#E69F00", "#D55E00", "#F4A261", "#C0392B", "#B8860B", "#E76F51"],      # condition 2: oranges and reds
    ["#009E73", "#2A9D8F", "#6A994E", "#95D5B2", "#386641", "#52B788"],      # condition 3: greens
    ["#CC79A7", "#D4537E", "#9D4EDD", "#F28482", "#8E3B6B", "#B5179E"],      # condition 4: pinks
    ["#8D6E63", "#E0C200", "#607D8B", "#A1887F", "#C9A227", "#455A64"],      # condition 5: browns and grays
]


def _color(fi, level):
    """Each condition draws its levels from its own color family, so wedges of different conditions stay apart."""
    ss = st.session_state
    if (fi, level) not in ss["an_colors"]:
        used = sum(1 for k in ss["an_colors"] if k[0] == fi)
        fam = FAMILIES[fi % len(FAMILIES)]
        ss["an_colors"][(fi, level)] = fam[used % len(fam)]
    return ss["an_colors"][(fi, level)]


def _push_undo():
    ss = st.session_state
    ss["an_undo"].append({"layout": ss["an_layout"].copy(), "levels": {k: list(v) for k, v in ss["an_levels"].items()},
                          "ctrl": dict(ss["an_ctrl"]), "nf": ss["an_nf"], "names": list(ss["an_names"])})
    del ss["an_undo"][:-20]


def _undo():
    ss = st.session_state
    if not ss["an_undo"]:
        return
    snap = ss["an_undo"].pop()
    ss["an_layout"], ss["an_levels"], ss["an_ctrl"] = snap["layout"], snap["levels"], snap["ctrl"]
    ss["an_nf"], ss["an_names"] = snap["nf"], snap["names"]
    for i in range(MAX_F):
        ss[f"an_name_w{i}"] = ss["an_names"][i]
    ss["an_brush"] = None
    _bump()


def _clear_plate():
    ss = st.session_state
    _push_undo()
    ss["an_layout"] = pd.DataFrame({"Excluded": False, **{f"f{i}": "" for i in range(MAX_F)}},
                                   index=ss["an_layout"].index)
    _bump()


def _set_brush(fi, level):
    ss = st.session_state
    ss["an_brush"] = None if ss.get("an_brush") == (fi, level) else (fi, level)


def _add_level(fi, key):
    ss = st.session_state
    val = (ss.get(key) or "").strip()
    ss[key] = ""
    if not val:
        return
    if val not in ss["an_levels"][fi]:
        ss["an_levels"][fi].append(val)
    ss["an_brush"] = (fi, val)


def _move_level(fi, k, d):
    lv = st.session_state["an_levels"][fi]
    j = k + d
    if 0 <= j < len(lv):
        lv[k], lv[j] = lv[j], lv[k]


def _set_control(fi, level):
    ss = st.session_state
    ss["an_ctrl"][fi] = None if ss["an_ctrl"].get(fi) == level else level


def _delete_level(fi, level):
    ss = st.session_state
    _push_undo()
    ss["an_levels"][fi] = [x for x in ss["an_levels"][fi] if x != level]
    lay = ss["an_layout"]
    lay.loc[lay[f"f{fi}"] == level, f"f{fi}"] = ""
    if ss["an_ctrl"].get(fi) == level:
        ss["an_ctrl"][fi] = None
    if ss.get("an_brush") == (fi, level):
        ss["an_brush"] = None
    _bump()


def _rename_level(fi, level, key):
    ss = st.session_state
    new = (ss.get(key) or "").strip()
    if not new or new == level or new in ss["an_levels"][fi]:
        ss[key] = level
        return
    _push_undo()
    lv = ss["an_levels"][fi]
    lv[lv.index(level)] = new
    lay = ss["an_layout"]
    lay.loc[lay[f"f{fi}"] == level, f"f{fi}"] = new
    ss["an_colors"][(fi, new)] = ss["an_colors"].pop((fi, level), _color(fi, level))
    if ss["an_ctrl"].get(fi) == level:
        ss["an_ctrl"][fi] = new
    if ss.get("an_brush") == (fi, level):
        ss["an_brush"] = (fi, new)
    _bump()


def _add_condition():
    ss = st.session_state
    if ss["an_nf"] < MAX_F:
        ss["an_nf"] += 1


def _delete_condition(fi):
    ss = st.session_state
    if ss["an_nf"] <= 1:
        return
    _push_undo()
    lay, nf = ss["an_layout"], ss["an_nf"]
    for j in range(fi, nf - 1):
        lay[f"f{j}"] = lay[f"f{j + 1}"]
        ss["an_names"][j] = ss["an_names"][j + 1]
        ss["an_levels"][j] = ss["an_levels"][j + 1]
        ss["an_ctrl"][j] = ss["an_ctrl"].get(j + 1)
    for (f, lvl) in [k for k in ss["an_colors"] if k[0] >= fi]:
        col = ss["an_colors"].pop((f, lvl))
        if f > fi:
            ss["an_colors"][(f - 1, lvl)] = col
    lay[f"f{nf - 1}"] = ""
    ss["an_levels"][nf - 1] = []
    ss["an_ctrl"][nf - 1] = None
    ss["an_names"][nf - 1] = f"Condition {nf}"
    ss["an_nf"] = nf - 1
    ss["an_brush"] = None
    for i in range(MAX_F):
        ss[f"an_name_w{i}"] = ss["an_names"][i]
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith(("an_cp_", "an_rn_", "an_new_"))]:
        del ss[k]
    _bump()


def _toggle_hide():
    st.session_state["an_hide"] = not st.session_state.get("an_hide", False)


UNASSIGNED = "#F3F4F6"


def _active_conditions(nf):
    act = [i for i in range(nf) if st.session_state["an_levels"].get(i)]
    return act or [0]


def _arc(cx, cy, r, a0, a1, n=26):
    th = np.radians(np.linspace(a0, a1, n))
    return cx + r * np.cos(th), cy + r * np.sin(th)


def layout_plate_figure(meta, wells, layout, names, active, click_mode=False):
    """Each well is split into one wedge per active condition (clockwise from the top)."""
    rows, cols = meta["rows"], meta["cols"]
    cap = 110.0 if cols <= 4 else 80.0 if cols <= 6 else 64.0
    cell = min(cap, 600.0 / cols)
    size = cell * 0.84
    r, n = 0.42, len(active)
    groups, ring_x, ring_y, cross_x, cross_y = {}, [], [], [], []
    hx, hy, hov, tag_x, tag_y, tag_t = [], [], [], [], [], []
    for w in wells:
        i, j = _well_pos(w, meta)
        cx, cy = j + 0.5, i + 0.5
        row = layout.loc[w]
        for k, fi in enumerate(active):
            lvl = row[f"f{fi}"]
            color = _color(fi, lvl) if lvl else UNASSIGNED
            ax, ay = _arc(cx, cy, r, -90 + 360 * k / n, -90 + 360 * (k + 1) / n)
            if n > 1:
                ax, ay = np.concatenate([[cx], ax, [cx]]), np.concatenate([[cy], ay, [cy]])
            g = groups.setdefault(color, ([], []))
            g[0].extend(ax.tolist() + [None]); g[1].extend(ay.tolist() + [None])
        rx, ry = _arc(cx, cy, r, 0, 360, 60)
        ring_x.extend(rx.tolist() + [None]); ring_y.extend(ry.tolist() + [None])
        excl = bool(row["Excluded"])
        if excl:
            d = r * 0.55
            cross_x.extend([cx - d, cx + d, None, cx - d, cx + d, None])
            cross_y.extend([cy - d, cy + d, None, cy + d, cy - d, None])
        hx.append(cx); hy.append(cy)
        hov.append(f"<b>Well {w}</b>" + "".join(f"<br>{names[k]}: {row[f'f{k}'] or '-'}" for k in range(len(names)))
                   + ("<br><b>Excluded from analysis</b>" if excl else ""))
        if meta["numbered"]:
            tag_x.append(cx - r * 0.85); tag_y.append(cy - r * 0.85); tag_t.append(w)
    fig = go.Figure()
    for color, (xs, ys) in groups.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=color,
                                 line=dict(color="white", width=1.2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=ring_x, y=ring_y, mode="lines", line=dict(color="#6B7280", width=1),
                             hoverinfo="skip", showlegend=False))
    if cross_x:
        fig.add_trace(go.Scatter(x=cross_x, y=cross_y, mode="lines", line=dict(color="#111827", width=2.4),
                                 hoverinfo="skip", showlegend=False))
    if tag_t:
        fig.add_trace(go.Scatter(x=tag_x, y=tag_y, mode="text", text=tag_t, textfont=dict(size=11, color=MUTED),
                                 hoverinfo="skip", showlegend=False))
    # transparent markers on top: they receive the drag selection and hover
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers", marker=dict(size=size, color="rgba(0,0,0,0)"),
        selected=dict(marker=dict(color="rgba(17,24,39,0.22)", opacity=1)),
        unselected=dict(marker=dict(opacity=1)), hovertext=hov, hoverinfo="text", showlegend=False))
    fig.update_layout(
height=int(rows * cell + 80), margin=dict(l=26, r=8, t=26, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=12),
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=12)),
        xaxis=dict(range=[0, cols], side="top", showgrid=False, zeroline=False, showline=False, ticks="",
                   fixedrange=True, constrain="domain",
                   tickvals=[] if meta["numbered"] else [j + 0.5 for j in range(cols)],
                   ticktext=[] if meta["numbered"] else [str(j + 1) for j in range(cols)]),
        yaxis=dict(range=[rows, 0], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, showline=False,
                   ticks="", fixedrange=True, constrain="domain",
                   tickvals=[] if meta["numbered"] else [i + 0.5 for i in range(rows)],
                   ticktext=[] if meta["numbered"] else [chr(65 + i) for i in range(rows)]))
    if not click_mode:
        # drag mode: a plain rectangle (never a full-height strip)
        fig.update_layout(dragmode="select", selectdirection="d")
    return fig


def _point_poly_dist(px, py, xs, ys):
    """0 if the point is inside the polygon, else its distance to the polygon outline."""
    n = len(xs)
    inside = False
    best = float("inf")
    for k in range(n):
        x1, y1, x2, y2 = xs[k], ys[k], xs[(k + 1) % n], ys[(k + 1) % n]
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
        dx, dy = x2 - x1, y2 - y1
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, ((px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2) ** 0.5)
    return 0.0 if inside else best


def _wells_from_event(event, wells, meta, tol=0.15):
    """Wells chosen by a drag. A box counts a well once it reaches within `tol` of the well's centre."""
    picked = set(_selected_wells(event, wells))
    try:
        boxes = event["selection"]["box"]
    except Exception:  # noqa: BLE001
        boxes = []
    for bx in boxes or []:
        try:
            x0, x1 = sorted(float(v) for v in bx["x"])
            y0, y1 = sorted(float(v) for v in bx["y"])
        except Exception:  # noqa: BLE001
            continue
        for w in wells:
            i, j = _well_pos(w, meta)
            cx, cy = j + 0.5, i + 0.5
            if x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol:
                picked.add(w)
    try:
        lassos = event["selection"]["lasso"]
    except Exception:  # noqa: BLE001
        lassos = []
    for ls in lassos or []:
        try:
            xs, ys = [float(v) for v in ls["x"]], [float(v) for v in ls["y"]]
        except Exception:  # noqa: BLE001
            continue
        if len(xs) < 2:
            continue
        for w in wells:
            i, j = _well_pos(w, meta)
            if _point_poly_dist(j + 0.5, i + 0.5, xs, ys) <= tol:
                picked.add(w)
    return [w for w in wells if w in picked]


def _clear_condition(fi):
    ss = st.session_state
    _push_undo()
    ss["an_layout"][f"f{fi}"] = ""
    _bump()


def _expand_shape(sel, shape, wells, meta):
    if shape == "Wells" or not sel:
        return sel
    pos = {w: _well_pos(w, meta) for w in wells}
    axis = 0 if shape == "Rows" else 1
    keep = {pos[w][axis] for w in sel}
    return [w for w in wells if pos[w][axis] in keep]


# ---------------------------------------------------------------- setup UI
def _toolbar(wells, base):
    ss = st.session_state
    c = st.columns([1.5, 0.9, 1.1, 1.2, 1.2, 1.2], vertical_alignment="center")
    with c[0]:
        _seg("Edit view", ["Plate", "Table"], "an_view", label_visibility="collapsed")
    c[1].button("Undo", on_click=_undo, disabled=not ss["an_undo"], key="an_undo_btn", icon=":material/undo:")
    c[2].button("Clear plate", on_click=_clear_plate, key="an_clear_btn")
    with c[4].popover("Load layout", icon=":material/upload:"):
        st.caption("A layout CSV saved from this tab. Also works as a template: save, fill in the columns, load it back.")
        up = st.file_uploader("Layout CSV", type=["csv"], key="an_up", label_visibility="collapsed")
        if up is not None:
            data = up.getvalue()
            sig = (up.name, len(data), hash(data))
            if ss.get("an_loaded") != sig:
                ss["an_loaded"] = sig
                layout, names, msgs, ok = parse_layout(data, wells)
                if ok:
                    _push_undo()
                    ss["an_layout"] = layout
                    ss["an_nf"] = max(1, len(names))
                    ss["an_names"] = names + [f"Condition {i + 1}" for i in range(len(names), MAX_F)]
                    for i in range(MAX_F):
                        ss[f"an_name_w{i}"] = ss["an_names"][i]
                    ss["an_levels"] = {i: [] for i in range(MAX_F)}
                    ss["an_ctrl"], ss["an_brush"] = {}, None
                    _sync_levels()
                    _bump()
                    ss["an_load_msg"] = ("ok", msgs)
                else:
                    ss["an_load_msg"] = ("error", msgs)
            kind, msgs = ss.get("an_load_msg", ("ok", []))
            if kind == "error":
                st.error(" ".join(msgs))
            else:
                st.success("Layout loaded.")
                for m in msgs:
                    st.warning(m)
    c[5].download_button("Save layout", layout_to_csv(ss["an_layout"], _clean_names(ss["an_names"]), ss["an_nf"], wells),
                         f"{base}_layout.csv", "text/csv", key="an_dl_layout", icon=":material/download:")


def _condition_card(fi, names):
    ss = st.session_state
    brush = ss.get("an_brush")
    with st.container(border=True):
        c_name, c_del = st.columns([5, 2], vertical_alignment="center")
        v = c_name.text_input(f"Condition {fi + 1} name", value=ss["an_names"][fi], key=f"an_name_w{fi}",
                              label_visibility="collapsed", placeholder=f"Condition {fi + 1}")
        ss["an_names"][fi] = v
        c_del.button("Remove", key=f"an_rmc_{fi}", on_click=_delete_condition, args=(fi,),
                     icon=":material/delete:", disabled=ss["an_nf"] <= 1,
                     help="Remove this condition and its levels from the plate" if ss["an_nf"] > 1
                     else "At least one condition is needed")
        levels = _ordered_levels(fi)
        if levels:
            h1, h2, h3 = st.columns([0.8, 3.2, 1.1], vertical_alignment="center")
            h2.caption("Level (click to paint with it)")
            h3.caption("Control")
            for lv in levels:
                c1, c2, c3 = st.columns([0.8, 3.2, 1.1], vertical_alignment="center")
                ss["an_colors"][(fi, lv)] = c1.color_picker(lv, value=_color(fi, lv), key=f"an_cp_{fi}_{lv}",
                                                            label_visibility="collapsed")
                c2.button(lv, key=f"an_lv_{fi}_{lv}", on_click=_set_brush, args=(fi, lv),
                          type="primary" if brush == (fi, lv) else "secondary")
                is_ctrl = ss["an_ctrl"].get(fi) == lv
                c3.button("★" if is_ctrl else "☆", key=f"an_ct_{fi}_{lv}", on_click=_set_control, args=(fi, lv),
                          help="Use as the control level when normalizing")
        else:
            st.caption("No levels yet. Type one below and press Enter.")
        nk = f"an_new_{fi}"
        st.text_input(f"New level for {names[fi]}", key=nk, label_visibility="collapsed",
                      placeholder="Add a level and press Enter", on_change=_add_level, args=(fi, nk))
        b1, b2 = st.columns(2)
        b1.button("Eraser", key=f"an_er_{fi}", on_click=_set_brush, args=(fi, ERASE), icon=":material/ink_eraser:",
                  type="primary" if brush == (fi, ERASE) else "secondary", help="Paint over wells to remove this condition")
        b2.button("Clear all", key=f"an_cl_{fi}", on_click=_clear_condition, args=(fi,),
                  help=f"Remove {names[fi]} from every well (levels are kept)")
        with st.popover("Rename, reorder or delete", icon=":material/tune:"):
            if not levels:
                st.caption("Levels you add show up here.")
            for k, lv in enumerate(levels):
                c1, c2, c3, c4 = st.columns([3, 0.8, 0.8, 0.8], vertical_alignment="center")
                rk = f"an_rn_{fi}_{lv}"
                c1.text_input(f"Rename {lv}", value=lv, key=rk, label_visibility="collapsed",
                              on_change=_rename_level, args=(fi, lv, rk))
                c2.button("▲", key=f"an_up_{fi}_{lv}", on_click=_move_level, args=(fi, k, -1), disabled=k == 0,
                          help="Move up")
                c3.button("▼", key=f"an_dn_{fi}_{lv}", on_click=_move_level, args=(fi, k, 1),
                          disabled=k == len(levels) - 1, help="Move down")
                c4.button("✕", key=f"an_dl_{fi}_{lv}", on_click=_delete_level, args=(fi, lv),
                          help="Delete this level (its wells become unassigned)")


def _conditions_panel(names, nf):
    st.markdown("**Conditions**")
    for fi in range(nf):
        _condition_card(fi, names)
    st.button("Add condition", on_click=_add_condition, disabled=nf >= MAX_F, key="an_add",
              icon=":material/add:")


def _plate_legend(names, nf, active):
    ss = st.session_state
    dot = ("<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
           "<span style='width:12px;height:12px;border-radius:50%;background:{c};display:inline-block'></span>{t}</span>")
    rows = []
    for fi in active:
        chips = "".join(dot.format(c=_color(fi, x), t=x) for x in _ordered_levels(fi))
        rows.append(f"<div style='margin-bottom:.25rem'><b>{names[fi]}</b> &nbsp; {chips}</div>")
    rows.append(dot.format(c="#F3F4F6;border:1px solid #9CA3AF", t="Unassigned")
                + "<span style='display:inline-flex;align-items:center;gap:.3rem'>"
                  "<span style='font-weight:700'>&#10005;</span> Excluded</span>")
    note = ""
    if len(active) > 1:
        note = (f"<div style='color:{MUTED};margin-top:.3rem'>Each well is split into one wedge per condition, "
                f"clockwise from the top: {', '.join(names[i] for i in active)}.</div>")
    st.markdown(f"<div style='font-size:.85rem'>{''.join(rows)}{note}</div>", unsafe_allow_html=True)


SHAPES = {"Just the wells": "Wells", "Whole rows": "Rows", "Whole columns": "Columns"}


def _plate_key():
    ss = st.session_state
    code = "c" if ss.get("an_selby", "Drag a box") == "Click wells" else "d"
    return f"an_plate_{ss['an_ver']}_{ss['an_gen']}_{code}"


def _apply_pending_paint(wells, meta):
    """Apply the last plate selection before anything is drawn, so a paint costs one pass and no flicker."""
    ss = st.session_state
    if ss.get("an_view", "Plate") != "Plate":
        return
    state = ss.get(_plate_key())
    if not state:
        return
    shape = SHAPES.get(ss.get("an_shape3", "Just the wells"), "Wells")
    sel = _expand_shape(_wells_from_event(state, wells, meta), shape, wells, meta)
    if not sel:
        return
    layout = ss["an_layout"]
    if ss.get("an_mode", "Paint conditions") == "Exclude wells":
        _push_undo()
        layout.loc[sel, "Excluded"] = ss.get("an_exclact", "Exclude") == "Exclude"
        _bump()
    elif ss.get("an_brush"):
        _push_undo()
        fi, lv = ss["an_brush"]
        layout.loc[sel, f"f{fi}"] = "" if lv == ERASE else lv
        _bump()


def _plate_legend(names, nf, active):
    dot = ("<span style='display:inline-flex;align-items:center;gap:.3rem;margin-right:.8rem'>"
           "<span style='width:12px;height:12px;border-radius:50%;background:{c};display:inline-block'></span>{t}</span>")
    rows = []
    for fi in active:
        chips = "".join(dot.format(c=_color(fi, x), t=x) for x in _ordered_levels(fi))
        rows.append(f"<div style='margin-bottom:.25rem'><b>{names[fi]}</b> &nbsp; {chips}</div>")
    rows.append(dot.format(c="#F3F4F6;border:1px solid #9CA3AF", t="Unassigned")
                + "<span style='display:inline-flex;align-items:center;gap:.3rem'>"
                  "<span style='font-weight:700'>&#10005;</span> Excluded</span>")
    note = ""
    if len(active) > 1:
        note = (f"<div style='color:{MUTED};margin-top:.3rem'>Each well is split into one wedge per condition, "
                f"clockwise from the top: {', '.join(names[i] for i in active)}.</div>")
    st.markdown(f"<div style='font-size:.85rem'>{''.join(rows)}{note}</div>", unsafe_allow_html=True)


def _plate_panel(wells, meta, names, nf):
    ss = st.session_state
    layout = ss["an_layout"]
    m1, m2 = st.columns([1.25, 1.25])
    m1.caption("Mode")
    with m1:
        mode = _seg("Mode", ["Paint conditions", "Exclude wells"], "an_mode", label_visibility="collapsed")
    m2.caption("Select by")
    with m2:
        selby = _seg("Select by", ["Drag a box", "Click wells"], "an_selby", label_visibility="collapsed")
    st.caption("Each selection covers")
    shape_label = _seg("Each selection covers", list(SHAPES), "an_shape3", label_visibility="collapsed")
    click_mode = selby == "Click wells"
    verb = "Click" if click_mode else "Drag over"
    brush = ss.get("an_brush")
    exclude_mode = mode == "Exclude wells"
    action = "Exclude"
    if exclude_mode:
        action = _seg("Action", ["Exclude", "Restore"], "an_exclact", label_visibility="collapsed")
        st.markdown(
            f"<div style='background:rgba(255,193,7,.16);border-radius:8px;padding:8px 12px;font-size:.9rem'>"
            f"{verb} wells to {action.lower()} them. Excluded wells keep their conditions and are left out of "
            f"every chart and table.</div>", unsafe_allow_html=True)
    elif brush:
        fi, lv = brush
        dot = "#9CA3AF" if lv == ERASE else _color(fi, lv)
        label = f"Erasing {names[fi]}" if lv == ERASE else f"Painting {names[fi]}: {lv}"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.6rem;background:rgba(0,114,178,.08);border-radius:8px;"
            f"padding:8px 12px;font-size:.95rem'><span style='width:14px;height:14px;border-radius:50%;"
            f"background:{dot};display:inline-block'></span><b>{label}</b>"
            f"<span style='color:{MUTED}'>{verb} wells to paint. Click the level again to stop.</span></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:rgba(128,128,128,.10);border-radius:8px;padding:8px 12px;"
                    f"font-size:.95rem;color:{MUTED}'>Click a level on the left, then {verb.lower()} wells to paint "
                    f"them.</div>", unsafe_allow_html=True)

    if ss.get("an_view", "Plate") == "Table":
        snap = ss["an_snap"]
        tbl = pd.DataFrame({"Well": snap.index})
        for i in range(nf):
            tbl[names[i]] = snap[f"f{i}"].to_numpy()
        tbl["Excluded"] = snap["Excluded"].to_numpy()
        st.caption("Edit cells or paste a block from Excel.")
        cfg = {n: st.column_config.TextColumn(n) for n in names[:nf]}
        cfg["Excluded"] = st.column_config.CheckboxColumn("Excluded")
        edited = st.data_editor(tbl, hide_index=True, disabled=["Well"], column_config=cfg, height=420,
                                key=f"an_tbl_{ss['an_ver']}_{hash(tuple(names[:nf]))}")
        for i in range(nf):
            layout[f"f{i}"] = edited[names[i]].fillna("").astype(str).str.strip().to_numpy()
        layout["Excluded"] = edited["Excluded"].fillna(False).astype(bool).to_numpy()
        _sync_levels()
        return

    active = _active_conditions(nf)
    event = st.plotly_chart(
        layout_plate_figure(meta, wells, layout, names[:nf], active, click_mode), key=_plate_key(),
        on_select="rerun", selection_mode=("points",) if click_mode else ("points", "box", "lasso"),
        config={"displayModeBar": False})
    pending = _wells_from_event(event, wells, meta)
    if pending and not exclude_mode and not brush:
        st.caption(f"{len(pending)} well(s) selected. Click a level on the left to paint them.")

    _plate_legend(names, nf, active)
    excl = list(layout.index[layout["Excluded"]])
    if excl:
        st.caption(f"{len(excl)} excluded: {', '.join(excl[:12])}{'...' if len(excl) > 12 else ''}")


def _summary_bar(wells, nf, hide):
    ss = st.session_state
    lay = ss["an_layout"]
    assigned = int((lay[[f"f{i}" for i in range(nf)]] != "").any(axis=1).sum())
    excl = int(lay["Excluded"].sum())
    text = f"**Plate setup:** {nf} condition{'s' if nf != 1 else ''}, {assigned} of {len(wells)} wells assigned"
    if excl:
        text += f", {excl} excluded"
    with st.container(border=True):
        a, b = st.columns([4, 1.2], vertical_alignment="center")
        a.markdown(text)
        b.button("Show setup" if hide else "Collapse setup", on_click=_toggle_hide, key="an_hide_btn",
                 icon=":material/expand_more:" if hide else ":material/expand_less:")


def _clean_names(names):
    out, seen = [], set()
    for i, n in enumerate(names):
        n = (n or "").strip() or f"Condition {i + 1}"
        base, k = n, 2
        while n.lower() in seen or n.lower() in ("well", "excluded"):
            n, k = f"{base} ({k})", k + 1
        seen.add(n.lower())
        out.append(n)
    return out


def layout_to_csv(layout, names, nf, wells):
    out = pd.DataFrame({"Well": wells})
    for i in range(nf):
        out[names[i]] = layout.loc[wells, f"f{i}"].to_numpy()
    out["Excluded"] = np.where(layout.loc[wells, "Excluded"].to_numpy(), "yes", "")
    return out.to_csv(index=False).encode()


def parse_layout(raw_bytes, wells):
    """Read a layout CSV in our format. Returns (layout, names, messages, ok)."""
    msgs = []
    try:
        tab = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return None, None, [f"Could not read the file: {exc}"], False
    cols = {c.strip().lower(): c for c in tab.columns}
    if "well" not in cols:
        return None, None, ["The file needs a 'Well' column."], False
    wcol = cols["well"]
    ecol = cols.get("excluded")
    fcols = [c for c in tab.columns if c not in (wcol, ecol) and c.strip()]
    if len(fcols) > MAX_F:
        return None, None, [f"The file has {len(fcols)} condition columns. The maximum is {MAX_F}."], False

    def norm(w):
        w = w.strip().upper()
        return w[4:] if w.startswith("WELL") else w

    ids = tab[wcol].map(norm)
    if ids.duplicated().any():
        dup = ", ".join(ids[ids.duplicated()].unique()[:6])
        return None, None, [f"Duplicate wells in the file: {dup}."], False
    unknown = [w for w in ids if w not in set(wells)]
    if unknown:
        msgs.append(f"{len(unknown)} well(s) in the file are not on this plate and were skipped: "
                    f"{', '.join(unknown[:8])}{'...' if len(unknown) > 8 else ''}.")
    layout = pd.DataFrame({"Excluded": False, **{f"f{i}": "" for i in range(MAX_F)}}, index=wells)
    tab = tab.assign(_id=ids)
    tab = tab[tab["_id"].isin(set(wells))]
    for i, c in enumerate(fcols):
        layout.loc[tab["_id"].to_numpy(), f"f{i}"] = tab[c].str.strip().to_numpy()
    if ecol is not None:
        flag = tab[ecol].str.strip().str.lower().isin(["yes", "true", "1", "x", "y"])
        layout.loc[tab["_id"].to_numpy(), "Excluded"] = flag.to_numpy()
    missing = len(wells) - len(tab)
    if missing:
        msgs.append(f"{missing} well(s) on this plate were not in the file and stay unassigned.")
    names = _clean_names([c.strip() for c in fcols] or ["Condition 1"])
    return layout, names, msgs, True


def _selected_wells(state, wells):
    try:
        idx = state["selection"]["point_indices"]
    except Exception:  # noqa: BLE001
        return []
    return [wells[i] for i in idx if 0 <= i < len(wells)]


# ---- group statistics
def group_data(cl, xi, ci, col, normalize, ctrl):
    d = cl[(~cl["Excluded"]) & (cl[f"f{xi}"] != "")]
    if ci is not None:
        d = d[d[f"f{ci}"] != ""]
    d = d.assign(X=d[f"f{xi}"], C=(d[f"f{ci}"] if ci is not None else ""), V=d[col]).dropna(subset=["V"])
    if normalize and ctrl:
        cm = d[d["X"] == ctrl].groupby(["T_index", "C"])["V"].mean().rename("CM").reset_index()
        d = d.merge(cm, on=["T_index", "C"], how="left")
        d["V"] = np.where(d["CM"] > 1e-6, d["V"] / d["CM"] * 100, np.nan)
        d = d.dropna(subset=["V"])
    return d


def _agg(d, keys):
    g = d.groupby(keys)["V"].agg(n="count", mean="mean", sd="std").reset_index()
    g["sem"] = g["sd"] / np.sqrt(g["n"])
    return g


def default_hour(d_raw, n_t):
    """First timepoint at which any group's mean closure reaches CLOSE_AT, else the last timepoint."""
    if d_raw.empty:
        return n_t
    ts = _agg(d_raw, ["X", "C", "T_index"])
    hit = ts[ts["mean"] >= CLOSE_AT]
    return int(hit["T_index"].min()) if len(hit) else n_t


def _group_label(x, c):
    return f"{x} / {c}" if c else x


def bar_figure(s, g, x_levels, c_levels, x_color, c_color, err, ytitle, cname, has_c):
    fig = go.Figure()
    ncl = max(len(c_levels), 1)
    w = 0.8 / ncl
    rng = np.random.default_rng(0)
    for k, cl_ in enumerate(c_levels if has_c else [""]):
        off = (k - (ncl - 1) / 2) * w
        sub_g = g[g["C"] == cl_].set_index("X")
        bx, by, be, bcol, hov = [], [], [], [], []
        for i, xl in enumerate(x_levels):
            if xl not in sub_g.index:
                continue
            row = sub_g.loc[xl]
            bx.append(i + off); by.append(row["mean"])
            e = row[err]
            be.append(0 if pd.isna(e) else e)
            bcol.append(c_color.get(cl_, MUTED) if has_c else x_color.get(xl, MUTED))
            hov.append(f"<b>{_group_label(xl, cl_)}</b><br>mean {row['mean']:.1f}<br>n = {int(row['n'])}")
        fig.add_trace(go.Bar(
            x=bx, y=by, width=w * 0.92, marker=dict(color=bcol), name=cl_ if has_c else "", showlegend=has_c,
            error_y=dict(type="data", array=be, visible=True, color=INK, thickness=1.5, width=4),
            hovertext=hov, hoverinfo="text"))
        pts = s[s["C"] == cl_]
        px, py, ph = [], [], []
        for i, xl in enumerate(x_levels):
            v = pts[pts["X"] == xl]
            for wl_, val in zip(v["Well"], v["V"]):
                px.append(i + off + rng.uniform(-w * 0.22, w * 0.22)); py.append(val)
                ph.append(f"Well {wl_}: {val:.1f}")
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers", showlegend=False, hovertext=ph, hoverinfo="text",
                                 marker=dict(size=7, color="rgba(17,24,39,.55)", line=dict(color="white", width=1))))
    fig.update_layout(
        height=380, margin=dict(l=60, r=10, t=20, b=50), barmode="overlay", bargap=0,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=13),
        legend=dict(title=dict(text=cname), orientation="h", y=1.08, x=0),
        xaxis=dict(tickvals=list(range(len(x_levels))), ticktext=x_levels,
                   range=[-0.5 - max(0, (5 - len(x_levels)) / 2), len(x_levels) - 0.5 + max(0, (5 - len(x_levels)) / 2)],
                   showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(title=ytitle, gridcolor="#EEF0F3", zeroline=False, rangemode="tozero", fixedrange=True),
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=12)))
    return fig


def curve_figure(ts, x_levels, c_levels, x_color, c_color, err, ytitle, now_h, has_c):
    fig = go.Figure()
    for xi_, xl in enumerate(x_levels):
        for cl_ in (c_levels if has_c else [""]):
            sub = ts[(ts["X"] == xl) & (ts["C"] == cl_)].sort_values("Elapsed_h")
            if sub.empty:
                continue
            color = c_color.get(cl_, MUTED) if has_c else x_color.get(xl, MUTED)
            dash = DASHES[xi_ % len(DASHES)] if has_c else "solid"
            e = sub[err].fillna(0)
            fig.add_trace(go.Scatter(x=sub["Elapsed_h"], y=sub["mean"] + e, mode="lines", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=sub["Elapsed_h"], y=sub["mean"] - e, mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=_rgba(color, 0.14), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=sub["Elapsed_h"], y=sub["mean"], mode="lines", name=_group_label(xl, cl_),
                                     line=dict(color=color, width=2.2, dash=dash),
                                     hovertemplate=f"<b>{_group_label(xl, cl_)}</b><br>%{{x:.1f}} h, %{{y:.1f}}<extra></extra>"))
    fig.add_vline(x=now_h, line=dict(color=INK, width=1, dash="dot"))
    fig.update_layout(
        height=340, margin=dict(l=60, r=10, t=20, b=50), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, color=MUTED, size=13),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(title="Elapsed hours", showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(title=ytitle, gridcolor="#EEF0F3", zeroline=False, rangemode="tozero", fixedrange=True),
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=12)))
    return fig


def _analysis_panel_body(raw, df, meta, base, names, nf, n_t, hours):
    """Right side: choose the analysis, then charts, table and downloads."""
    ss = st.session_state
    layout = ss["an_layout"]
    used = [i for i in range(nf) if (layout[f"f{i}"] != "").any()]
    if not used:
        st.info("Name your conditions and assign wells on the left. Charts appear here as soon as at least "
                "one condition has levels.")
        st.markdown("**How it works**  \n1. Name up to 5 conditions, for example cell type or media, and add their "
                    "levels.  \n2. Click a level, then drag over wells on the plate to paint them. Use Undo if you "
                    "slip. You can also paste into the table view or load a saved layout.  \n"
                    "3. Pick what to compare and read the charts.")
        return

    r1 = st.columns([1.5, 1.2, 1.2, 1, 1.3])
    measure = r1[0].selectbox("Measure", ["Scratch closure", "Confluency"], key="an_measure")
    multi = [i for i in used if layout.loc[layout[f"f{i}"] != "", f"f{i}"].nunique() >= 2]
    x_default = (multi or used)[0]
    xi = r1[1].selectbox("Compare (x-axis)", used, index=used.index(x_default), format_func=lambda i: names[i],
                         key=f"an_x_{hash((tuple(used), tuple(multi)))}")
    copts = ["None"] + [i for i in used if i != xi]
    cv = r1[2].selectbox("Color by", copts, format_func=lambda i: "None" if i == "None" else names[i],
                         key=f"an_c_{hash(tuple(copts))}")
    ci = None if cv == "None" else cv
    err_name = r1[3].selectbox("Error bars", ["SD", "SEM"], key="an_err")
    err = "sd" if err_name == "SD" else "sem"
    ctrl = ss["an_ctrl"].get(xi)
    normalize = r1[4].checkbox("Normalize to control", value=False, key="an_norm", disabled=not ctrl,
                               help="Divides each value by the control level's mean (same color group, same time) "
                                    "and shows % of control. Star a control level in the condition card.")
    if not ctrl:
        r1[4].caption(f"Star a control level for {names[xi]} on the left.")
    normalize = bool(normalize and ctrl)

    closure = measure == "Scratch closure"
    if closure:
        r2 = st.columns([1.3, 1.3, 3])
        full = r2[0].number_input("Fully closed at (% confluency)", min_value=1.0, max_value=100.0, value=95.0,
                                  step=1.0, key="an_full")
        cap = r2[1].checkbox("Cap closure at 0 to 100%", value=True, key="an_cap")
    else:
        full, cap = 95.0, True
    bundle = closure_bundle(raw, float(full), bool(cap))
    cdf = bundle["cdf"]
    lay = layout.reset_index().rename(columns={"index": "Well"})
    cl = cdf.merge(lay, on="Well", how="left")
    cl["Excluded"] = cl["Excluded"].fillna(False).astype(bool)
    for i in range(MAX_F):
        cl[f"f{i}"] = cl[f"f{i}"].fillna("")

    col = "Closure_pct" if closure else "Confluency"
    ytitle = ("% of control" if normalize else ("% closure" if closure else "% confluency"))
    d_all = group_data(cl, xi, ci, col, normalize, ctrl)
    if d_all.empty:
        st.warning("No wells have a level for the chosen conditions yet.")
        return

    # default hour: first time any group reaches full closure
    if closure:
        d_raw = group_data(cl, xi, ci, "Closure_pct", False, None)
        t_def = default_hour(d_raw, n_t)
    else:
        t_def = n_t
    sig = (measure, float(full), bool(cap), xi, ci, int(pd.util.hash_pandas_object(layout, index=True).sum() % 10**9))
    t = st.select_slider("Hour for the bar chart and table", options=list(range(1, n_t + 1)), value=t_def,
                         format_func=lambda k: f"{hours[k - 1]:.1f} h", key=f"an_hour_{hash(sig)}")
    if closure:
        st.caption(f"Default is the first hour any group's mean closure reaches {CLOSE_AT:g}% "
                   f"({hours[t_def - 1]:.1f} h), or the last timepoint if none does. Move the slider to change it.")

    x_levels = [x for x in _ordered_levels(xi) if x in set(d_all["X"])]
    c_levels = [c for c in (_ordered_levels(ci) if ci is not None else [""]) if c in set(d_all["C"])]
    x_color = {x: _color(xi, x) for x in x_levels}
    c_color = {c: _color(ci, c) for c in c_levels} if ci is not None else {}
    cname = names[ci] if ci is not None else ""

    s = d_all[d_all["T_index"] == t]
    g = _agg(s, ["X", "C"])
    if g.empty:
        st.warning("No data at this hour for the chosen groups.")
        return

    st.markdown(f"**{measure} by {names[xi]}" + (f" and {names[ci]}" if ci is not None else "")
                + f" at {hours[t - 1]:.1f} h**")
    st.plotly_chart(bar_figure(s, g, x_levels, c_levels, x_color, c_color, err, ytitle, cname, ci is not None),
                    config=AN_CONFIG, key="an_bar")
    if (g["n"] == 1).any():
        st.caption("Some groups have a single well, so they have no error bars.")

    ts = _agg(d_all, ["X", "C", "T_index", "Elapsed_h"])
    st.markdown("**Over time**")
    st.plotly_chart(curve_figure(ts, x_levels, c_levels, x_color, c_color, err, ytitle, hours[t - 1], ci is not None),
                    config=AN_CONFIG, key="an_curve")

    # summary table and downloads
    g["Group"] = [_group_label(x, c) for x, c in zip(g["X"], g["C"])]
    g["xo"] = g["X"].map({x: k for k, x in enumerate(x_levels)})
    g["co"] = g["C"].map({c: k for k, c in enumerate(c_levels)})
    g = g.sort_values(["xo", "co"])
    cols = {"Group": "Group", "n": "n", "mean": "Mean", "sd": "SD", "sem": "SEM"}
    summ = g[list(cols)].rename(columns=cols)
    if ci is not None:
        summ.insert(1, names[ci], g["C"].to_numpy())
        summ.insert(1, names[xi], g["X"].to_numpy())
        summ = summ.drop(columns="Group")
    else:
        summ.insert(1, names[xi], g["X"].to_numpy())
        summ = summ.drop(columns="Group")
    summ = summ.round({"Mean": 2, "SD": 2, "SEM": 2})
    summ.insert(0, "Hour", round(float(hours[t - 1]), 2))
    summ.insert(1, "Measure", ytitle)
    st.markdown("**Summary**")
    st.dataframe(summ, hide_index=True)

    s2 = s.assign(Group=[_group_label(x, c) for x, c in zip(s["X"], s["C"])])
    order = list(g["Group"])
    prism = s2.assign(Rep=s2.groupby("Group").cumcount() + 1).pivot(index="Rep", columns="Group", values="V")
    prism = prism.reindex(columns=order).reset_index()
    prism.columns.name = None

    long_out = cl.sort_values(["RowIdx", "ColIdx", "T_index"]).copy()
    long_out["Time"] = long_out["ReferenceTime"].dt.strftime("%Y-%m-%d %H:%M")
    long_out = long_out.rename(columns={f"f{i}": names[i] for i in range(nf)}).rename(columns={"T_index": "Timepoint"})
    long_out["Excluded"] = np.where(long_out["Excluded"], "yes", "")
    keep = ["Well", "Row", "Column"] + [names[i] for i in range(nf)] + ["Excluded", "Timepoint", "Time", "Elapsed_h",
                                                                         "Confluency", "Closure_pct"]
    st.markdown("**Export**")
    e1, e2, e3, _ = st.columns([1, 1, 1.4, 1])
    e1.download_button("Summary (CSV)", summ.to_csv(index=False).encode(), f"{base}_group_summary.csv",
                       "text/csv", key="an_dl_sum")
    e2.download_button("Prism layout (CSV)", prism.to_csv(index=False).encode(), f"{base}_group_prism.csv",
                       "text/csv", key="an_dl_prism")
    e3.download_button("All wells with conditions (CSV)", long_out[keep].to_csv(index=False).encode(),
                       f"{base}_wells_with_conditions.csv", "text/csv", key="an_dl_long")
    st.caption("Prism layout has one column per group and one row per replicate, at the hour chosen above.")


@st.fragment
def analysis_panel(raw, df, meta, base):
    ss = st.session_state
    wells = meta["well_order"]
    _init_state(meta, wells)
    _sync_levels()
    n_t = int(df["T_index"].max())
    hours = df.groupby("T_index")["Elapsed_h"].first().to_numpy()
    hide = ss.get("an_hide", False)
    if not hide:
        _apply_pending_paint(wells, meta)
        _toolbar(wells, base)
        names, nf = _clean_names(ss["an_names"]), ss["an_nf"]
        left, right = st.columns([1, 1.5], gap="large")
        with left:
            _conditions_panel(names, nf)
        with right:
            _plate_panel(wells, meta, _clean_names(ss["an_names"]), nf)
    names, nf = _clean_names(ss["an_names"]), ss["an_nf"]
    _summary_bar(wells, nf, hide)
    _analysis_panel_body(raw, df, meta, base, names, nf, n_t, hours)


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="CM30 Plate Viewer", page_icon="🧫", layout="wide")
    st.markdown(TAB_CSS, unsafe_allow_html=True)
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
    tab_heat, tab_scratch, tab_analysis = st.tabs(["Heatmap", "Scratch assay", "Analysis"])

    with tab_heat:
        left, right = st.columns([1, 3.4], gap="large")
        conf = confluency_export(raw)
        with left:
            st.markdown("**Scale**")
            a, b = st.columns(2)
            vmin = a.number_input("Min %", value=0.0, step=5.0, key="h_min")
            vmax = b.number_input("Max %", value=100.0, step=5.0, key="h_max")
            st.markdown("**Colors**")
            a, b = st.columns(2)
            col0 = a.color_picker("Low color (at Min %)", "#FFFFFF", key="h_c0")
            col1 = b.color_picker("High color (at Max %)", "#8B0000", key="h_c1")
            st.caption("Click a swatch to pick a color.")
            st.markdown("**Wells**")
            show_vals = st.toggle("Show values on wells", value=values_default, key=f"h_vals_{plate_size}")
            st.markdown("**Export**")
            st.download_button("Confluency data (CSV)", conf["csv"], f"{base}_confluency.csv",
                               "text/csv", key="dl_conf")
            st.caption("One row per well and timepoint, ready for pivot tables and Prism.")
        with right:
            if vmax <= vmin:
                st.warning("Max must be greater than min.")
            else:
                mean_conf = df.groupby("T_index")["Confluency"].mean().to_numpy()
                run_player("heat", n_t, df, meta,
                           lambda sl, resp: heatmap_svg(sl, meta, vmin, vmax, col0, col1, show_vals, resp),
                           (hours, mean_conf), _chart_color(col1), base)
        with st.expander("Preview export data"):
            st.dataframe(conf["preview"], hide_index=True)

    with tab_scratch:
        left, right = st.columns([1, 3.4], gap="large")
        with left:
            st.markdown("**Closure**")
            full = st.number_input("Fully closed at (% confluency)", min_value=1.0, max_value=100.0,
                                   value=95.0, step=1.0, key="s_full",
                                   help="Closure is 100% when a well reaches this confluency. "
                                        "Closure = (current - start) / (this value - start).")
            cap = st.checkbox("Cap closure at 0 to 100%", value=True, key="s_cap")
            st.markdown("**Colors**")
            a, b = st.columns(2)
            c_start = a.color_picker("Starting cells color", "#94A3B8", key="s_c0")
            c_mig = b.color_picker("Migration color", "#0F9D8A", key="s_c1")
            st.caption("Click a swatch to pick a color.")
            st.markdown("**Wells**")
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
        with right:
            mean_closure = bundle["cdf"].groupby("T_index")["Closure_pct"].mean().to_numpy()
            run_player("scratch", n_t, bundle["cdf"], meta,
                       lambda sl, resp: scratch_svg(sl, meta, c_start, c_mig, show_vals, resp),
                       (hours, mean_closure), _chart_color(c_mig), base)
        with st.expander("Preview export data"):
            t_long, t_wide = st.tabs(["Long", "Wide"])
            t_long.dataframe(bundle["preview"], hide_index=True)
            t_wide.dataframe(bundle["wide_preview"], hide_index=True)

    with tab_analysis:
        analysis_panel(raw, df, meta, base)


if __name__ == "__main__":
    main()
