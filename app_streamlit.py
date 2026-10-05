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


def _player(prefix, n_t, df, meta, svg_fn, series, chart_color, used_speed, base):
    """Controls, time chart, header, plate and footnote. Runs as a fragment so ticks only redraw this block."""
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
        c_play, c_prev, c_next, c_speed, c_hint = st.columns([1, 0.5, 0.5, 2.4, 3.2], vertical_alignment="center")
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
        c_hint.markdown(f"<div style='text-align:right;font-size:0.85rem;color:{FAINT}'>"
                        "Click the chart to jump to a time</div>", unsafe_allow_html=True)

    t = int(min(max(st.session_state[tkey], 1), n_t))

    if n_t > 1:
        st.markdown(_playhead_html(hours, t), unsafe_allow_html=True)
        event = st.plotly_chart(
            time_chart(hours, values, chart_color), config={"displayModeBar": False},
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


def _init_state(meta, wells):
    ss = st.session_state
    sig = (tuple(wells), meta["rows"], meta["cols"])
    if ss.get("an_sig") != sig:
        ss["an_sig"] = sig
        ss["an_layout"] = pd.DataFrame({"Excluded": False, **{f"f{i}": "" for i in range(MAX_F)}}, index=wells)
        ss["an_nf"] = 2
        ss["an_names"] = ["Cell type", "Treatment"] + [f"Condition {i + 1}" for i in range(2, MAX_F)]
        ss["an_order"], ss["an_colors"], ss["an_ctrl"] = {}, {}, {}
        ss["an_ver"], ss["an_gen"] = 0, 0
        ss["an_snap"] = ss["an_layout"].copy()


def _bump():
    ss = st.session_state
    ss["an_ver"] += 1
    ss["an_gen"] = 0
    ss["an_snap"] = ss["an_layout"].copy()


def _ordered_levels(fi):
    ss = st.session_state
    col = ss["an_layout"][f"f{fi}"]
    present = [v for v in pd.unique(col) if v != ""]
    order = [lv for lv in ss["an_order"].get(fi, []) if lv in present]
    order += [lv for lv in present if lv not in order]
    ss["an_order"][fi] = order
    return order


def _color(fi, level):
    ss = st.session_state
    if (fi, level) not in ss["an_colors"]:
        used = sum(1 for k in ss["an_colors"] if k[0] == fi)
        ss["an_colors"][(fi, level)] = PALETTE[used % len(PALETTE)]
    return ss["an_colors"][(fi, level)]


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


def layout_plate_figure(meta, wells, layout, color_fi, names):
    rows, cols = meta["rows"], meta["cols"]
    cell = min(64.0, 520.0 / cols)
    size = cell * 0.84
    xs, ys, fills, syms, texts, tcols, hov = [], [], [], [], [], [], []
    for w in wells:
        i, j = _well_pos(w, meta)
        r = layout.loc[w]
        lvl, excl = r[f"f{color_fi}"], bool(r["Excluded"])
        fill = "#D1D5DB" if excl else (_color(color_fi, lvl) if lvl else "#F3F4F6")
        xs.append(j + 0.5); ys.append(i + 0.5); fills.append(fill)
        syms.append("circle-x" if excl else "circle")
        texts.append(w if size >= 26 else "")
        tcols.append(_text_on(_rgb(fill)))
        hov.append(f"<b>Well {w}</b>" + "".join(f"<br>{names[k]}: {r[f'f{k}'] or '-'}" for k in range(len(names)))
                   + ("<br>Excluded" if excl else ""))
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textfont=dict(size=10 if cols > 12 else 11, color=tcols),
        marker=dict(size=size, color=fills, symbol=syms, line=dict(color="#6B7280", width=1)),
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.35)),
        hovertext=hov, hoverinfo="text", showlegend=False))
    fig.update_layout(
        height=int(rows * cell + 80), margin=dict(l=26, r=8, t=26, b=6), dragmode="select",
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
    return fig


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
                   range=[-0.5 - max(0, (3 - len(x_levels)) * 0.6), len(x_levels) - 0.5 + max(0, (3 - len(x_levels)) * 0.6)],
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


def _setup_panel(wells, meta, base):
    """Left side: conditions, plate/table editing, levels, save and load."""
    ss = st.session_state
    c_names, c_edit, c_levels, c_io = st.container(), st.container(), st.container(), st.container()

    # --- save and load (processed first so loaded names are in place before the name boxes draw)
    with c_io:
        st.markdown("**Save and load**")
        nf0 = ss["an_nf"]
        st.download_button("Download layout (CSV)", layout_to_csv(ss["an_layout"], _clean_names(ss["an_names"]),
                                                               nf0, wells),
                           f"{base}_layout.csv", "text/csv", key="an_dl_layout")
        st.caption("Also works as a template: download, fill in the columns, and load it back.")
        up = st.file_uploader("Load layout (CSV)", type=["csv"], key="an_up")
        if up is not None:
            data = up.getvalue()
            sig = (up.name, len(data), hash(data))
            if ss.get("an_loaded") != sig:
                ss["an_loaded"] = sig
                layout, names, msgs, ok = parse_layout(data, wells)
                if ok:
                    ss["an_layout"] = layout
                    ss["an_nf"] = max(1, len(names))
                    ss["an_names"] = names + [f"Condition {i + 1}" for i in range(len(names), MAX_F)]
                    for i in range(MAX_F):
                        ss[f"an_name_w{i}"] = ss["an_names"][i]
                    ss["an_order"], ss["an_ctrl"] = {}, {}
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

    # --- condition names
    with c_names:
        st.markdown("**Conditions**")
        nf = ss["an_nf"]
        for i in range(nf):
            v = st.text_input(f"Condition {i + 1} name", value=ss["an_names"][i], key=f"an_name_w{i}",
                              label_visibility="collapsed", placeholder=f"Condition {i + 1}")
            ss["an_names"][i] = v
        a, b, _ = st.columns([1.1, 1.1, 1.4])
        if a.button("Add condition", disabled=nf >= MAX_F, key="an_add"):
            ss["an_nf"] += 1
            st.rerun(scope="fragment")
        if b.button("Remove last", disabled=nf <= 1, key="an_rem"):
            ss["an_layout"][f"f{nf - 1}"] = ""
            ss["an_nf"] -= 1
            _bump()
            st.rerun(scope="fragment")
    names = _clean_names(ss["an_names"])
    nf = ss["an_nf"]

    # --- edit wells
    with c_edit:
        st.markdown("**Assign wells**")
        view = _seg("Edit view", ["Plate", "Table"], "an_view", label_visibility="collapsed")
        layout = ss["an_layout"]
        if view == "Plate":
            color_fi = st.selectbox("Color wells by", list(range(nf)), format_func=lambda i: names[i],
                                    key="an_colorby")
            sel_key = f"an_plate_{ss['an_ver']}_{ss['an_gen']}"
            sel = _selected_wells(ss.get(sel_key), wells)
            st.caption(f"{len(sel)} well(s) selected. Drag a box or lasso on the plate, or click wells.")
            a, b = st.columns([1, 1.4])
            tgt = a.selectbox("Set condition", list(range(nf)), format_func=lambda i: names[i],
                              index=min(color_fi, nf - 1), key="an_target")
            with b:
                level = _level_input(_ordered_levels(tgt), f"an_level_{tgt}")
            r1 = st.columns(4)
            apply_ = r1[0].button("Apply", type="primary", key="an_apply", disabled=not sel or not level)
            excl = r1[1].button("Exclude", key="an_excl", disabled=not sel)
            incl = r1[2].button("Include", key="an_incl", disabled=not sel)
            clr = r1[3].button("Clear level", key="an_clr", disabled=not sel)
            r2 = st.columns(4)
            desel = r2[0].button("Deselect", key="an_desel", disabled=not sel)
            if apply_:
                layout.loc[sel, f"f{tgt}"] = str(level).strip()
            if excl:
                layout.loc[sel, "Excluded"] = True
            if incl:
                layout.loc[sel, "Excluded"] = False
            if clr:
                layout.loc[sel, f"f{tgt}"] = ""
            if apply_ or excl or incl or clr:
                _bump()
                st.rerun(scope="fragment")
            if desel:
                ss["an_gen"] += 1
                st.rerun(scope="fragment")
            st.plotly_chart(layout_plate_figure(meta, wells, layout, color_fi, names), key=sel_key,
                            on_select="rerun", selection_mode=("points", "box", "lasso"),
                            config={"displayModeBar": False})
            lv = _ordered_levels(color_fi)
            chips = "".join(
                f"<span style='display:inline-flex;align-items:center;gap:.35rem;margin:0 .9rem .3rem 0'>"
                f"<span style='width:12px;height:12px;border-radius:50%;background:{_color(color_fi, x)};"
                f"display:inline-block'></span>{x}</span>" for x in lv)
            chips += (f"<span style='display:inline-flex;align-items:center;gap:.35rem;margin:0 .9rem .3rem 0'>"
                      f"<span style='width:12px;height:12px;border-radius:50%;background:#F3F4F6;"
                      f"border:1px solid #9CA3AF;display:inline-block'></span>Unassigned</span>")
            st.markdown(f"<div style='font-size:.85rem'>{chips}</div>", unsafe_allow_html=True)
        else:
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

    # --- levels: order, colors, control
    with c_levels:
        with st.expander("Levels: order, colors and control"):
            for fi in range(nf):
                st.markdown(f"**{names[fi]}**")
                levels = _ordered_levels(fi)
                if not levels:
                    st.caption("No levels yet.")
                    continue
                for k, lv in enumerate(levels):
                    c1, c2, c3, c4 = st.columns([1, 4, 1, 1], vertical_alignment="center")
                    ss["an_colors"][(fi, lv)] = c1.color_picker(
                        lv, value=_color(fi, lv), key=f"an_cp_{fi}_{lv}", label_visibility="collapsed")
                    c2.write(lv)
                    if c3.button("▲", key=f"an_up_{fi}_{lv}", disabled=k == 0):
                        levels[k - 1], levels[k] = levels[k], levels[k - 1]
                        ss["an_order"][fi] = levels
                        st.rerun(scope="fragment")
                    if c4.button("▼", key=f"an_dn_{fi}_{lv}", disabled=k == len(levels) - 1):
                        levels[k + 1], levels[k] = levels[k], levels[k + 1]
                        ss["an_order"][fi] = levels
                        st.rerun(scope="fragment")
                cur = ss["an_ctrl"].get(fi)
                opts = ["None"] + levels
                ctl = st.selectbox("Control level", opts, index=opts.index(cur) if cur in opts else 0,
                                   key=f"an_ctl_{fi}_{hash(tuple(levels))}")
                ss["an_ctrl"][fi] = None if ctl == "None" else ctl
    return names, nf


def _analysis_panel_body(raw, df, meta, base, names, nf, n_t, hours):
    """Right side: choose the analysis, then charts, table and downloads."""
    ss = st.session_state
    layout = ss["an_layout"]
    used = [i for i in range(nf) if (layout[f"f{i}"] != "").any()]
    if not used:
        st.info("Name your conditions and assign wells on the left. Charts appear here as soon as at least "
                "one condition has levels.")
        st.markdown("**How it works**  \n1. Name up to 5 conditions, for example cell type or media.  \n"
                    "2. Drag over wells on the plate, pick or type a level, and press Apply. "
                    "You can also paste into the table view or load a saved layout.  \n"
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
                                    "and shows % of control. Set the control level under Levels.")
    if not ctrl:
        r1[4].caption(f"Set a control level for {names[xi]} under Levels.")
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
    n_t = int(df["T_index"].max())
    hours = df.groupby("T_index")["Elapsed_h"].first().to_numpy()
    hide = ss.get("an_hide", False)
    if hide:
        if st.button("Show setup", key="an_show"):
            ss["an_hide"] = False
            st.rerun(scope="fragment")
        _analysis_panel_body(raw, df, meta, base, _clean_names(ss["an_names"]), ss["an_nf"], n_t, hours)
        return
    left, right = st.columns([1.15, 2], gap="large")
    with left:
        if st.button("Hide setup", key="an_hide_btn", help="Give the charts the full width"):
            ss["an_hide"] = True
            st.rerun(scope="fragment")
        names, nf = _setup_panel(wells, meta, base)
    with right:
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
