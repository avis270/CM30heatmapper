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
    fs = 21 if cols <= 6 else 18 if cols <= 12 else 15

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
    fs = 20 if cols <= 6 else 17 if cols <= 12 else 14
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
                pwid, phei = len(txt) * fs * 0.58 + 18, fs * 1.7
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
    tab_heat, tab_scratch = st.tabs(["Heatmap", "Scratch assay"])

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


if __name__ == "__main__":
    main()
