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
TRAY = "#F5F6F8"
TRAY_EDGE = "#E3E6EB"
GAP_FILL = "#FFFFFF"
EMPTY_RING = "#C9CED6"
GAP_MAX = 0.6

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


@st.cache_data(show_spinner=False)
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


# ---------------------------------------------------------------- scratch maths
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


# ---------------------------------------------------------------- figures
def _rounded_rect(x0, y0, x1, y1, r):
    return (
        f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} L {x1},{y1 - r} "
        f"Q {x1},{y1} {x1 - r},{y1} L {x0 + r},{y1} Q {x0},{y1} {x0},{y1 - r} "
        f"L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z"
    )


def _gap_path(cx, cy, r, half, n=18):
    """Horizontal band of half-thickness `half`, centred on the well and clipped to the circle."""
    th = np.arcsin(min(half / r, 1.0))
    right = np.linspace(-th, th, n)
    left = np.linspace(np.pi - th, np.pi + th, n)
    pts = [(cx + r * np.cos(p), cy + r * np.sin(p)) for p in right]
    pts += [(cx + r * np.cos(p), cy + r * np.sin(p)) for p in left]
    return "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in pts) + " Z"


def _plate_canvas(rows, cols, numbered):
    cell = min(130.0, 1000.0 / cols)
    left = 0.1 if numbered else 0.45
    fig = go.Figure()
    fig.update_layout(
        height=int(rows * cell + 120),
        margin=dict(l=8, r=8, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=13),
        showlegend=False,
        dragmode=False,
        hoverlabel=dict(bgcolor="white", bordercolor=TRAY_EDGE, font=dict(family=FONT, color=INK, size=13)),
        xaxis=dict(
            range=[-left, cols + 0.1], side="top", showgrid=False, zeroline=False, showline=False,
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
    fig.add_shape(
        type="path", path=_rounded_rect(0, 0, cols, rows, 0.28), layer="below",
        fillcolor=TRAY, line=dict(color=TRAY_EDGE, width=1),
    )
    return fig, cell


def _colorbar_trace(vmin, vmax, c0, c1, title):
    return go.Scatter(
        x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(
            colorscale=[[0, c0], [1, c1]], cmin=vmin, cmax=vmax, color=[vmin], showscale=True,
            colorbar=dict(title=dict(text=title, side="right"), thickness=12, len=0.75,
                          outlinewidth=0, ticks="", tickfont=dict(size=12)),
        ),
    )


def _number_labels(fig, meta, r):
    if not meta["numbered"]:
        return
    for i in range(meta["rows"]):
        for j in range(meta["cols"]):
            n = i * meta["cols"] + j + 1
            fig.add_annotation(x=j + 0.5 - r * 0.78, y=i + 0.5 - r * 0.78, text=str(n), showarrow=False,
                               font=dict(size=11, color=MUTED))


def _hover_layer(fig, xs, ys, size, custom, template, text=None, text_colors=None, fs=13):
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text" if text else "markers",
        marker=dict(size=size, color="rgba(0,0,0,0)"),
        text=text, textfont=dict(size=fs, family=FONT, color=text_colors) if text else None,
        textposition="middle center",
        customdata=custom, hovertemplate=template + "<extra></extra>",
    ))


def heatmap_figure(sl, meta, vmin, vmax, c0, c1):
    rows, cols = meta["rows"], meta["cols"]
    fig, cell = _plate_canvas(rows, cols, meta["numbered"])
    r = 0.42
    vals = dict(zip(sl["Well"], sl["Confluency"]))
    fmt = ".1f" if cols <= 6 else ".0f"
    fs = int(max(9, min(16, cell * 0.17)))
    xs, ys, custom, text, tcols = [], [], [], [], []

    for i in range(rows):
        for j in range(cols):
            well = (str(i * cols + j + 1) if meta["numbered"] else f"{chr(65 + i)}{j + 1}")
            cx, cy = j + 0.5, i + 0.5
            v = vals.get(well)
            if v is None or pd.isna(v):
                fig.add_shape(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                              line=dict(color=EMPTY_RING, width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")
                continue
            t = float(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1))
            rgb = lerp_color(c0, c1, t)
            fig.add_shape(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                          fillcolor=_css(rgb), line=dict(color="rgba(17,24,39,0.22)", width=1))
            xs.append(cx); ys.append(cy); custom.append([well, v])
            text.append(f"{v:{fmt}}%"); tcols.append(_text_on(rgb))

    _number_labels(fig, meta, r)
    _hover_layer(fig, xs, ys, cell * 0.8, custom, "<b>Well %{customdata[0]}</b><br>Confluency %{customdata[1]:.1f}%",
                 text, tcols, fs)
    fig.add_trace(_colorbar_trace(vmin, vmax, c0, c1, "% confluency"))
    fig.update_shapes(layer="below")  # keep hover and labels above the well shapes
    return fig


def scratch_figure(sl, meta, c0, c1):
    rows, cols = meta["rows"], meta["cols"]
    fig, cell = _plate_canvas(rows, cols, meta["numbered"])
    r = 0.42
    by_well = sl.set_index("Well")
    fmt = ".1f" if cols <= 6 else ".0f"
    fs = int(max(9, min(15, cell * 0.16)))
    xs, ys, custom = [], [], []

    for i in range(rows):
        for j in range(cols):
            well = (str(i * cols + j + 1) if meta["numbered"] else f"{chr(65 + i)}{j + 1}")
            cx, cy = j + 0.5, i + 0.5
            if well not in by_well.index:
                fig.add_shape(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                              line=dict(color=EMPTY_RING, width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")
                continue
            row = by_well.loc[well]
            cl = row["Closure_pct"]
            xs.append(cx); ys.append(cy)
            custom.append([well, cl if pd.notna(cl) else np.nan, row["Confluency"], row["Confluency_T0"]])
            if pd.isna(cl):
                fig.add_shape(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
                              line=dict(color=EMPTY_RING, width=1), fillcolor="#ECEEF1")
                continue
            t = float(np.clip(cl / 100, 0, 1))
            rgb = lerp_color(c0, c1, t)
            circle = dict(x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r)
            fig.add_shape(type="circle", fillcolor=_css(rgb), line=dict(width=0), **circle)
            # gap thickness is proportional to the remaining open fraction; 0% closure draws a band
            # GAP_MAX of the well diameter so cells stay visible on both sides
            half = (1 - t) * r * GAP_MAX
            if half > 0.004:
                fig.add_shape(type="path", path=_gap_path(cx, cy, r, half), fillcolor=GAP_FILL,
                              line=dict(color="rgba(17,24,39,0.15)", width=0.8))
            fig.add_shape(type="circle", fillcolor="rgba(0,0,0,0)", line=dict(color="rgba(17,24,39,0.28)", width=1),
                          **circle)
            fig.add_annotation(x=cx, y=cy, text=f"{cl:{fmt}}%", showarrow=False,
                               font=dict(size=fs, color=INK, family=FONT),
                               bgcolor="rgba(255,255,255,0.88)", borderpad=2)

    _number_labels(fig, meta, r)
    _hover_layer(
        fig, xs, ys, cell * 0.8, custom,
        "<b>Well %{customdata[0]}</b><br>Closure %{customdata[1]:.1f}%"
        "<br>Confluency %{customdata[2]:.1f}% (start %{customdata[3]:.1f}%)",
    )
    fig.add_trace(_colorbar_trace(0, 100, c0, c1, "% closure"))
    fig.update_shapes(layer="below")
    return fig


PLOT_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                               "autoScale2d", "resetScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3, "filename": "plate"},
}


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

    df, meta = parse_cm30(up.getvalue())
    if df is None:
        st.error("No well data found. Upload the analysis file whose name contains EV.")
        return

    base = Path(up.name).stem
    n_t = int(df["T_index"].max())
    t = st.slider("Select timepoint", 1, n_t, 1, format="Timepoint %d") if n_t > 1 else 1
    sl_all = df[df["T_index"] == t]
    ref = sl_all["ReferenceTime"].iloc[0]
    elapsed = sl_all["Elapsed_h"].iloc[0]
    st.caption(f"{meta['project']}: {meta['vessel']}, {ref:%Y-%m-%d %H:%M}, {elapsed:.1f} h elapsed")

    tab_heat, tab_scratch = st.tabs(["Heatmap", "Scratch assay"])

    with tab_heat:
        c1_, c2_, c3_, c4_, _ = st.columns([1, 1, 1, 1, 3])
        vmin = c1_.number_input("Min %", value=0.0, step=5.0, key="h_min")
        col0 = c2_.color_picker("Color at min", "#FFFFFF", key="h_c0")
        vmax = c3_.number_input("Max %", value=100.0, step=5.0, key="h_max")
        col1 = c4_.color_picker("Color at max", "#8B0000", key="h_c1")
        if vmax <= vmin:
            st.warning("Max must be greater than min.")
        else:
            st.plotly_chart(heatmap_figure(sl_all, meta, vmin, vmax, col0, col1), config=PLOT_CONFIG,
                            key="heat_plot")
        tidy = df.sort_values(["RowIdx", "ColIdx", "T_index"]).copy()
        tidy["Time"] = tidy["ReferenceTime"].dt.strftime("%Y-%m-%d %H:%M")
        tidy = tidy.rename(columns={"T_index": "Timepoint"})[
            ["Well", "Row", "Column", "Timepoint", "Time", "Elapsed_h", "Confluency"]]
        st.download_button("Download confluency data (CSV)", tidy.to_csv(index=False).encode(),
                           f"{base}_confluency.csv", "text/csv", key="dl_conf")

    with tab_scratch:
        s1, s2, s3, s4, _ = st.columns([1.3, 1.3, 1, 1, 2])
        full = s1.number_input("Fully closed at (% confluency)", min_value=1.0, max_value=100.0,
                               value=95.0, step=1.0, key="s_full")
        cap = s2.checkbox("Cap closure at 0 to 100%", value=True, key="s_cap")
        sc0 = s3.color_picker("Color at 0%", "#A7D8D0", key="s_c0")
        sc1 = s4.color_picker("Color at 100%", "#0F766E", key="s_c1")

        cdf = add_closure(df, full, cap)
        sl = cdf[cdf["T_index"] == t]
        bad = sl["Closure_pct"].isna().sum()
        if bad:
            st.warning(f"{bad} well(s) start at or above {full:g}% confluency, so closure can't be calculated "
                       "for them. Raise the fully closed value if that is unexpected.")
        st.plotly_chart(scratch_figure(sl, meta, sc0, sc1), config=PLOT_CONFIG, key="scratch_plot")
        st.caption("Each well is colored by % closure. The white band shows the remaining gap, "
                   "thinning as the scratch closes. Gap size is schematic. Time zero is the first timepoint.")

        long_df = export_long(cdf)
        wide_df = export_wide(cdf, meta["well_order"])
        d1, d2, _ = st.columns([1.4, 1.4, 4])
        d1.download_button("Download closure, long (CSV)", long_df.to_csv(index=False).encode(),
                           f"{base}_closure_long.csv", "text/csv", key="dl_long")
        d2.download_button("Download closure, wide (CSV)", wide_df.to_csv(index=False).encode(),
                           f"{base}_closure_wide.csv", "text/csv", key="dl_wide")
        with st.expander("Preview export"):
            st.dataframe(long_df.head(200), hide_index=True)


if __name__ == "__main__":
    main()
