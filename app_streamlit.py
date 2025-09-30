# app_streamlit.py
import io, re, time, zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

import streamlit as st

# ------------------------
# Plate layouts
# ------------------------
PLATE_LAYOUTS: Dict[str, Tuple[List[str], List[int]]] = {
    "6well":   (["A", "B"], list(range(1, 4))),           # 2 x 3
    "12well":  (["A", "B", "C"], list(range(1, 5))),      # 3 x 4
    "24well":  (["A", "B", "C", "D"], list(range(1, 7))), # 4 x 6
    "96well":  (list("ABCDEFGH"), list(range(1, 13))),    # 8 x 12
}

# Map 6-well "Well1..Well6" to A/B grid positions
SIX_WELL_MAP = {
    "Well1": "WellA1",
    "Well2": "WellA2",
    "Well3": "WellA3",
    "Well4": "WellB1",
    "Well5": "WellB2",
    "Well6": "WellB3",
}

# ------------------------
# Utilities
# ------------------------
def detect_plate_type(lines: List[str]) -> Optional[str]:
    """Detect plate type from lines (e.g., in <vessel Type> section)."""
    for line in lines:
        lower = line.lower()
        if "well" in lower:
            # token-scan for '6well', '12well', etc.
            tokens = re.split(r"[\t, ]+", line.strip())
            for t in tokens:
                t_clean = t.strip().lower()
                if t_clean in PLATE_LAYOUTS:
                    return t_clean
    return None

def try_parse_time(s: str) -> Optional[datetime]:
    """Robust timestamp parsing: supports 'YYYY/MM/DD HH:MM', 'MM/DD/YYYY HH:MM[:SS]'."""
    s = s.strip()
    fmts = ["%Y/%m/%d %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def parse_file(file) -> Tuple[str, Dict[str, List[Tuple[datetime, float]]], List[datetime]]:
    """Parse CSV/TXT from CM system. Returns (plate_type, data, sorted_times)."""
    text = file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    plate_type = detect_plate_type(lines)
    if not plate_type:
        raise ValueError("Unsupported or undetected plate type in file.")

    data: Dict[str, List[Tuple[datetime, float]]] = {}
    times: set = set()
    current_well: Optional[str] = None
    in_results = False

    # Results sections may be either:
    #   <Single Result>  (96/24/12 style with WellA1, WellB2, ...)
    #   <Colony Forming Result> (6-well style with Well1..Well6)
    for raw in lines:
        line = raw.strip()
        if line.startswith("<Single Result>") or line.startswith("<Colony Forming Result>"):
            in_results = True
            continue

        if not in_results or not line:
            continue

        if line.startswith("Well"):  # new well block header
            # grab the first token; handles "WellA1" or "Well1"
            current_well = line.split()[0]
            if plate_type == "6well" and current_well in SIX_WELL_MAP:
                current_well = SIX_WELL_MAP[current_well]  # normalize to WellA1..WellB3
            data.setdefault(current_well, [])
            continue

        # Skip header row inside block
        if current_well and not line.startswith("Passage#"):
            # Split by comma or tab
            parts = re.split(r"[\t,]", line)
            # Expected: Passage#, Time, Estimatevalue(Confluency), Estimatevalue(Count)
            if len(parts) >= 3:
                t = try_parse_time(parts[1])
                # Confluency at col 2
                try:
                    conf = float(parts[2])
                except Exception:
                    conf = None

                if t is not None and conf is not None:
                    data[current_well].append((t, conf))
                    times.add(t)

    times_sorted = sorted(list(times))
    return plate_type, data, times_sorted

def get_value_for_time(well_series: List[Tuple[datetime, float]], timepoint: datetime) -> Optional[float]:
    """Pick the closest-in-time value for a well."""
    if not well_series:
        return None
    return min(well_series, key=lambda tc: abs((tc[0] - timepoint).total_seconds()))[1]

def make_two_color_cmap(color_min: str, color_max: str) -> mcolors.LinearSegmentedColormap:
    """Linear colormap between two hex colors."""
    return mcolors.LinearSegmentedColormap.from_list("custom_two", [color_min, color_max], N=256)

# ------------------------
# Plotting (digital plate with circles)
# ------------------------
def render_plate_circles(
    plate_type: str,
    data: Dict[str, List[Tuple[datetime, float]]],
    timepoint: datetime,
    vmin: float,
    vmax: float,
    color_min: str,
    color_max: str,
    show_labels: bool = True,
) -> plt.Figure:
    rows, cols = PLATE_LAYOUTS[plate_type]
    n_rows, n_cols = len(rows), len(cols)

    # sizing: a bit of padding for labels; keep circles round
    cell_size = 1.2  # visual scale
    fig_w = n_cols * cell_size + 1.5
    fig_h = n_rows * cell_size + 1.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")

    # Create colormap
    cmap = make_two_color_cmap(color_min, color_max)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # grid with margins for labels
    x0, y0 = 1.0, 1.0  # margins for row/col labels
    radius = cell_size * 0.42  # circle radius per cell

    # Draw circles
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            cx = x0 + j * cell_size
            cy = y0 + i * cell_size

            well_id = f"Well{row}{col}"
            val = None
            if well_id in data:
                val = get_value_for_time(data[well_id], timepoint)

            # Color mapping; missing/negative -> black fill
            if val is None or val < 0:
                face = "#000000"  # black for missing/negative
                txt = "NA"
                txt_color = "white"
            else:
                # clamp into [vmin, vmax]
                v_clamped = max(vmin, min(vmax, val))
                face = cmap(norm(v_clamped))
                txt = f"{val:.1f}"
                # text color based on luminance
                r, g, b, _ = mcolors.to_rgba(face)
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = "black" if luminance > 0.6 else "white"

            circle = patches.Circle((cx, cy), radius=radius, facecolor=face, edgecolor="black", linewidth=0.8)
            ax.add_patch(circle)

            if show_labels:
                ax.text(cx, cy, txt, ha="center", va="center", fontsize=10, color=txt_color)

    # Row/Col headers aligned to circle centers
    for i, row in enumerate(rows):
        cy = y0 + i * cell_size
        ax.text(x0 - cell_size * 0.6, cy, row, ha="center", va="center", fontsize=12)

    for j, col in enumerate(cols):
        cx = x0 + j * cell_size
        ax.text(cx, y0 - cell_size * 0.6, str(col), ha="center", va="center", fontsize=12)

    # Axes clean-up
    ax.set_xlim(x0 - cell_size, x0 + (n_cols - 1) * cell_size + cell_size)
    ax.set_ylim(y0 - cell_size, y0 + (n_rows - 1) * cell_size + cell_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()  # A at top-left

    # Legend bar (min→max)
    # Draw a simple two-color bar at bottom-right
    from matplotlib.colors import to_hex
    grad_ax = fig.add_axes([0.82, 0.12, 0.04, 0.3])  # [left, bottom, width, height]
    gradient = [[i / 255.0] for i in range(256)]
    grad_ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[0, 1, vmin, vmax],
    )
    grad_ax.set_xlabel("")
    grad_ax.set_ylabel("%")
    for spine in grad_ax.spines.values():
        spine.set_visible(False)
    grad_ax.tick_params(labelsize=8)

    return fig

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Well Plate Heatmapper", layout="wide")
st.title("Well Plate Heatmapper — Circles, Custom Colors, Playback")

# Session state for playback
if "playing" not in st.session_state:
    st.session_state.playing = False
if "t_index" not in st.session_state:
    st.session_state.t_index = 0

with st.sidebar:
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload plate file (.csv or .txt)", type=["csv", "txt"])

    st.markdown("**Confluency scale**")
    vmin = st.number_input("Min %", value=0.0, step=1.0)
    vmax = st.number_input("Max %", value=100.0, step=1.0)

    st.markdown("**Colors**")
    color_min = st.color_picker("Color at Min %", "#FFFFFF")   # white
    color_max = st.color_picker("Color at Max %", "#8B0000")   # dark red

    st.markdown("---")
    show_numbers = st.checkbox("Show numbers in wells", value=True)

if not uploaded:
    st.info("Upload a CSV/TXT exported from your imaging system to begin.")
    st.stop()

# Parse
try:
    plate_type, data, times = parse_file(uploaded)
except Exception as e:
    st.error(f"Could not parse file: {e}")
    st.stop()

if not times:
    st.error("No timepoints found in file.")
    st.stop()

# Build time index labels (show first well's time per index for context)
# Use the lexicographically first well with data as reference
wells_with_data = [w for w, series in data.items() if series]
ref_well = sorted(wells_with_data)[0] if wells_with_data else None
time_labels = []
for t in times:
    label = t.strftime("%Y-%m-%d %H:%M")
    time_labels.append(label)

# Controls row
col1, col2, col3, col4 = st.columns([2, 5, 1.2, 1.8])

with col1:
    st.markdown(f"**Detected plate:** `{plate_type}`")

with col2:
    # Slider for timepoint index with label showing the actual time
    st.session_state.t_index = st.slider(
        "Timepoint",
        0,
        len(times) - 1,
        st.session_state.t_index,
        key="time_slider",
        help="Move between timepoints",
    )
    st.caption(f"Selected time: {time_labels[st.session_state.t_index]}")

with col3:
    # Play/Pause toggle
    if st.session_state.playing:
        if st.button("⏸ Pause"):
            st.session_state.playing = False
    else:
        if st.button("▶ Play"):
            st.session_state.playing = True

with col4:
    dpi = 220
    st.caption(f"Export DPI: {dpi}")

# Render current frame
current_time = times[st.session_state.t_index]
fig = render_plate_circles(
    plate_type=plate_type,
    data=data,
    timepoint=current_time,
    vmin=vmin,
    vmax=vmax,
    color_min=color_min,
    color_max=color_max,
    show_labels=show_numbers,
)
st.pyplot(fig, use_container_width=True)

# Downloads
dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    st.download_button(
        "Download current timepoint (PNG)",
        data=buf.getvalue(),
        file_name=f"heatmap_{plate_type}_{time_labels[st.session_state.t_index].replace(' ','_').replace(':','')}.png",
        mime="image/png",
    )

with dl_col2:
    if st.button("Download ALL timepoints (ZIP)"):
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for idx, t in enumerate(times):
                fig_i = render_plate_circles(
                    plate_type=plate_type,
                    data=data,
                    timepoint=t,
                    vmin=vmin,
                    vmax=vmax,
                    color_min=color_min,
                    color_max=color_max,
                    show_labels=show_numbers,
                )
                png_bytes = io.BytesIO()
                fig_i.savefig(png_bytes, format="png", dpi=dpi, bbox_inches="tight")
                plt.close(fig_i)
                fname = f"heatmap_{plate_type}_{t.strftime('%Y%m%d_%H%M')}.png"
                zf.writestr(fname, png_bytes.getvalue())
        st.download_button(
            "Save ZIP",
            data=zbuf.getvalue(),
            file_name=f"heatmaps_{plate_type}.zip",
            mime="application/zip",
        )

# Playback loop (auto-advance)
if st.session_state.playing and len(times) > 1:
    time.sleep(1.0)  # ~1 fps
    st.session_state.t_index = (st.session_state.t_index + 1) % len(times)
    st.experimental_rerun()
