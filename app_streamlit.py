import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/cloud
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st
from datetime import datetime

# -------------------------
# Plate layouts
# -------------------------
PLATE_LAYOUTS = {
    "6well": (["A", "B"], 3),
    "12well": (["A", "B", "C"], 4),
    "24well": (["A", "B", "C", "D"], 6),
    "96well": (list("ABCDEFGH"), 12),
}

# -------------------------
# Helpers
# -------------------------
def detect_plate_type(lines):
    """Detect plate type from lines of file."""
    for line in lines:
        lower = line.lower()
        if "well" in lower:
            tokens = [t.strip().lower() for t in line.replace(",", "\t").split("\t")]
            for t in tokens:
                if t in PLATE_LAYOUTS:
                    return t
    return None

def normalize_well_name(well_name, plate_type):
    """Normalize CM30 well labels to A1, B2 format."""
    if well_name.startswith("Well"):
        core = well_name[4:]
        if core.isdigit():
            rows, cols = PLATE_LAYOUTS[plate_type]
            num = int(core)
            r = (num - 1) // cols
            c = (num - 1) % cols + 1
            return f"{rows[r]}{c}"
        else:
            return core
    return well_name

def parse_cm30_file(uploaded_file):
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    plate_type = detect_plate_type(lines)
    if plate_type is None:
        raise ValueError("Unsupported or undetected plate type")

    # find start of result section
    start_idx = None
    for i, line in enumerate(lines):
        if "<single result>" in line.lower() or "<colony forming result>" in line.lower():
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No result section found")

    data = {}
    current_well = None
    for line in lines[start_idx + 1:]:
        if line.strip() == "":
            continue
        parts = line.split("\t")
        if parts[0].startswith("Well"):
            current_well = parts[0]
            continue
        if current_well and len(parts) >= 4:
            try:
                t = pd.to_datetime(parts[1], errors="coerce")
                val = float(parts[2])
                if t and not pd.isna(val):
                    data.setdefault(normalize_well_name(current_well, plate_type), []).append((t, val))
            except:
                pass

    records = []
    for well, vals in data.items():
        for t, v in vals:
            records.append({"Well": well, "Time": t, "Confluency": v})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No confluency data parsed")

    # group into aligned timepoint index
    df = df.sort_values("Time")
    df["Timepoint"] = df.groupby("Well").cumcount() + 1
    return df, plate_type

def render_plate(df, plate_type, t_index, min_val, max_val, min_color, max_color):
    rows, cols = PLATE_LAYOUTS[plate_type]
    fig, ax = plt.subplots(figsize=(cols, len(rows)))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, len(rows))
    ax.set_aspect("equal")
    ax.axis("off")

    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [min_color, max_color])

    # timepoint filtering
    subset = df[df["Timepoint"] == t_index]

    for i, r in enumerate(rows):
        for c in range(1, cols + 1):
            well = f"{r}{c}"
            val = subset.loc[subset["Well"] == well, "Confluency"]
            if not val.empty:
                v = val.values[0]
                color = cmap(norm(v))
                label = f"{v:.0f}"
            else:
                color = "black"
                label = "NA"
            circle = plt.Circle((c - 0.5, len(rows) - i - 0.5), 0.4, facecolor=color, edgecolor="black")
            ax.add_patch(circle)
            ax.text(c - 0.5, len(rows) - i - 0.5, label, ha="center", va="center", fontsize=6, color="white")

    ax.set_title(f"Plate: {plate_type}, Timepoint {t_index}")
    return fig

# -------------------------
# Streamlit App
# -------------------------
st.title("📊 CM30 Well Plate Heatmap")

uploaded_file = st.file_uploader("Upload CM30 CSV", type=["csv"])

if uploaded_file:
    try:
        df, plate_type = parse_cm30_file(uploaded_file)
        timepoints = sorted(df["Timepoint"].unique())
        times = df.groupby("Timepoint")["Time"].min().sort_index()

        # Sidebar controls
        st.sidebar.subheader("Confluency Range & Colors")
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            min_val = st.number_input("Min %", value=0.0, step=1.0)
        with col2:
            min_color = st.color_picker("Color", "#ffffff")

        col3, col4 = st.sidebar.columns([2, 1])
        with col3:
            max_val = st.number_input("Max %", value=100.0, step=1.0)
        with col4:
            max_color = st.color_picker("Color", "#8B0000")

        # Timepoint slider
        t_index = st.slider(
            "Select timepoint",
            min_value=1,
            max_value=len(timepoints),
            value=1,
            step=1,
            format="Timepoint %d",
        )

        fig = render_plate(df, plate_type, t_index, min_val, max_val, min_color, max_color)
        st.pyplot(fig, dpi=220)

        # Download buttons
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
        st.download_button("Download current timepoint as PNG", buf.getvalue(), file_name=f"plate_time{t_index}.png", mime="image/png")

        all_buf = io.BytesIO()
        with zipfile.ZipFile(all_buf, "w") as zf:
            for tp in timepoints:
                fig_tp = render_plate(df, plate_type, tp, min_val, max_val, min_color, max_color)
                tmp = io.BytesIO()
                fig_tp.savefig(tmp, format="png", dpi=220, bbox_inches="tight")
                zf.writestr(f"plate_time{tp}.png", tmp.getvalue())
        st.download_button("Download all timepoints as ZIP", all_buf.getvalue(), file_name="all_timepoints.zip", mime="application/zip")

    except Exception as e:
        st.error(f"Could not parse file: {e}")

