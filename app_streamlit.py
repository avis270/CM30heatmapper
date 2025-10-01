import io, zipfile
from datetime import datetime
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless/cloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import streamlit as st

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
def split_tokens(line: str):
    """Split a line by comma OR tab and trim blanks."""
    return [t.strip() for t in line.replace(",", "\t").split("\t") if t.strip() != ""]

def detect_plate_type(lines):
    """Detect plate type from any line that contains 'well' token."""
    for line in lines:
        low = line.lower()
        if "well" in low:
            for t in split_tokens(line):
                tl = t.lower()
                if tl in PLATE_LAYOUTS:
                    return tl
    return None

def normalize_well_name(well_name: str, plate_type: str) -> str:
    """
    CM30 uses 'Well1'..'Well6' for 6-well, and 'WellA1' style for others.
    Normalize to 'A1' etc.
    """
    name = well_name.strip()
    if not name.startswith("Well"):
        return name

    core = name[4:]  # after 'Well'
    if core and core[0].isalpha():
        return core.upper()

    if core.isdigit():
        rows, ncols = PLATE_LAYOUTS[plate_type]
        idx = int(core)
        r = (idx - 1) // ncols
        c = (idx - 1) % ncols + 1
        if 0 <= r < len(rows):
            return f"{rows[r]}{c}"
    return core.upper()

def parse_cm30_file(uploaded_file):
    """Parse CM30 CSV/TSV text into dataframe + plate type + project name."""
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    project_name = "Unknown Project"
    for i, line in enumerate(lines[:10]):
        toks = split_tokens(line)
        if toks and toks[0].lower() == "name" and len(lines) > i+1:
            project_name = split_tokens(lines[i+1])[0]
            break

    plate_type = detect_plate_type(lines)
    if plate_type is None:
        raise ValueError("Unsupported or undetected plate type")

    start_idx = None
    for i, line in enumerate(lines):
        l = line.lower()
        if "<single result>" in l or "<colony forming result>" in l:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No result section found")

    data = {}
    current_well = None

    for raw in lines[start_idx + 1:]:
        if not raw.strip():
            continue
        toks = split_tokens(raw)
        if not toks:
            continue

        if toks[0].startswith("Well"):
            current_well = normalize_well_name(toks[0], plate_type)
            continue

        if current_well and len(toks) >= 3:
            t = pd.to_datetime(toks[1], errors="coerce")
            v = pd.to_numeric(toks[2], errors="coerce")
            if pd.notna(t) and pd.notna(v):
                data.setdefault(current_well, []).append((t, v))

    records = []
    for well, vals in data.items():
        vals = sorted(vals, key=lambda x: x[0])
        for t, v in vals:
            records.append({"Well": well, "Time": t, "Confluency": v})

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No confluency data parsed")

    df = df.sort_values(["Well", "Time"])
    df["Timepoint"] = df.groupby("Well").cumcount() + 1

    return df, plate_type, project_name

def render_plate(df, plate_type, t_index, min_val, max_val, min_color, max_color, project_name, tp_time):
    """Draw circular wells + black bounding box. Add row/col labels for 96well. Custom title."""
    rows, ncols = PLATE_LAYOUTS[plate_type]
    nrows = len(rows)

    fig, ax = plt.subplots(figsize=(ncols*0.6, nrows*0.6))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.axis("off")

    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [min_color, max_color])

    sub = df[df["Timepoint"] == t_index]

    for ri, r in enumerate(rows):
        for c in range(1, ncols + 1):
            well = f"{r}{c}"
            val_ser = sub.loc[sub["Well"] == well, "Confluency"]

            if not val_ser.empty:
                v = float(val_ser.iloc[0])
                color = cmap(norm(v))
                label = f"{v:.0f}"
            else:
                color = "black"
                label = "NA"

            cx, cy = (c - 0.5, nrows - ri - 0.5)
            circ = plt.Circle((cx, cy), 0.42, facecolor=color, edgecolor="black", linewidth=0.6)
            ax.add_patch(circ)
            ax.text(cx, cy, label, ha="center", va="center", fontsize=6, color="white")

    rect = plt.Rectangle((0, 0), ncols, nrows, linewidth=1.2, edgecolor="black", facecolor="none")
    ax.add_patch(rect)

    if plate_type == "96well":
        for c in range(1, ncols + 1):
            ax.text(c - 0.5, nrows + 0.4, str(c), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="black")
        for ri, r in enumerate(rows):
            ax.text(-0.9, nrows - ri - 0.5, r, ha="center", va="center",
                    fontsize=14, fontweight="bold", color="black")

    tp_str = tp_time.strftime("%Y-%m-%d %H:%M") if pd.notna(tp_time) else "Unknown"
    fig.suptitle(f"{project_name} — Timepoint {t_index} ({tp_str})", fontsize=12, y=1.05)

    return fig

# -------------------------
# Streamlit App
# -------------------------
st.title("📊 CM30 Well Plate Heatmap")

st.markdown(
    """
    **Step 1:** Export and download your analysis .csv from the CM30  
    **Step 2:** Extract the files  
    **Step 3:** Upload the file that contains the averages per well.  The filename will contain EV
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    try:
        df, plate_type, project_name = parse_cm30_file(uploaded_file)

        tpoints = sorted(df["Timepoint"].unique())
        times_by_tp = (
            df.sort_values("Time")
              .groupby("Timepoint")["Time"]
              .min()
              .reindex(tpoints)
        )

        st.sidebar.subheader("Confluency Range & Colors")

        col1, col2 = st.sidebar.columns([2, 1], gap="small")
        with col1:
            min_val = st.number_input("Min %", value=0.0, step=1.0)
        with col2:
            min_color = st.color_picker("Color at Min %", "#ffffff")

        col3, col4 = st.sidebar.columns([2, 1], gap="small")
        with col3:
            max_val = st.number_input("Max %", value=100.0, step=1.0)
        with col4:
            max_color = st.color_picker("Color at Max %", "#8B0000")

        t_index = st.slider(
            "Select timepoint",
            min_value=int(min(tpoints)),
            max_value=int(max(tpoints)),
            value=int(min(tpoints)),
            step=1,
            format="Timepoint %d",
        )

        tp_time = times_by_tp.loc[t_index]

        fig = render_plate(df, plate_type, t_index, min_val, max_val, min_color, max_color,
                           project_name=project_name, tp_time=tp_time)
        st.pyplot(fig, dpi=220)

        # ⬇️ Move download buttons to sidebar
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
        st.sidebar.download_button(
            "Download current timepoint as PNG",
            buf.getvalue(),
            file_name=f"{plate_type}_timepoint_{t_index}.png",
            mime="image/png",
        )

        if st.sidebar.button("Build ZIP of all timepoints"):
            all_buf = io.BytesIO()
            with zipfile.ZipFile(all_buf, "w") as zf:
                for tp in tpoints:
                    fig_tp = render_plate(df, plate_type, tp, min_val, max_val, min_color, max_color,
                                          project_name=project_name, tp_time=times_by_tp.loc[tp])
                    tmp = io.BytesIO()
                    fig_tp.savefig(tmp, format="png", dpi=220, bbox_inches="tight")
                    zf.writestr(f"{plate_type}_timepoint_{tp}.png", tmp.getvalue())
                    plt.close(fig_tp)
            st.sidebar.download_button(
                "Download all timepoints (ZIP)",
                all_buf.getvalue(),
                file_name=f"{plate_type}_all_timepoints.zip",
                mime="application/zip",
            )

    except Exception as e:
        st.error(f"Could not parse file: {e}")
