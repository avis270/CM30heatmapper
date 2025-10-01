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
    # Case 1: already letter+number (e.g., A1, B12)
    if core and core[0].isalpha():
        return core.upper()

    # Case 2: pure digits (e.g., '1'..'6') -> map into grid
    if core.isdigit():
        rows, ncols = PLATE_LAYOUTS[plate_type]
        idx = int(core)
        r = (idx - 1) // ncols
        c = (idx - 1) % ncols + 1
        if 0 <= r < len(rows):
            return f"{rows[r]}{c}"
    return core.upper()

def parse_cm30_file(uploaded_file):
    """Parse CM30 CSV/TSV text into tidy dataframe with Well, Time, Confluency, Timepoint."""
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    # 1) Detect plate type
    plate_type = detect_plate_type(lines)
    if plate_type is None:
        raise ValueError("Unsupported or undetected plate type")

    # ✅ Project name: line after "Name"
    project_name = "Unknown Project"
    for i, line in enumerate(lines[:20]):  # look near the top
        toks = split_tokens(line)
        if toks and toks[0].lower() == "name" and len(lines) > i+1:
            project_name = split_tokens(lines[i+1])[0]
            break

    # 3) Find start of result section
    start_idx = None
    for i, line in enumerate(lines):
        l = line.lower()
        if "<single result>" in l or "<colony forming result>" in l:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No result section found")

    # 4) Walk lines, collect rows
    data = {}
    current_well = None

    for raw in lines[start_idx + 1:]:
        if not raw.strip():
            continue
        toks = split_tokens(raw)
        if not toks:
            continue

        # new well header
        if toks[0].startswith("Well"):
            current_well = normalize_well_name(toks[0], plate_type)
            continue

        # data row
        if current_well and len(toks) >= 3:
            t = pd.to_datetime(toks[1], errors="coerce")
            v = pd.to_numeric(toks[2], errors="coerce")
            if pd.notna(t) and pd.notna(v):
                data.setdefault(current_well, []).append((t, v))

    # 5) Build dataframe
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

def render_plate(df, plate_type, t_index, min_val, max_val, min_color, max_color,
                 project_name=None, tp_time=None, scale=1.0):
    """Draw circular wells with values (or NA) for a given timepoint index."""
    rows, ncols = PLATE_LAYOUTS[plate_type]
    nrows = len(rows)

    fig, ax = plt.subplots(figsize=(ncols * scale, nrows * scale))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # colormap
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [min_color, max_color])

    sub = df[df["Timepoint"] == t_index]

    # font size scaling by plate type
    if plate_type == "6well":
        font_size = 14
    elif plate_type == "12well":
        font_size = 11
    elif plate_type == "24well":
        font_size = 9
    else:  # 96well
        font_size = 7

    # draw wells
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
            ax.text(cx, cy, label, ha="center", va="center", fontsize=font_size, color="white")

    # black rectangle
    ax.add_patch(plt.Rectangle((0, 0), ncols, nrows, fill=False, edgecolor="black", linewidth=1.2))

    # labels for 96well
    if plate_type == "96well":
        for ri, r in enumerate(rows):
            cy = nrows - ri - 0.5
            ax.text(-0.6, cy, r, ha="right", va="center", fontsize=12, fontweight="bold")
        for c in range(1, ncols + 1):
            cx = (c - 0.5)
            ax.text(cx, nrows + 0.3, str(c), ha="center", va="bottom", fontsize=12, fontweight="bold")

    # title
    title = ""
    if project_name:
        title += f"{project_name} – "
    title += f"Timepoint {t_index}"
    if tp_time is not None and pd.notna(tp_time):
        title += f" ({tp_time})"

    # Adjust padding: 96well needs extra room for column labels, others don't
    if plate_type == "96well":
        ax.set_title(title, fontsize=14, pad=25)
    else:
        ax.set_title(title, fontsize=14, pad=1)

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

        # Available timepoints
        tpoints = sorted(df["Timepoint"].unique())
        times_by_tp = (
            df.sort_values("Time")
              .groupby("Timepoint")["Time"]
              .min()
              .reindex(tpoints)
        )

        # Sidebar controls
        st.sidebar.markdown("<br><br>", unsafe_allow_html=True)  # space
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

        # Timepoint slider
        t_index = st.slider(
            "Select timepoint",
            min_value=int(min(tpoints)),
            max_value=int(max(tpoints)),
            value=int(min(tpoints)),
            step=1,
            format="Timepoint %d",
        )

        tp_time = times_by_tp.loc[t_index]

        # scaling
        if plate_type == "6well":
            scale = 2.0
        elif plate_type == "12well":
            scale = 1.5
        else:
            scale = 1.0

        # Render
        fig = render_plate(df, plate_type, t_index, min_val, max_val, min_color, max_color,
                           project_name=project_name, tp_time=tp_time, scale=scale)
        st.pyplot(fig, dpi=220)

        # Sidebar download buttons
        st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
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
                    fig_tp = render_plate(df, plate_type, tp, min_val, max_val,
                                          min_color, max_color, project_name, times_by_tp.loc[tp], scale)
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



