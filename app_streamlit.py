import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# ---------------- Plate definitions ----------------
plate_dims = {
    "6well":  (2, 3),    # rows, cols
    "12well": (3, 4),
    "24well": (4, 6),
    "96well": (8, 12),
}

# ---------------- Parsing functions ----------------
def detect_plate_type(raw: str) -> str:
    """Detect plate type by parsing the <vessel Type> section and extracting VesselName."""
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("<vessel Type>"):
            if i + 2 < len(lines):
                header = lines[i+1].split(",")
                values = lines[i+2].split(",")
                if "VesselName" in header:
                    idx = header.index("VesselName")
                    if idx < len(values):
                        return values[idx].strip().lower()
    return "96well"  # fallback

def parse_file(uploaded) -> pd.DataFrame:
    """Parse uploaded CSV into tidy DataFrame: Well, Time, Confluency, PlateType"""
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")

    # Detect plate type
    plate_type = detect_plate_type(raw)

    # Which section to search
    if "96well" in plate_type:
        section = "<Single Result>"
    else:
        section = "<Colony Forming Result>"

    m = None
    try:
        import re
        m = re.search(section + r".*", raw, flags=re.S)
    except Exception:
        pass
    if not m:
        raise ValueError("Could not find data section")

    block = m.group(0)
    dfs = []
    # split by "Well"
    for well_block in block.split("Well"):
        if not well_block.strip():
            continue
        lines = well_block.strip().splitlines()
        well_name = lines[0].strip().replace(":", "").replace("\t", "")
        if not well_name:
            continue
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[1:])))
        except Exception:
            continue
        df["Well"] = well_name
        dfs.append(df)

    if not dfs:
        raise ValueError("Parsed no data rows")

    all_df = pd.concat(dfs, ignore_index=True)

    # normalize column names
    all_df = all_df.rename(columns=lambda c: c.strip())

    # find confluency column flexibly
    confluency_col = None
    for c in all_df.columns:
        if "confluency" in c.lower():
            confluency_col = c
            break
    if not confluency_col:
        raise ValueError(f"No confluency column found in columns: {list(all_df.columns)}")

    # standardize name
    all_df = all_df.rename(columns={confluency_col: "Confluency"})

    # Convert time
    if "Time" in all_df.columns:
        all_df["Time"] = pd.to_datetime(all_df["Time"], errors="coerce")

    all_df["PlateType"] = plate_type
    return all_df

# ---------------- Plotting ----------------
def plot_plate(df: pd.DataFrame, timepoint: int, vmin: float, vmax: float,
               cmap_name="Reds", dpi=220):
    """Plot one timepoint of a plate as heatmap-like wells."""
    plate_type = df["PlateType"].iloc[0]
    rows, cols = plate_dims.get(plate_type, (8, 12))

    wells = sorted(df["Well"].unique())
    wells_this_tp = (
        df.groupby("Well").nth(timepoint, dropna="any").reset_index()
    )
    if wells_this_tp.empty:
        raise ValueError(f"No data for timepoint {timepoint}")

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_aspect("equal")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # draw wells
    for i, well in enumerate(sorted(df["Well"].unique())):
        row = i // cols
        col = i % cols
        rec = wells_this_tp[wells_this_tp["Well"] == well]
        if rec.empty or pd.isna(rec["Confluency"].iloc[0]) or rec["Confluency"].iloc[0] < 0:
            color = "black"
            val = None
        else:
            val = rec["Confluency"].iloc[0]
            color = cmap(norm(val))
        circ = patches.Circle((col+0.5, rows-row-0.5), 0.4,
                              facecolor=color, edgecolor="black")
        ax.add_patch(circ)
        if val is not None:
            ax.text(col+0.5, rows-row-0.5, f"{val:.0f}%",
                    ha="center", va="center", fontsize=8, color="white")

    return fig

# ---------------- Streamlit App ----------------
st.title("📊 Well Plate Confluency Heatmapper")

uploaded = st.file_uploader("Upload CSV export", type=["csv"])
if uploaded:
    try:
        df = parse_file(uploaded)
        st.success(f"Parsed {len(df)} rows from {df['PlateType'].iloc[0]}")

        max_tp = df.groupby("Well").size().max() - 1
        tp = st.slider("Timepoint index", 0, int(max_tp), 0)
        target_conf = st.number_input("Target confluency", value=100, min_value=1)

        fig = plot_plate(df, tp, vmin=0, vmax=target_conf)
        st.pyplot(fig)

        # download current image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220)
        buf.seek(0)
        st.download_button("Download current PNG", buf,
                           file_name=f"plate_tp{tp}.png", mime="image/png")

        # download all
        if st.button("Download all timepoints as ZIP"):
            tmpbuf = io.BytesIO()
            with zipfile.ZipFile(tmpbuf, "w") as zf:
                for t in range(int(max_tp)+1):
                    fig = plot_plate(df, t, vmin=0, vmax=target_conf)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", dpi=220)
                    img_buf.seek(0)
                    zf.writestr(f"plate_tp{t}.png", img_buf.read())
                    plt.close(fig)
            tmpbuf.seek(0)
            st.download_button("Download ZIP", tmpbuf,
                               file_name="all_timepoints.zip", mime="application/zip")

    except Exception as e:
        st.error(f"Could not parse file: {e}")
