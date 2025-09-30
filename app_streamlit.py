import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# ---------- Plate type detection ----------
def detect_plate_type(raw: str) -> str:
    """Detect plate type by parsing the <vessel Type> section and extracting VesselName."""
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("<vessel Type>"):
            # Header should be the next line, data row after that
            if i + 2 < len(lines):
                header = lines[i+1].split(",")
                values = lines[i+2].split(",")
                if "VesselName" in header:
                    idx = header.index("VesselName")
                    if idx < len(values):
                        return values[idx].strip().lower()
    return "96well"  # fallback

# Mapping of plate type to (rows, cols)
plate_dims = {
    "6well": (2, 3),
    "12well": (3, 4),
    "24well": (4, 6),
    "96well": (8, 12),
}

# ---------- File parsing ----------
def parse_file(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    plate_type = detect_plate_type(raw)

    # find all well data sections
    dfs = []
    lines = raw.splitlines()
    current_well = None
    buffer = []
    for line in lines:
        if line.startswith("Well") and not line.startswith("Well List"):
            if current_well and buffer:
                dfs.append(pd.read_csv(io.StringIO("\n".join(buffer))))
                buffer = []
            current_well = line.split(",")[0].strip()
        if current_well:
            buffer.append(line)
    if current_well and buffer:
        dfs.append(pd.read_csv(io.StringIO("\n".join(buffer))))

    if not dfs:
        raise ValueError("Parsed no data rows")

    # tag each DF with well name
    out = []
    for df, (well,) in zip(dfs, [(l.split(",")[0].strip(),) for l in lines if l.startswith("Well") and not l.startswith("Well List")]):
        df["Well"] = well
        out.append(df)

    all_df = pd.concat(out, ignore_index=True)

    # normalize column names
    all_df = all_df.rename(columns=lambda c: c.strip())
    if "Estimatevalue(Confluency)" not in all_df.columns:
        raise ValueError("No confluency column found")
    return plate_type, all_df

# ---------- Heatmap drawing ----------
def plot_plate(plate_type, df, timepoint, vmin=0, vmax=100, cmap="Reds"):
    nrows, ncols = plate_dims.get(plate_type, (8, 12))
    wells = sorted(df["Well"].unique())

    # pivot by time
    df_time = df[df["Time"] == timepoint]

    fig, ax = plt.subplots(figsize=(ncols, nrows))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.axis("off")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    for i, well in enumerate(wells):
        row = i // ncols
        col = i % ncols
        val = df_time[df_time["Well"] == well]["Estimatevalue(Confluency)"]
        if len(val) > 0:
            color = cmap(norm(float(val.iloc[0])))
        else:
            color = "black"
        circle = plt.Circle((col+0.5, nrows-row-0.5), 0.4,
                            facecolor=color, edgecolor="black")
        ax.add_patch(circle)
        ax.text(col+0.5, nrows-row-0.5, well.replace("Well", ""),
                ha="center", va="center", fontsize=6, color="white")

    return fig

# ---------- Streamlit UI ----------
st.title("📊 Plate Heatmap Viewer")

uploaded = st.file_uploader("Upload CSV", type="csv")
if uploaded:
    try:
        plate_type, df = parse_file(uploaded)
        st.success(f"Detected plate type: **{plate_type}**")

        timepoints = sorted(df["Time"].unique())
        timepoint = st.selectbox("Select timepoint", timepoints)

        vmin = st.number_input("Min % (color scale)", 0, 100, 0)
        vmax = st.number_input("Max % (color scale)", 0, 100, 100)

        if st.button("Generate Heatmap"):
            fig = plot_plate(plate_type, df, timepoint, vmin=vmin, vmax=vmax)
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=220)
            buf.seek(0)
            st.download_button("Download PNG", buf, file_name="heatmap.png", mime="image/png")

    except Exception as e:
        st.error(f"Could not parse file: {e}")
