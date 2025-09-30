import io, os, tempfile, zipfile, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# ---------------------------
# Utility: detect plate type
# ---------------------------
def detect_plate_type(df_raw):
    """Look for plate type in the <vessel Type> section or infer from well names."""
    text = "\n".join(df_raw.iloc[:,0].astype(str))
    # Direct match
    if "6well" in text.lower():
        return "6well"
    if "12well" in text.lower():
        return "12well"
    if "24well" in text.lower():
        return "24well"
    if "96well" in text.lower():
        return "96well"
    # Fallback: infer from well names
    wells = re.findall(r"Well[A-H]\d{1,2}", text)  # e.g., WellA1
    if wells:
        return "96well"
    wells_num = re.findall(r"Well\d+", text)  # e.g., Well1
    if wells_num:
        if len(set(wells_num)) == 6:
            return "6well"
        elif len(set(wells_num)) == 12:
            return "12well"
        elif len(set(wells_num)) == 24:
            return "24well"
    return "unknown"

# ---------------------------
# Parse CSV
# ---------------------------
def parse_cm30_csv(uploaded_file):
    try:
        df_raw = pd.read_csv(uploaded_file, header=None)
        plate_type = detect_plate_type(df_raw)

        # Find where results section starts
        result_line = df_raw[0].str.contains(r"<.*Result>", na=False)
        if not result_line.any():
            raise ValueError("Could not find results section in file.")
        start_idx = result_line.idxmax() + 1  # data starts after header

        # Extract well data
        df = df_raw.iloc[start_idx:].dropna(how="all")
        # Expect: WellName in col0, then Passage#, Time, Confluency, Count
        df.columns = ["Well", "Passage", "Time", "Confluency", "Count"]

        # Clean numeric cols
        df["Confluency"] = pd.to_numeric(df["Confluency"], errors="coerce")
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce")

        return plate_type, df
    except Exception as e:
        raise ValueError(f"Could not parse file: {e}")

# ---------------------------
# Heatmap plotting
# ---------------------------
def plot_heatmap(plate_type, df, cmap_min="#0000ff", cmap_max="#ff0000"):
    plate_dims = {
        "6well": (2, 3),
        "12well": (3, 4),
        "24well": (4, 6),
        "96well": (8, 12)
    }
    if plate_type not in plate_dims:
        raise ValueError(f"Unsupported plate type: {plate_type}")

    rows, cols = plate_dims[plate_type]
    # Take last confluency value for each well
    summary = df.groupby("Well")["Confluency"].last()

    # Normalize values
    norm = mcolors.Normalize(vmin=summary.min(skipna=True), vmax=summary.max(skipna=True))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", [cmap_min, cmap_max])

    fig, ax = plt.subplots(figsize=(cols, rows))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis("off")

    wells = sorted(summary.index)
    for i, well in enumerate(wells):
        r = i // cols
        c = i % cols
        val = summary[well]
        color = cmap(norm(val)) if pd.notnull(val) else "lightgray"
        rect = patches.Rectangle((c, rows-1-r), 1, 1, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(c+0.5, rows-1-r+0.5, f"{val:.1f}" if pd.notnull(val) else "NA",
                ha="center", va="center", fontsize=8, color="white")

    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CM30 Heatmapper (Iteration 1)")

uploaded_file = st.file_uploader("Upload CM30 CSV file", type=["csv"])
cmap_min = st.color_picker("Choose color for low values (0%)", "#0000ff")
cmap_max = st.color_picker("Choose color for high values (100%)", "#ff0000")

if uploaded_file:
    try:
        plate_type, df = parse_cm30_csv(uploaded_file)
        st.success(f"Detected plate type: {plate_type}")
        fig = plot_heatmap(plate_type, df, cmap_min, cmap_max)
        st.pyplot(fig)

        # Download option
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
        with open(tmpfile.name, "rb") as f:
            st.download_button("Download Heatmap as PNG", f, file_name="heatmap.png")
        os.unlink(tmpfile.name)
    except Exception as e:
        st.error(str(e))


