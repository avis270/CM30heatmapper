import io, os, tempfile, zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for packaging/headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import streamlit as st

# Plate layouts: row letters × column counts
PLATE_LAYOUTS = {
    "6well":   (["A", "B"], 3),
    "12well":  (["A", "B", "C"], 4),
    "24well":  (["A", "B", "C", "D"], 6),
    "96well":  (list("ABCDEFGH"), 12),
}

def parse_cm30_file(uploaded_file):
    """Parse CM30 CSV export into a tidy dataframe with Well + Confluency."""
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    # Detect plate type
    plate_type = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Plate"):
            parts = line.split(",")
            if len(parts) >= 6:
                plate_type = parts[5].strip()
                break
    if plate_type not in PLATE_LAYOUTS:
        raise ValueError(f"Unsupported or unknown plate type: {plate_type}")

    # Find where results start
    result_idx = None
    result_type = None
    for i, line in enumerate(lines):
        if line.strip() == "<Single Result>":
            result_idx, result_type = i, "single"
            break
        if line.strip() == "<Colony Forming Result>":
            result_idx, result_type = i, "colony"
            break
    if result_idx is None:
        raise ValueError("No results section (<Single Result> or <Colony Forming Result>) found.")

    # Parse section after results
    data = {}
    well_name, collecting = None, False
    for line in lines[result_idx+1:]:
        if line.startswith("Well"):
            well_name = line.strip()
            data[well_name] = []
            collecting = True
            continue
        if collecting and line.strip() == "":
            well_name, collecting = None, False
            continue
        if collecting and well_name:
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    conf = float(parts[2])
                    data[well_name].append(conf)
                except ValueError:
                    continue

    # Aggregate (mean confluency per well)
    df = pd.DataFrame([
        {"Well": well, "Confluency": pd.Series(vals).mean()}
        for well, vals in data.items()
    ])

    return df, plate_type


def generate_plate_layout(plate_type, data_df):
    """Return dataframe with all wells for given plate_type merged with data_df."""
    rows, cols = PLATE_LAYOUTS[plate_type]
    wells = [f"{r}{c}" for r in rows for c in range(1, cols+1)]
    full_df = pd.DataFrame({"Well": wells})
    merged = full_df.merge(data_df, on="Well", how="left")
    return merged, rows, cols


def plot_plate_heatmap(plate_df, rows, cols, vmin=0, vmax=100):
    """Render heatmap of plate data with missing wells in black."""
    fig, ax = plt.subplots(figsize=(cols, len(rows)))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, len(rows))
    ax.axis("off")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    for i, row in enumerate(rows):
        for j in range(1, cols+1):
            well = f"{row}{j}"
            val = plate_df.loc[plate_df["Well"] == well, "Confluency"].values
            color = "black"
            if len(val) > 0 and pd.notna(val[0]):
                color = cmap(norm(val[0]))
            rect = patches.Rectangle((j-1, len(rows)-i-1), 1, 1,
                                     facecolor=color, edgecolor="white")
            ax.add_patch(rect)
            ax.text(j-0.5, len(rows)-i-0.5, well,
                    ha="center", va="center", color="white", fontsize=8)

    return fig


# ---------------- Streamlit UI ----------------
st.title("CM30 Heatmap Viewer")

uploaded_file = st.file_uploader("Upload a CM30 CSV file", type=["csv"])
if uploaded_file:
    try:
        df, plate_type = parse_cm30_file(uploaded_file)
        st.success(f"Detected plate type: {plate_type}")

        plate_df, rows, cols = generate_plate_layout(plate_type, df)

        vmin = st.number_input("Minimum value (color scale)", 0.0, 100.0, 0.0)
        vmax = st.number_input("Maximum value (color scale)", 0.0, 100.0, 100.0)

        fig = plot_plate_heatmap(plate_df, rows, cols, vmin, vmax)
        st.pyplot(fig)

        # Download option
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        st.download_button("Download heatmap as PNG",
                           data=buf.getvalue(),
                           file_name=f"heatmap_{plate_type}.png",
                           mime="image/png")
    except Exception as e:
        st.error(f"Could not parse file: {e}")
